"""The aircraft screen: what is in the sky above you, as a list.

This is the second screen in the app that reports what a signal *says* rather
than what it looks like, and the first that is a place the radio goes rather
than a decoder hung off the audio path. Every airliner overhead broadcasts its
identity, altitude, position and speed twice a second on 1090 MHz, in the
clear, to nobody in particular - so the screen is a list that fills itself in,
with no tuning, no mode and no bandwidth to get wrong.

Like the other views this owns no threads and never touches the device. It
asks the engine to start receiving, then polls a mailbox on a timer. The
engine takes the radio to 1090 MHz for the duration and puts it back
afterwards, exactly as a scan does, because there is nothing to listen to
there: Mode S is a 1 Mbit/s data burst and audio would only hiss.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.engine import Engine
from ..decode.adsb import AdsbState, Aircraft
from .levels import Level
from .widgets.aircraftcard import AircraftCard
from .widgets.planemap import PlaneMap

REFRESH_HZ = 5

# How much of the screen the list gets when an aircraft is picked out on the
# map. A splitter opens at the size hints of its panes, and a list of cards
# hints at almost nothing until there are cards in it - so the pane that the
# click is *about* was a 60 px strip showing the top of one row. A third is
# enough for two aircraft and the beginning of a third, which is the answer
# to "what does this one say" without taking the map away from under the dot
# that was just clicked.
#
# A third of the *screen*, not of the splitter: the heading and the button
# above it are a fifth of the window, so a third of what is left is a quarter
# of what somebody is looking at, and this is a promise about what they see.
LIST_SHARE = 1.0 / 3.0
# ...but never so much of the splitter that the map stops being a map. The
# click that asked for this landed on it.
MAX_LIST_SHARE = 0.6
# What the screen opens at, before anybody has dragged the handle. The map is
# the better of the two on arrival - it is the one that says whether the
# aerial is working at all - but not by so much that the list should be a
# strip.
OPENING_SHARES = (0.6, 0.4)

VIEW_STYLE = """
QWidget#aircraft { background: #0b0e13; }
QLabel#heading { color: #e6edf3; font-size: 19px; font-weight: 600; }
QLabel#subheading { color: #8b98a5; font-size: 12px; }
QLabel#status { color: #8b98a5; font-size: 12px; }
QLabel#empty { color: #6d7b89; font-size: 13px; }
QPushButton#receive {
    background: #5ad1ff; color: #0b0e13; border: none;
    border-radius: 4px; padding: 8px 24px; font-weight: 600; font-size: 13px;
}
QPushButton#receive:hover { background: #7cdcff; }
/* Both the scroll area and its viewport have to be named - see
   `discover_view.py` for what happens when only one of them is. */
QScrollArea { border: none; background: #0b0e13; }
QWidget#planeList { background: #0b0e13; }
QSplitter::handle { background: #161c25; height: 3px; }
"""

WAITING = (
    "Nothing heard yet. Aircraft are heard best with the aerial by a window "
    "or outdoors, with a view of the sky - indoors, in the middle of a "
    "building, there may be nothing to hear at all."
)
IDLE = (
    "Press Listen for aircraft. Anything overhead shows up below as it is "
    "heard."
)


class AircraftView(QWidget):
    """Receive 1090 MHz and list the aircraft that are talking."""

    def __init__(
        self,
        engine: Engine,
        level: Level = Level.SIMPLE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.level = level
        self._cards: dict[int, AircraftCard] = {}
        self._order: tuple[int, ...] = ()
        # The aircraft the user has picked out, on the map or in the list.
        # One selection shared by both panes: two would drift apart within
        # seconds and neither would be wrong on its own.
        self._selected: int | None = None

        self._build()
        self.set_level(level)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(int(1000 / REFRESH_HZ))

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.setObjectName("aircraft")
        self.setStyleSheet(VIEW_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        heading = QLabel("Aircraft overhead")
        heading.setObjectName("heading")
        outer.addWidget(heading)

        subheading = QLabel(
            "Aircraft broadcast who they are, how high they are and where "
            "they are, on 1090 MHz. They are drawn on the map as they report "
            "a position, coloured by height. The radio listens there while "
            "this screen is running, so the station you were tuned to goes "
            "quiet until you stop."
        )
        subheading.setObjectName("subheading")
        subheading.setWordWrap(True)
        outer.addWidget(subheading)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.receive_button = QPushButton("Listen for aircraft")
        self.receive_button.setObjectName("receive")
        self.receive_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.receive_button.clicked.connect(self._receive_clicked)
        row.addWidget(self.receive_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.status = QLabel("")
        self.status.setObjectName("status")
        outer.addWidget(self.status)

        self.map = PlaneMap()
        self.map.setToolTip(
            "Everything that has said where it is, drawn to fit. Click an "
            "aircraft to find it in the list below. There is no street map "
            "underneath because nothing here is downloaded - what is on "
            "screen is what the aerial received."
        )
        self.map.selectionChanged.connect(self._map_selected)

        self.list_area = QScrollArea()
        self.list_area.setWidgetResizable(True)
        self.list_area.viewport().setAutoFillBackground(True)
        palette = self.list_area.viewport().palette()
        palette.setColor(self.list_area.viewport().backgroundRole(), QColor("#0b0e13"))
        self.list_area.viewport().setPalette(palette)
        holder = QWidget()
        holder.setObjectName("planeList")
        self.list_layout = QVBoxLayout(holder)
        self.list_layout.setContentsMargins(0, 0, 8, 0)
        self.list_layout.setSpacing(8)
        self.empty = QLabel(IDLE)
        self.empty.setObjectName("empty")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list_layout.addWidget(self.empty)
        self.list_layout.addStretch(1)
        self.list_area.setWidget(holder)

        # A splitter rather than a fixed share: the map is the better view
        # when several aircraft are moving and the list is the better one
        # when a single distant aircraft is being coaxed in, and which of
        # those is happening is not something this screen can know.
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(self.map)
        self.split.addWidget(self.list_area)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self.split.setChildrenCollapsible(True)
        # Stretch factors decide how *extra* room is shared out; the opening
        # sizes come from the panes' own hints, and an empty list hints at
        # nothing. Stated outright, or the screen opens with the list as a
        # strip and stays that way until somebody finds the handle.
        self.split.setSizes([int(share * 1000) for share in OPENING_SHARES])
        outer.addWidget(self.split, 1)

    # -- level -------------------------------------------------------------

    def set_level(self, level: Level) -> None:
        self.level = level
        for card in self._cards.values():
            card.set_level(level)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        """Leaving the screen gives the radio back.

        Not merely a paused display: the tuner is 1090 MHz away from anything
        anybody wanted to hear, and a user who walked back to the listening
        screen to find it silent would have no way of knowing why. Reception
        is what this screen *is*, so it ends with it.
        """
        self._timer.stop()
        if self.engine.receiving_adsb:
            self.engine.stop_adsb()

    # -- receiving ---------------------------------------------------------

    def _receive_clicked(self) -> None:
        if self.engine.receiving_adsb:
            self.engine.stop_adsb()
            return
        self._clear()
        self.status.setText("Tuning to 1090 MHz...")
        self.engine.start_adsb()

    def _tick(self) -> None:
        # Driven by what the engine is actually doing rather than by what the
        # button press assumed it would do: reception can refuse to start -
        # a scan is running, the window is too narrow - and the screen has to
        # agree with the radio either way.
        receiving = self.engine.receiving_adsb
        self.receive_button.setText(
            "Stop" if receiving else "Listen for aircraft"
        )
        state = self.engine.adsb_update()
        if state is None:
            return
        self._show(state.aircraft)
        self.map.show_aircraft(state.aircraft)
        self.status.setText(self._summary(state, receiving))
        self.empty.setText(WAITING if receiving else IDLE)

    def _summary(self, state: AdsbState, receiving: bool) -> str:
        """One line of what the receiver is doing, in the user's terms.

        The message rate is worth showing even at Simple: it is the difference
        between "the aerial is in the wrong place" and "the sky is quiet right
        now", and it moves within a second or two of an aerial being moved,
        which makes it the one number that helps somebody get this working.
        """
        if not receiving and not state.aircraft:
            return ""
        count = len(state.aircraft)
        where = "Listening on 1090 MHz" if receiving else "Stopped"
        planes = "1 aircraft" if count == 1 else f"{count} aircraft"
        parts = [where, planes, f"{state.rate_per_minute:.0f} messages a minute"]
        # Which of them are on the map, because the difference is a question
        # the screen will otherwise be asked. An aircraft is heard several
        # times before it has sent both halves of a position, and one that
        # never sends a position at all is a normal thing to hear.
        placed = self.map.plotted
        if placed < count:
            parts.append(f"{placed} on the map")
        if self.level >= Level.EXPERT and state.bad:
            # Bursts that looked like a message and failed their checkword.
            # A high count next to a low message rate is noise being tried
            # rather than aircraft being missed, which is worth telling
            # somebody who knows what to do about it.
            parts.append(f"{state.bad} unreadable")
        return "   ·   ".join(parts)

    # -- selection ---------------------------------------------------------

    def _map_selected(self, icao: object) -> None:
        """A symbol on the map was clicked, or the map was clicked past one."""
        self._apply_selection(icao if icao is None else int(icao), reveal=True)

    def _card_chosen(self, icao: object) -> None:
        """A row was clicked. The other half of the same selection.

        Clicking a row does not scroll it - it is already under the pointer -
        but it does light up the symbol, which is the answer to "which of
        these dots is the one I am reading about".
        """
        self._apply_selection(int(icao), reveal=False)

    def _apply_selection(self, icao: int | None, reveal: bool) -> None:
        self._selected = icao
        for address, card in self._cards.items():
            card.set_picked(address == icao)
        self.map.set_selected(icao)
        if reveal and icao is not None:
            self._reveal(icao)

    def _reveal(self, icao: int) -> None:
        """Bring the selected card into view, giving the list room to show it.

        Not only when the list has been dragged shut: scrolling a pane too
        short to hold a card is the same failure as not scrolling at all, and
        a click on the map is a request to read what that aircraft says. So
        the list is given `LIST_SHARE` of the screen whenever it has less,
        and left alone whenever it has more - somebody who dragged it larger
        does not want it snapped back to a third.
        """
        card = self._cards.get(icao)
        if card is None:
            return
        sizes = self.split.sizes()
        total = sum(sizes)
        if len(sizes) == 2 and total > 0:
            wanted = min(
                int(round(self.height() * LIST_SHARE)),
                int(round(total * MAX_LIST_SHARE)),
            )
            if sizes[1] < wanted:
                self.split.setSizes([total - wanted, wanted])
        # After the event loop has laid the list out: a card created on this
        # same tick has no position yet, and scrolling to it would scroll to
        # wherever the layout last had something.
        QTimer.singleShot(0, lambda: self._scroll_to(icao))

    def _scroll_to(self, icao: int) -> None:
        # Looked up again rather than captured: a deferred call holding a
        # widget the next snapshot has already deleted is an aircraft going
        # out of range taking the app with it.
        card = self._cards.get(icao)
        if card is not None:
            self.list_area.ensureWidgetVisible(card, 0, 40)

    # -- the list ----------------------------------------------------------

    def _show(self, aircraft: tuple[Aircraft, ...]) -> None:
        """Update the list in place, adding and removing only what changed.

        Cards are updated rather than rebuilt: an aircraft reports twice a
        second, and throwing the widget away each time would flicker and fight
        the scrollbar. Only when the *set* of aircraft changes does anything
        move in the layout.
        """
        seen = {plane.icao for plane in aircraft}
        for icao in [key for key in self._cards if key not in seen]:
            card = self._cards.pop(icao)
            self.list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()

        for plane in aircraft:
            card = self._cards.get(plane.icao)
            if card is None:
                card = AircraftCard(plane, level=self.level)
                card.chosen.connect(self._card_chosen)
                card.set_picked(plane.icao == self._selected)
                self._cards[plane.icao] = card
                self.list_layout.insertWidget(self.list_layout.count() - 1, card)
            else:
                card.update_from(plane)

        # The engine hands them back most recently heard first. Following that
        # order exactly would have rows swapping places several times a second
        # as messages arrive, so the list is only reordered when the set of
        # aircraft has actually changed - a new one appears at the position it
        # was reported at, and the rest stay where the eye left them.
        order = tuple(plane.icao for plane in aircraft)
        if set(order) != set(self._order):
            # The placeholder is put back at the top first, so the layout is
            # left in one known arrangement rather than one that depends on
            # how many times the set has changed.
            self.list_layout.insertWidget(0, self.empty)
            for position, icao in enumerate(order):
                self.list_layout.insertWidget(position + 1, self._cards[icao])
        self._order = order
        self.empty.setVisible(not aircraft)

    def _clear(self) -> None:
        for card in self._cards.values():
            self.list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._order = ()
        self._selected = None
        self.empty.setVisible(True)
        self.map.clear()


__all__ = ["AircraftView"]
