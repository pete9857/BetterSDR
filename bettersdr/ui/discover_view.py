"""The discovery screen: what is on the air around you, as a list.

This is the screen the project exists for. Every other SDR application opens
on a blank spectrum and expects you to already know which frequency is worth
visiting; this one sweeps, and shows what it found in plain English, the way a
Wi-Fi picker shows networks. The mental model is scan and browse, not tune and
configure.

Like `listen_view.py` this is a *view*: it owns no threads and never touches
the device. It asks the engine to scan, then polls a mailbox on a timer for
progress and results. The engine runs the sweep on the DSP thread it already
owns, so scanning and listening take turns rather than competing for the same
sample stream.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.bookmarks import BookmarkStore, from_signal
from ..core.engine import Engine, ScanUpdate
from ..core.settings import Settings
from ..scan import bandplan
from ..scan.classifier import Signal, format_frequency
from ..scan.detector import SENSITIVITY_DB
from . import results
from .levels import Level
from .widgets.help import HelpLabel, label_for
from .widgets.icons import glyph
from .widgets.signalcard import SignalCard

REFRESH_HZ = 20

# Three positions, worded as what they do rather than as decibels. The dB
# value only appears at Expert, where it means something to the reader.
SENSITIVITIES = (
    ("Strong signals only", "low"),
    ("Normal", "normal"),
    ("Include weak signals", "high"),
)

VIEW_STYLE = """
QWidget#discover { background: #0b0e13; }
QLabel#heading { color: #e6edf3; font-size: 19px; font-weight: 600; }
QLabel#subheading { color: #8b98a5; font-size: 12px; }
QLabel#status { color: #8b98a5; font-size: 12px; }
QLabel#empty { color: #6d7b89; font-size: 13px; }
QPushButton#band {
    background: #10151c; color: #8b98a5;
    border: 1px solid #2b323b; border-radius: 13px; padding: 5px 14px;
}
QPushButton#band:hover { border-color: #3d4650; color: #e6edf3; }
QPushButton#band:checked { background: #5ad1ff; color: #0b0e13; font-weight: 600; }
QPushButton#scan {
    background: #5ad1ff; color: #0b0e13; border: none;
    border-radius: 4px; padding: 8px 24px; font-weight: 600; font-size: 13px;
}
QPushButton#scan:hover { background: #7cdcff; }
QProgressBar {
    background: #10151c; border: 1px solid #1d232b; border-radius: 3px;
    height: 6px; text-align: center; color: transparent;
}
QProgressBar::chunk { background: #5ad1ff; border-radius: 3px; }
/* The viewport is a child widget of its own, and styling only the
   QScrollArea leaves it painting the default light background behind every
   card. Both have to be named. */
QScrollArea { border: none; background: #0b0e13; }
QWidget#cardList { background: #0b0e13; }
QComboBox {
    background: #161b22; color: #e6edf3;
    border: 1px solid #2b323b; border-radius: 3px; padding: 3px 8px;
}
QLabel#filterLabel { color: #6d7b89; font-size: 11px; }
/* Deliberately not the band chip's bright accent when on. A band chip marks
   one exclusive choice out of many; these are all on to begin with, and a row
   of cyan across the screen would read as a selection nobody made. What has
   to be visible here is the *off* state. */
QPushButton#kind {
    background: #161b22; color: #e6edf3;
    border: 1px solid #2b323b; border-radius: 11px;
    padding: 3px 11px; font-size: 11px;
}
QPushButton#kind:hover { border-color: #3d4650; }
QPushButton#kind:!checked {
    background: #0e1218; color: #4d5865; border-color: #1d232b;
}
QPushButton#showAll {
    background: transparent; color: #5ad1ff; border: none;
    padding: 3px 6px; font-size: 11px;
}
QPushButton#showAll:hover { color: #7cdcff; }
"""


class DiscoverView(QWidget):
    """Sweep a band and list what is transmitting in it."""

    listenRequested = QtSignal(object)
    # What this screen is currently *showing*, in the order it is showing it
    # and with the hidden kinds already taken out. Emitted so the listening
    # screen's step buttons can walk the same list the user is looking at;
    # nothing else here needs it, and the listening screen must not have to
    # reach into this one to find it.
    resultsChanged = QtSignal(object)
    # Somebody clicked the name of a control wanting to know what it means.
    # Same signal, same meaning and the same journey as the listening
    # screen's: the window decides where it goes, this screen does not know.
    helpRequested = QtSignal(str)

    def __init__(
        self,
        engine: Engine,
        level: Level = Level.SIMPLE,
        bookmarks: BookmarkStore | None = None,
        settings: Settings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.level = level
        self.bookmarks = bookmarks if bookmarks is not None else BookmarkStore()
        self.settings = settings
        self._levelled: list[tuple[QWidget, Level]] = []
        self._cards: list[SignalCard] = []
        self._chips: dict[str, QPushButton] = {}
        self._shown: tuple = ()
        self._kinds: frozenset[str] = frozenset()
        self._signals: tuple[Signal, ...] = ()
        # The listed subset, in listed order - what `resultsChanged` last
        # carried, and the answer to `listed` for anything asking later.
        self._listed: tuple[Signal, ...] = ()
        self._sort = results.DEFAULT_SORT
        self._hidden: frozenset[str] = frozenset()
        self._band: bandplan.Band | None = None
        self._running = False

        self._build()
        self._restore()
        self.set_level(level)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(int(1000 / REFRESH_HZ))

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.setObjectName("discover")
        self.setStyleSheet(VIEW_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        heading = QLabel(
            "What's on the air around you? Pick a band and press Scan to find out!"
        )
        heading.setObjectName("heading")
        outer.addWidget(heading)

        #subheading = QLabel(
        #    "Pick a band and press Scan. Anything transmitting shows up below."
        #)
        #subheading.setObjectName("subheading")
        #outer.addWidget(subheading)

        outer.addLayout(self._band_chips())
        outer.addLayout(self._actions())

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setObjectName("status")
        outer.addWidget(self.status)

        outer.addWidget(self._filters())

        self.list_area = QScrollArea()
        self.list_area.setWidgetResizable(True)
        # Deliberately *not* a stylesheet on the viewport. Setting one there
        # drags every descendant through the stylesheet style, and QLabel is a
        # QFrame, so each one starts painting a frame it never asked for - a
        # box around every card's frequency line.
        self.list_area.viewport().setAutoFillBackground(True)
        palette = self.list_area.viewport().palette()
        palette.setColor(self.list_area.viewport().backgroundRole(), QColor("#0b0e13"))
        self.list_area.viewport().setPalette(palette)
        holder = QWidget()
        holder.setObjectName("cardList")
        self.list_layout = QVBoxLayout(holder)
        self.list_layout.setContentsMargins(0, 0, 8, 0)
        self.list_layout.setSpacing(8)
        self.empty = QLabel(
            "Nothing found yet. Choose a band above and press Scan."
        )
        self.empty.setObjectName("empty")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list_layout.addWidget(self.empty)
        self.list_layout.addStretch(1)
        self.list_area.setWidget(holder)
        outer.addWidget(self.list_area, 1)

    def _band_chips(self) -> QHBoxLayout:
        """One button per scannable band, straight out of the band plan."""
        row = QHBoxLayout()
        row.setSpacing(6)
        self._bands = QButtonGroup(self)
        self._bands.setExclusive(True)
        for index, band in enumerate(bandplan.scannable()):
            button = QPushButton(f"{glyph(band.icon)}  {band.name}")
            button.setObjectName("band")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(
                f"{format_frequency(band.start_hz)} to "
                f"{format_frequency(band.end_hz)} - {band.description}"
            )
            self._bands.addButton(button, index)
            row.addWidget(button)
            if band.name == "FM Radio":
                button.setChecked(True)
                self._band = band
        self._bands.idClicked.connect(self._band_chosen)
        row.addStretch(1)
        return row

    def _actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.scan_button = QPushButton("Scan")
        self.scan_button.setObjectName("scan")
        self.scan_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_button.clicked.connect(self._scan_clicked)
        row.addWidget(self.scan_button)

        sensitivity_label = self._help_label("Sensitivity", topic="sensitivity")
        self.sensitivity = QComboBox()
        for label, key in SENSITIVITIES:
            self.sensitivity.addItem(label, key)
        self.sensitivity.setCurrentIndex(1)
        self.sensitivity.setToolTip(
            "How far above the background noise something has to be before it "
            "counts as a signal."
        )
        row.addWidget(sensitivity_label)
        row.addWidget(self.sensitivity)
        self._levelled.append((sensitivity_label, Level.STANDARD))
        self._levelled.append((self.sensitivity, Level.STANDARD))

        # Standard, not Simple. Strongest-first is the right answer for
        # somebody who does not yet know what they are looking for, and a
        # beginner who wants the airband tidied has the type chips below,
        # which say what they do without having to be understood first.
        sort_label = self._help_label("Order", topic="sort-order")
        self.sort = QComboBox()
        for label, key in results.SORTS:
            self.sort.addItem(label, key)
        self.sort.setToolTip(
            "What goes at the top of the list. Sorting does not change what "
            "was found."
        )
        self.sort.currentIndexChanged.connect(self._sort_chosen)
        row.addWidget(sort_label)
        row.addWidget(self.sort)
        self._levelled.append((sort_label, Level.STANDARD))
        self._levelled.append((self.sort, Level.STANDARD))

        # The one-click route from "the app found these" to "these are mine",
        # which is the whole reason the frequency manager knows about
        # `Signal`: a saved scan result keeps the mode and bandwidth the
        # classifier chose, so recalling it later just works.
        self.save_found = QPushButton("Save these to my list")
        self.save_found.setObjectName("band")
        self.save_found.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_found.setEnabled(False)
        self.save_found.clicked.connect(self._save_found)
        row.addWidget(self.save_found)
        self._levelled.append((self.save_found, Level.STANDARD))

        row.addStretch(1)
        return row

    def _help_label(self, text: str, topic: str = "") -> QLabel:
        """A control caption that is also the way in to what it means.

        The two on this screen are worth having for the same reason the forty
        on the listening screen are: "Sensitivity" is a perfectly ordinary
        English word that means something quite specific here, and the moment
        somebody wonders which is the moment they are looking straight at it.
        """
        label = label_for(text, topic)
        if isinstance(label, HelpLabel):
            label.helpRequested.connect(self.helpRequested.emit)
        return label

    def _filters(self) -> QWidget:
        """The row of type chips, one per kind of thing this sweep found.

        Shown at every level, including Simple, and that is a deliberate
        exception to "controls appear as you ask for them". These are not a
        control: they are the list describing itself, in the same plain
        English as the cards, with the count of each kind on the chip. An
        indoor aerial fills the airband with 83 unmodulated carriers, and
        being able to put them aside in one click is the difference between a
        readable list and an unreadable one - which a beginner needs more
        than anybody, not less.
        """
        holder = QWidget()
        self._filter_row = QHBoxLayout(holder)
        self._filter_row.setContentsMargins(0, 0, 0, 0)
        self._filter_row.setSpacing(6)

        label = QLabel("Showing")
        label.setObjectName("filterLabel")
        self._filter_row.addWidget(label)

        self.show_all = QPushButton("Show everything")
        self.show_all.setObjectName("showAll")
        self.show_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_all.setVisible(False)
        self.show_all.clicked.connect(self._show_everything)
        self._filter_row.addWidget(self.show_all)
        self._filter_row.addStretch(1)

        self._filter_holder = holder
        holder.setVisible(False)
        return holder

    # -- ordering and filtering --------------------------------------------

    def _sort_chosen(self) -> None:
        self._sort = self.sort.currentData() or results.DEFAULT_SORT
        self._remember()
        self._relist()

    def _kind_toggled(self, label: str, shown: bool) -> None:
        hidden = set(self._hidden)
        if shown:
            hidden.discard(label)
        else:
            hidden.add(label)
        self._hidden = frozenset(hidden)
        self._remember()
        self._relist()

    def _show_everything(self) -> None:
        self._hidden = frozenset()
        self._remember()
        self._relist()

    def _relist(self) -> None:
        """Re-render the list the sweep last handed over, unchanged."""
        self._show(self._signals)

    def _restore(self) -> None:
        """Adopt the order and the filter this user left behind."""
        settings = self.settings
        if settings is None:
            return
        stored = str(settings["scan_sort"])
        if stored in results.SORT_KEYS:
            self._sort = stored
        index = self.sort.findData(self._sort)
        if index >= 0:
            # Blocked because setting the index emits, and the emission would
            # write the restored value straight back out again before the
            # window is even up.
            self.sort.blockSignals(True)
            self.sort.setCurrentIndex(index)
            self.sort.blockSignals(False)
        hidden = settings["hidden_kinds"]
        if isinstance(hidden, (list, tuple)):
            self._hidden = frozenset(str(label) for label in hidden)

    def _remember(self) -> None:
        settings = self.settings
        if settings is None:
            return
        settings.update(
            scan_sort=self._sort,
            hidden_kinds=sorted(self._hidden),
        )
        settings.save()

    def _update_filters(self, kinds: tuple[results.Kind, ...]) -> None:
        """Keep the chip row agreeing with what the sweep has found so far.

        Rebuilt only when the *set* of kinds changes; a chip whose count has
        merely gone up is relabelled where it stands. The set, not the order:
        the chips are ordered by how many each is holding, so a sweep's second
        pass overtaking one kind with another would otherwise swap two chips
        under the cursor of somebody reaching for one of them.
        """
        self._filter_holder.setVisible(bool(kinds))
        self.show_all.setVisible(any(not kind.shown for kind in kinds))
        present = frozenset(kind.label for kind in kinds)
        if present == self._kinds:
            for kind in kinds:
                self._chips[kind.label].setText(
                    f"{glyph(kind.icon)}  {kind.chip}"
                )
            return
        self._kinds = present
        self._clear_chips()
        for index, kind in enumerate(kinds):
            chip = QPushButton(f"{glyph(kind.icon)}  {kind.chip}")
            chip.setObjectName("kind")
            chip.setCheckable(True)
            chip.setChecked(kind.shown)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(
                f"{kind.count} of these were found. Click to hide them; the "
                f"chip stays here with the count on it."
            )
            # Connected after setChecked, so restoring a remembered filter
            # does not read as the user having just clicked it.
            chip.toggled.connect(
                lambda shown, label=kind.label: self._kind_toggled(label, shown)
            )
            self._filter_row.insertWidget(1 + index, chip)
            self._chips[kind.label] = chip

    def _clear_chips(self) -> None:
        for chip in self._chips.values():
            self._filter_row.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips.clear()

    def _save_found(self) -> None:
        """Add every signal currently listed to the saved frequencies."""
        signals = [card.signal for card in self._cards]
        if not signals:
            return
        group = self._band.name if self._band is not None else "Found by scanning"
        for signal in signals:
            self.bookmarks.add(from_signal(signal, group=group), replace_existing=False)
        self.bookmarks.save()
        self.save_found.setText(f"Saved {len(signals)}")

    # -- level -------------------------------------------------------------

    def set_level(self, level: Level) -> None:
        self.level = level
        for widget, minimum in self._levelled:
            widget.setVisible(level >= minimum)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    # -- what is on screen -------------------------------------------------

    @property
    def listed(self) -> tuple[Signal, ...]:
        """The cards currently on screen, in the order they are drawn."""
        return self._listed

    def _publish(self, listed: tuple[Signal, ...]) -> None:
        self._listed = listed
        self.resultsChanged.emit(listed)

    # -- scanning ----------------------------------------------------------

    def _band_chosen(self, index: int) -> None:
        bands = bandplan.scannable()
        if 0 <= index < len(bands):
            self._band = bands[index]

    def _scan_clicked(self) -> None:
        if self.engine.scanning:
            self.engine.stop_scan()
            return
        band = self._band
        if band is None:
            return
        self._clear()
        # The chips describe the sweep that is being thrown away, so they go
        # with it. What the user chose to hide does not: that is a standing
        # preference and it survives the band change, which is the whole
        # reason it is remembered on disk as well.
        self._clear_chips()
        self._kinds = frozenset()
        self._filter_holder.setVisible(False)
        self._signals = ()
        self._shown = ()
        self._publish(())
        self._update_empty()
        self._running = True
        self.progress.setValue(0)
        self.status.setText(f"Listening across the {band.name} band...")
        self.engine.start_scan(
            band.start_hz,
            band.end_hz,
            threshold_db=SENSITIVITY_DB[self.sensitivity.currentData()],
            # Some bands need a narrower window than the default - see
            # `frontend.safe_sample_rate`. The engine clamps this anyway; the
            # band plan is where the preference is allowed to be stated.
            sample_rate_hz=band.sample_rate_hz,
        )

    def _tick(self) -> None:
        update = self.engine.scan_update()
        if update is None:
            return
        # Driven by what the engine is actually doing rather than by what the
        # button press assumed it would do. A scan can end on its own, or fail
        # to start at all, and either way the screen has to agree with reality.
        scanning = self.engine.scanning
        self.progress.setVisible(scanning)
        self.scan_button.setText("Stop" if scanning else "Scan")
        self.progress.setValue(int(update.progress.fraction * 1000))
        self._show(update.signals)

        if scanning:
            self.status.setText(
                f"Listening around {format_frequency(update.progress.center_hz)}"
                f"  -  {len(update.signals)} found so far"
                f"{self._hidden_note(update.signals)}"
            )
        elif self._running:
            # Once, on the edge. The mailbox keeps handing back the final
            # update, and rewriting the summary twenty times a second would
            # stamp on anything else that wanted to say something.
            self._running = False
            self._finished(update)

    def _finished(self, update: ScanUpdate) -> None:
        band = self._band.name if self._band else "that range"
        count = len(update.signals)
        if count:
            self.status.setText(
                f"Found {count} in the {band} band"
                f"{self._hidden_note(update.signals)}. "
                f"Press Listen on any of them."
            )
        else:
            self.status.setText(
                f"Nothing found in the {band} band. Try turning the "
                f"sensitivity up, or check the aerial is plugged in."
            )

    # -- the list ----------------------------------------------------------

    def _show(self, signals: tuple[Signal, ...]) -> None:
        """Rebuild the list, but only when it has actually changed.

        Cards carry state - an expanded "What is this?" - and rebuilding them
        twenty times a second would slam it shut under the user's cursor. The
        signature is what is on screen, so an unchanged sweep leaves the list
        entirely alone. It covers the chosen order and the filter as well as
        the signals, because changing either changes the screen without a
        single new signal having arrived.
        """
        self._signals = tuple(signals)
        ordered = results.sort_signals(self._signals, self._sort)
        listed = results.visible(ordered, self._hidden)
        kinds = results.summarise(self._signals, self._hidden)
        signature = (
            self._sort,
            tuple((kind.label, kind.count, kind.shown) for kind in kinds),
            tuple((s.frequency_hz, s.label) for s in listed),
        )
        if signature == self._shown:
            return
        self._shown = signature
        self._publish(listed)

        self._update_filters(kinds)
        self._clear()
        for signal in listed:
            card = SignalCard(signal)
            card.listenRequested.connect(self.listenRequested.emit)
            self.list_layout.insertWidget(self.list_layout.count() - 1, card)
            self._cards.append(card)
        self._update_empty()
        # "Save these to my list" means what is on the screen, so a filtered
        # list saves the filtered set - and it is reset rather than left
        # saying "Saved 12" over a different one.
        self.save_found.setEnabled(bool(listed))
        self.save_found.setText("Save these to my list")

    def _update_empty(self) -> None:
        """An empty list has two quite different causes; say which one."""
        held = results.hidden_count(self._signals, self._hidden)
        if self._cards:
            self.empty.setVisible(False)
            return
        if held:
            self.empty.setText(
                f"All {held} found here are hidden. Turn one of the chips "
                f"above back on to see them."
            )
        else:
            self.empty.setText(
                "Nothing found yet. Choose a band above and press Scan."
            )
        self.empty.setVisible(True)

    def _hidden_note(self, signals: tuple[Signal, ...]) -> str:
        """", 83 hidden" - said out loud, every time, wherever a count is.

        A filter that persists between sittings has one real failure mode: a
        user meeting a short list weeks later and reading it as a sweep that
        found nothing. The count is on the chip, and it is here too.
        """
        held = results.hidden_count(signals, self._hidden)
        return f"  -  {held} hidden" if held else ""

    def _clear(self) -> None:
        for card in self._cards:
            self.list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self.empty.setVisible(True)


__all__ = ["DiscoverView"]
