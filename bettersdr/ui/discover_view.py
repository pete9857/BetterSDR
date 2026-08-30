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
from ..scan import monitor as mon
from ..scan.classifier import Signal, format_frequency
from ..scan.detector import SENSITIVITY_DB
from . import results
from .levels import Level
from .widgets.activitycard import ActivityCard
from .widgets.help import HelpButton, HelpLabel, label_for
from .widgets.icons import glyph
from .widgets.rangepicker import RangePicker, ranges_for
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
QPushButton#scan:disabled { background: #23303a; color: #4d5865; }
/* Deliberately not the accent Scan wears. They are two ways of asking the
   same screen a question, not a primary action and a secondary one, and a
   second bright button beside the first would read as the important one. */
QPushButton#monitor {
    background: #161b22; color: #e6edf3; border: 1px solid #2b323b;
    border-radius: 4px; padding: 8px 22px; font-weight: 600; font-size: 13px;
}
QPushButton#monitor:hover { border-color: #4ade80; color: #4ade80; }
QPushButton#monitor:disabled { background: #10151c; color: #4d5865; }
QPushButton#monitor[running="true"] {
    background: #4ade80; color: #0b0e13; border-color: #4ade80;
}
QLabel#nowPlaying { color: #4ade80; font-size: 12px; font-weight: 600; }
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
QPushButton#voiceChip {
    background: #161b22; color: #8b98a5;
    border: 1px solid #2b323b; border-radius: 11px;
    padding: 3px 11px; font-size: 11px;
}
QPushButton#voiceChip:hover { border-color: #3d4650; }
QPushButton#voiceChip:checked {
    background: #4ade80; color: #0b0e13; border-color: #4ade80; font-weight: 600;
}
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

        # Monitoring. The cards here are keyed and updated rather than
        # rebuilt - see `widgets/activitycard.py` - so what is kept is a map
        # from channel to card plus the order they are currently drawn in.
        self._activity: dict[int, ActivityCard] = {}
        self._activity_order: tuple[int, ...] = ()
        self._monitor_sort = mon.ACTIVITY_SORTS[0][1]
        self._voice_only = False
        self._monitoring = False
        # Which question this screen is currently answering. It survives the
        # session ending, so stopping a monitor leaves its ledger on screen
        # rather than snapping back to whatever sweep preceded it - the same
        # bargain a finished scan gets.
        self._mode = "scan"

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

        # Reworded at Expert, because the two pickers ask different
        # questions: the chips are "where should I start?" and the whole-dial
        # list is "what is on 150.8 MHz?". A heading that still said "pick a
        # band" over sixty tick boxes would be describing the other screen.
        self.heading = QLabel("")
        self.heading.setObjectName("heading")
        self.heading.setWordWrap(True)
        outer.addWidget(self.heading)

        #subheading = QLabel(
        #    "Pick a band and press Scan. Anything transmitting shows up below."
        #)
        #subheading.setObjectName("subheading")
        #outer.addWidget(subheading)

        outer.addWidget(self._band_chips())
        outer.addWidget(self._range_picker())
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

    def _band_chips(self) -> QWidget:
        """One button per scannable band, straight out of the band plan.

        The short list, and deliberately so: the handful of allocations worth
        putting in front of somebody who has never used a receiver. Expert
        swaps it for the whole dial - see `_range_picker`.
        """
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
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
        self._chip_holder = holder
        return holder

    def _range_picker(self) -> QWidget:
        """Every stretch of dial the dongle can reach, at Expert only.

        Not an extra control beside the chips but a replacement for them, and
        that is the point: the chips answer "where should I start?" and this
        answers "what is on 150.8 MHz?", which are different questions asked
        by different people. Showing both at once would put two selections on
        one screen with no way to tell which one Scan is about to use.
        """
        picker = RangePicker()
        picker.selectionChanged.connect(self._selection_chosen)
        picker.setVisible(False)
        self.picker = picker
        return picker

    def _actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.scan_button = QPushButton("Scan")
        self.scan_button.setObjectName("scan")
        self.scan_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_button.setToolTip(
            "Sweep this band a few times and list what is transmitting right "
            "now."
        )
        self.scan_button.clicked.connect(self._scan_clicked)
        row.addWidget(self.scan_button)

        # At every level, including Simple. A scan is the wrong tool for most
        # of the dial - a fire crew's channel is silent fifty-nine minutes an
        # hour, so a single sweep of it honestly reports nothing and is wrong
        # about the band - and somebody who does not yet know that is
        # precisely the person who needs the other button to exist.
        self.monitor_button = QPushButton("Monitor")
        self.monitor_button.setObjectName("monitor")
        self.monitor_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.monitor_button.setToolTip(
            "Keep sweeping this band, count how often each channel is up, and "
            "stop and play anything that turns out to be somebody talking."
        )
        self.monitor_button.clicked.connect(self._monitor_clicked)
        row.addWidget(self.monitor_button)

        # A question mark rather than a link on the caption, for the reason
        # `widgets/help.py` gives: a button's own text has to go on pressing
        # the button. Monitor is the control on this screen most in need of
        # one - "keep sweeping and stop on anything that talks" is a whole
        # mental model, not a setting.
        monitor_help = HelpButton(topic="monitor-mode")
        monitor_help.helpRequested.connect(self.helpRequested.emit)
        row.addWidget(monitor_help)

        self.now_playing = QLabel("")
        self.now_playing.setObjectName("nowPlaying")
        self.now_playing.setVisible(False)
        row.addWidget(self.now_playing)

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

        self.voice_only = QPushButton("")
        self.voice_only.setObjectName("voiceChip")
        self.voice_only.setCheckable(True)
        self.voice_only.setCursor(Qt.CursorShape.PointingHandCursor)
        self.voice_only.setToolTip(
            "Show only the channels somebody has actually been heard talking "
            "on. Everything else stays on the list underneath."
        )
        self.voice_only.setVisible(False)
        self.voice_only.toggled.connect(self._voice_only_toggled)
        self._filter_row.addWidget(self.voice_only)

        self.voice_help = HelpButton(topic="voice-detection")
        self.voice_help.helpRequested.connect(self.helpRequested.emit)
        self.voice_help.setVisible(False)
        self._filter_row.addWidget(self.voice_help)

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

    # -- what is about to be swept -----------------------------------------

    @property
    def _expert(self) -> bool:
        """Whether the whole-dial picker is the one being used."""
        return self.level >= Level.EXPERT

    def _targets(self) -> tuple[bandplan.Segment, ...]:
        """The stretches of dial Scan and Monitor are about to cover.

        One place, asked by everything - the two buttons, the status line, the
        empty-list message and what a saved result gets filed under. The
        alternative is each of them deciding for itself which picker is in
        charge, and the failure that produces is a screen that scans one band
        while saying it scanned another.
        """
        if self._expert:
            return self.picker.selection
        if self._band is None:
            return ()
        return (bandplan.Segment.of(self._band),)

    def _target_phrase(self) -> str:
        """What to call the selection in a sentence, article and all."""
        targets = self._targets()
        if not targets:
            return "that range"
        if len(targets) == 1:
            only = targets[0]
            return f"the {only.name} band" if only.band is not None else only.name
        return f"{len(targets)} selected ranges"

    def _update_target_state(self) -> None:
        """Nothing selected is a real state, and the buttons have to say so.

        The same rule as "nothing may look clickable and then do nothing": a
        Scan button that is bright and does nothing because every box is
        unticked reads as a broken sweep, not as an empty selection.
        """
        have = bool(self._targets())
        self.scan_button.setEnabled(
            (have and not self.engine.monitoring) or self.engine.scanning
        )
        self.monitor_button.setEnabled(have or self.engine.monitoring)

    def _selection_chosen(self) -> None:
        """The whole-dial picker changed. Same consequences as a band chip."""
        self._remember()
        self._update_target_state()
        if self.engine.monitoring:
            # The ledger counts sightings across one selection and means
            # something else across another, so the session restarts rather
            # than carrying its figures over. Same reasoning as a band chip.
            self.engine.stop_monitor()
            self._monitor_clicked()

    # -- ordering and filtering --------------------------------------------

    def _sort_chosen(self) -> None:
        chosen = self.sort.currentData()
        if not chosen:
            return
        if self._mode == "monitor":
            self._monitor_sort = chosen
        else:
            self._sort = chosen
        self._remember()
        self._relist()

    def _set_sort_options(self, monitoring: bool) -> None:
        """Offer the orders that mean something on the list being shown.

        "Busiest first" has no answer for a single sweep and "strongest first"
        is a poor one for a band being watched, so the two sets are not the
        same. Rebuilt rather than permanently offered because an order that
        silently falls back to another is a control that looks like it did
        something and did not.
        """
        wanted = (
            (*mon.ACTIVITY_SORTS, *results.SORTS) if monitoring else results.SORTS
        )
        if [self.sort.itemData(i) for i in range(self.sort.count())] == [
            key for _, key in wanted
        ]:
            return
        current = self._monitor_sort if monitoring else self._sort
        self.sort.blockSignals(True)
        self.sort.clear()
        for label, key in wanted:
            self.sort.addItem(label, key)
        index = self.sort.findData(current)
        self.sort.setCurrentIndex(max(0, index))
        self.sort.blockSignals(False)

    def _voice_only_toggled(self, only: bool) -> None:
        self._voice_only = bool(only)
        self._remember()

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
        """Re-render whatever this screen last had, in the new order."""
        if self._mode == "monitor":
            state = self.engine.monitor_update()
            if state is not None:
                self._activity_order = ()
                self._show_monitor(state)
            return
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
        stored = str(settings["monitor_sort"])
        if stored in {key for _, key in (*mon.ACTIVITY_SORTS, *results.SORTS)}:
            self._monitor_sort = stored
        self._voice_only = bool(settings["monitor_voice_only"])
        self.voice_only.blockSignals(True)
        self.voice_only.setChecked(self._voice_only)
        self.voice_only.blockSignals(False)
        # Silently, because restoring is not choosing: `set_keys` does not
        # emit, so a remembered selection cannot read as sixty clicks - which
        # for a running session would be sixty restarts.
        self.picker.set_keys(settings["scan_ranges"])

    def _remember(self) -> None:
        settings = self.settings
        if settings is None:
            return
        settings.update(
            scan_sort=self._sort,
            hidden_kinds=sorted(self._hidden),
            monitor_sort=self._monitor_sort,
            monitor_voice_only=self._voice_only,
            scan_ranges=list(self.picker.keys),
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
        targets = self._targets()
        group = targets[0].name if len(targets) == 1 else "Found by scanning"
        for signal in signals:
            self.bookmarks.add(from_signal(signal, group=group), replace_existing=False)
        self.bookmarks.save()
        self.save_found.setText(f"Saved {len(signals)}")

    # -- level -------------------------------------------------------------

    def set_level(self, level: Level) -> None:
        self.level = level
        for widget, minimum in self._levelled:
            widget.setVisible(level >= minimum)
        expert = self._expert
        self.heading.setText(
            "What's on the air around you? Tick any part of the dial - the "
            "bands, and everything between them - then Scan for a snapshot "
            "or Monitor to keep watching."
            if expert
            else "What's on the air around you? Pick a band, then Scan for a "
            "snapshot or Monitor to keep watching it."
        )
        # Arriving at Expert with a band chosen on the chips should show that
        # band ticked, not an empty list and a dead Scan button. Seeded only
        # when nothing is ticked, so a selection made at Expert survives a
        # trip down to Simple and back.
        if expert and not self.picker.selection:
            self.picker.set_band(self._band)
        self._chip_holder.setVisible(not expert)
        self.picker.setVisible(expert)
        self._update_target_state()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        # A sweep ends on its own in a few seconds; a monitor session never
        # does. Walking away from this page with one running would leave the
        # radio parked in another band for as long as the app is open, with
        # the listening screen showing a frequency it is not on.
        if self.engine.monitoring:
            self.engine.stop_monitor()

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
        self._update_target_state()
        if self.engine.monitoring:
            # The ledger counts sightings in one band and means nothing in
            # another, so the session restarts rather than carrying figures
            # across. Stopped and started in one gesture: the engine's own
            # guard would otherwise refuse the start.
            self.engine.stop_monitor()
            self._monitor_clicked()

    def _scan_clicked(self) -> None:
        if self.engine.scanning:
            self.engine.stop_scan()
            return
        ranges = ranges_for(self._targets())
        if not ranges:
            return
        if self.engine.monitoring:
            # Only one thing may borrow the radio at a time, and the engine
            # refuses the second - so the button that was pressed has to end
            # the session it is replacing rather than silently doing nothing.
            self.engine.stop_monitor()
        self._mode = "scan"
        self._clear_activity()
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
        self.status.setText(f"Listening across {self._target_phrase()}...")
        self.engine.start_scan(
            # Several stretches at once, each carrying its own window
            # preference. Some bands need a narrower one than the default -
            # see `frontend.safe_sample_rate` - and a selection spanning both
            # AM and FM has to answer that question once per range, because
            # one answer for both is wrong about one of them. The engine
            # clamps them anyway; the band plan is where the preference is
            # allowed to be stated.
            ranges=ranges,
            threshold_db=SENSITIVITY_DB[self.sensitivity.currentData()],
        )


    # -- monitoring --------------------------------------------------------

    def _monitor_clicked(self) -> None:
        if self.engine.monitoring:
            self.engine.stop_monitor()
            return
        ranges = ranges_for(self._targets())
        if not ranges:
            return
        phrase = self._target_phrase()
        self._mode = "monitor"
        self._clear()
        self._clear_activity()
        self._clear_chips()
        self._kinds = frozenset()
        self._signals = ()
        self._shown = ()
        self._publish(())
        self._filter_holder.setVisible(True)
        self.progress.setVisible(False)
        self.status.setText(f"Starting to watch {phrase}...")
        # Not `_update_empty`, which speaks for the sweep and would greet a
        # starting session with "press Scan".
        self.empty.setText(f"Listening across {phrase}...")
        self.empty.setVisible(True)
        self.engine.start_monitor(
            band_name=phrase,
            ranges=ranges,
            threshold_db=SENSITIVITY_DB[self.sensitivity.currentData()],
        )

    def _monitor_tick(self) -> None:
        """Poll the ledger and keep the chrome agreeing with the engine.

        Driven by what the engine is actually doing rather than by what the
        button press assumed - a session can fail to start, or be ended from
        somewhere else - which is the same rule `_tick` follows for a sweep.
        """
        running = self.engine.monitoring
        if running != self._monitoring:
            self._monitoring = running
            self._set_sort_options(running)
            self._update_target_state()
            self.monitor_button.setText("Stop" if running else "Monitor")
            self.monitor_button.setProperty("running", "true" if running else "false")
            self.monitor_button.style().unpolish(self.monitor_button)
            self.monitor_button.style().polish(self.monitor_button)
            self.voice_only.setVisible(running)
            self.voice_help.setVisible(running)
            if not running:
                self.now_playing.setVisible(False)

        state = self.engine.monitor_update()
        if state is None:
            if not running:
                # The engine refused to start - no radio, or something else
                # already has it. Saying nothing would leave "Starting to
                # watch..." on screen for ever.
                self.status.setText(
                    "Could not start watching this band. Something else may "
                    "have the radio."
                )
            return
        self._show_monitor(state)
        self.status.setText(self._monitor_status(state, running))
        playing = running and state.listening and state.target_hz is not None
        self.now_playing.setVisible(playing)
        if playing:
            self.now_playing.setText(
                f"{glyph('walkie')}  Playing {format_frequency(state.target_hz)}"
            )

    def _monitor_status(self, state: mon.MonitorState, running: bool) -> str:
        band = state.band_name or self._target_phrase()
        if not running:
            return (
                f"Stopped watching {band} after {state.passes} sweeps. "
                f"{len(state.channels)} channels found; press Monitor to carry on."
            )
        if state.target_hz is not None and state.phase == mon.HOLDING:
            return (
                f"Playing {format_frequency(state.target_hz)} - back to sweeping "
                f"when it goes quiet."
            )
        if state.target_hz is not None:
            return (
                f"Listening to {format_frequency(state.target_hz)} to hear what "
                f"is on it..."
            )
        return (
            f"Watching {band}  -  sweep {state.passes}  -  "
            f"{len(state.channels)} channels, {state.busy} up now, "
            f"{state.voices} with voices on them{self._held_note(state)}"
        )

    def _held_note(self, state: mon.MonitorState) -> str:
        """", 12 hidden" - the same promise the sweep's chips make.

        A filter that persists between sittings must never be mistakable for a
        band where nothing was found, so whatever is being held back is said
        out loud wherever a count appears.
        """
        held = sum(
            1
            for channel in state.channels
            if channel.label in self._hidden
            or (self._voice_only and not channel.heard_voice)
        )
        return f"  -  {held} hidden" if held else ""

    def _visible_activity(self, state: mon.MonitorState) -> tuple[mon.Activity, ...]:
        channels = mon.sort_activities(state.channels, self._monitor_sort)
        if self._hidden:
            channels = tuple(c for c in channels if c.label not in self._hidden)
        if self._voice_only:
            channels = mon.with_voice(channels)
        return channels

    def _show_monitor(self, state: mon.MonitorState) -> None:
        """Bring the list up to date without rebuilding what is already there.

        The order changes rarely and the numbers change constantly, so the two
        are separated: cards are created, removed and re-ordered only when the
        set on screen actually differs, and everything that merely ticks is
        written into the labels already on the widgets. Rebuilding at the poll
        rate would slam every open explanation shut and move the Hold and Skip
        buttons out from under whoever was reaching for one.
        """
        channels = self._visible_activity(state)
        # Summarised from the activities rather than from the signals inside
        # them: `Activity.label` is what the card shows, and a chip counting
        # something under a name the card no longer uses would hide the wrong
        # rows. `summarise` needs only `label` and `icon`, which both carry.
        self._update_filters(results.summarise(state.channels, self._hidden))
        self.voice_only.setText(f"{glyph('walkie')}  Voices only ({state.voices})")
        showing = self._monitoring or self._voice_only
        self.voice_only.setVisible(showing)
        self.voice_help.setVisible(showing)

        order = tuple(int(round(c.frequency_hz)) for c in channels)
        if order != self._activity_order:
            self._reorder_activity(channels, order)
        target = state.target_hz if state.listening else None
        for channel in channels:
            card = self._activity.get(int(round(channel.frequency_hz)))
            if card is not None:
                card.update_activity(
                    channel,
                    playing=target is not None
                    and abs(channel.frequency_hz - target) < mon.MATCH_TOLERANCE_HZ,
                )
        listed = tuple(c.signal for c in channels)
        if tuple(s.frequency_hz for s in listed) != tuple(
            s.frequency_hz for s in self._listed
        ):
            self._publish(listed)
        self._update_empty_monitor(state, bool(channels))

    def _reorder_activity(
        self, channels: tuple[mon.Activity, ...], order: tuple[int, ...]
    ) -> None:
        self._activity_order = order
        wanted: dict[int, ActivityCard] = {}
        for channel in channels:
            key = int(round(channel.frequency_hz))
            card = self._activity.get(key)
            if card is None:
                card = ActivityCard(channel)
                card.listenRequested.connect(self.listenRequested.emit)
                card.holdRequested.connect(self.engine.monitor_hold)
                card.releaseRequested.connect(self.engine.monitor_release)
                card.skipRequested.connect(self.engine.monitor_skip)
                card.resumeRequested.connect(self.engine.monitor_resume)
                self._activity[key] = card
            wanted[key] = card
        for key in [k for k in self._activity if k not in wanted]:
            card = self._activity.pop(key)
            self.list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        for index, key in enumerate(order):
            self.list_layout.insertWidget(index, wanted[key])

    def _update_empty_monitor(self, state: mon.MonitorState, any_shown: bool) -> None:
        """An empty monitor list has three causes, and they are not the same.

        Nothing heard yet, everything filtered out, or a band where nothing
        transmits. Saying "nothing found" to somebody four seconds into a
        session would be wrong about all three.
        """
        if any_shown:
            self.empty.setVisible(False)
            return
        if state.channels:
            self.empty.setText(
                "Everything heard here is hidden by the chips above. Turn one "
                "back on to see it."
            )
        elif state.passes < 2:
            self.empty.setText(
                "Listening... nothing has been heard twice yet. A channel has "
                "to come up in more than one sweep before it counts."
            )
        else:
            self.empty.setText(
                f"Nothing heard in {state.passes} sweeps of this band. Quiet "
                f"bands are normal - leave it running, or try turning the "
                f"sensitivity up."
            )
        self.empty.setVisible(True)

    def _clear_activity(self) -> None:
        for card in self._activity.values():
            self.list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._activity.clear()
        self._activity_order = ()
        self.now_playing.setVisible(False)

    def _tick(self) -> None:
        if self._mode == "monitor":
            self._monitor_tick()
            return
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
        phrase = self._target_phrase()
        count = len(update.signals)
        if count:
            self.status.setText(
                f"Found {count} in {phrase}"
                f"{self._hidden_note(update.signals)}. "
                f"Press Listen on any of them."
            )
        else:
            self.status.setText(
                f"Nothing found in {phrase}. Try turning the "
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
        elif self._expert and not self._targets():
            self.empty.setText(
                "Nothing selected. Tick one or more ranges above, then press "
                "Scan or Monitor."
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
