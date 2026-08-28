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

from ..core.engine import Engine, ScanUpdate
from ..scan import bandplan
from ..scan.classifier import Signal, format_frequency
from ..scan.detector import SENSITIVITY_DB
from .levels import Level
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
"""


class DiscoverView(QWidget):
    """Sweep a band and list what is transmitting in it."""

    listenRequested = QtSignal(object)

    def __init__(
        self,
        engine: Engine,
        level: Level = Level.SIMPLE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.level = level
        self._levelled: list[tuple[QWidget, Level]] = []
        self._cards: list[SignalCard] = []
        self._shown: tuple[tuple[float, str], ...] = ()
        self._band: bandplan.Band | None = None
        self._running = False

        self._build()
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

        heading = QLabel("What's on the air around you")
        heading.setObjectName("heading")
        outer.addWidget(heading)

        subheading = QLabel(
            "Pick a band and press Scan. Anything transmitting shows up below."
        )
        subheading.setObjectName("subheading")
        outer.addWidget(subheading)

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

        sensitivity_label = QLabel("Sensitivity")
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

        row.addStretch(1)
        return row

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
        self._shown = ()
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
                f"Found {count} in the {band} band. "
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
        entirely alone.
        """
        # Strongest first, the way a Wi-Fi picker orders networks. In a band
        # full of interference this is what puts the real stations at the top
        # instead of leaving them scattered through eighty steady tones.
        ordered = sorted(signals, key=lambda s: s.snr_db, reverse=True)
        signature = tuple((s.frequency_hz, s.label) for s in ordered)
        if signature == self._shown:
            return
        self._shown = signature

        self._clear()
        for signal in ordered:
            card = SignalCard(signal)
            card.listenRequested.connect(self.listenRequested.emit)
            self.list_layout.insertWidget(self.list_layout.count() - 1, card)
            self._cards.append(card)
        self.empty.setVisible(not signals)

    def _clear(self) -> None:
        for card in self._cards:
            self.list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self.empty.setVisible(True)


__all__ = ["DiscoverView"]
