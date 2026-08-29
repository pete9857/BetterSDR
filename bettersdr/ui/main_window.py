"""Application shell: the window, the level selector, and startup diagnosis.

The startup path matters more than it looks. If the dongle is missing or still
bound to the Windows TV driver, the honest thing is a screen that says what to
do about it in plain English - not a stack trace and not an empty spectrum
that leaves the user wondering whether the radio is simply quiet.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core import doctor
from ..core.bookmarks import BookmarkStore
from ..core.engine import Engine
from ..core.history import History
from ..core.settings import Settings
from ..scan.classifier import Signal
from .aircraft_view import AircraftView
from .discover_view import DiscoverView
from .levels import Level
from .listen_view import ListenView

WINDOW_STYLE = """
QMainWindow, QWidget#root { background: #0b0e13; }
QStatusBar { background: #10151c; color: #8b98a5; }
QStatusBar::item { border: none; }
QWidget#toolbar { background: #10151c; border-bottom: 1px solid #1d232b; }
QPushButton#level, QPushButton#nav {
    background: #161b22; color: #8b98a5;
    border: 1px solid #2b323b; padding: 3px 14px;
}
QPushButton#level:checked { background: #5ad1ff; color: #0b0e13; font-weight: 600; }
QPushButton#nav:checked { background: #2b323b; color: #e6edf3; font-weight: 600; }
QLabel#problemTitle { color: #e6edf3; font-size: 17px; font-weight: 600; }
QLabel#problemBody { color: #8b98a5; font-size: 12px; }
"""


class ProblemView(QWidget):
    """Shown instead of the radio when the dongle is not usable yet.

    The text comes straight from `core/doctor.py`, which has no UI imports so
    that the CLI and this screen can never drift apart in what they tell the
    user.
    """

    def __init__(self, title: str, body: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(14)
        layout.addStretch(1)

        heading = QLabel(title)
        heading.setObjectName("problemTitle")
        heading.setWordWrap(True)
        detail = QLabel(body)
        detail.setObjectName("problemBody")
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addStretch(2)


class MainWindow(QMainWindow):
    """The whole app."""

    def __init__(
        self,
        engine: Engine | None = None,
        level: Level = Level.STANDARD,
        settings: Settings | None = None,
        bookmarks: BookmarkStore | None = None,
        history: History | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("BetterSDR")
        self.resize(1180, 760)
        self.setStyleSheet(WINDOW_STYLE)
        self.engine = engine
        self.settings = settings
        # One store shared by both screens, so a frequency saved from a scan
        # result and one saved while listening land in the same list and the
        # star on either screen agrees with the other.
        self.bookmarks = bookmarks if bookmarks is not None else BookmarkStore.open()
        # One history, for the same reason there is one bookmark store: the
        # listening screen writes it and the Discover strip reads it, and two
        # copies would have the strip showing a station the radio left
        # minutes ago.
        self.history = history if history is not None else History.open()
        self.view: ListenView | None = None
        self.discover: DiscoverView | None = None
        self.aircraft: AircraftView | None = None
        self._stack: QStackedWidget | None = None

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar(level))

        if engine is None:
            headline, steps = self._explain_failure()
            layout.addWidget(ProblemView(headline, steps), 1)
            self._levels.setExclusive(False)
            for button in self._levels.buttons():
                button.setEnabled(False)
            for button in self._nav.buttons():
                button.setEnabled(False)
        else:
            # Discover is the landing screen at every level. Opening on a
            # spectrum is what every other SDR application does, and being
            # shown a list of what is actually out there is the entire
            # argument this one is making.
            self.discover = DiscoverView(
                engine,
                level=level,
                bookmarks=self.bookmarks,
                history=self.history,
            )
            self.view = ListenView(
                engine,
                level=level,
                settings=settings,
                bookmarks=self.bookmarks,
                history=self.history,
            )
            self.aircraft = AircraftView(engine, level=level)
            self.discover.listenRequested.connect(self._listen_to)
            self.discover.tuneRequested.connect(self._tune_to)
            self._stack = QStackedWidget()
            self._stack.addWidget(self.discover)
            self._stack.addWidget(self.view)
            self._stack.addWidget(self.aircraft)
            layout.addWidget(self._stack, 1)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self._status_text = ""
        self._show_status()
        # The status line reports hardware state that the views change behind
        # its back - the window narrows to 240 kS/s the moment anyone tunes to
        # the AM band - so it is polled rather than pushed. Slowly, and only
        # repainted when the text actually differs.
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._show_status)
        self._status_timer.start(500)

    @staticmethod
    def _explain_failure() -> tuple[str, str]:
        """Why the radio would not start, in the user's terms.

        The doctor covers the driver-shaped problems. It cannot see the other
        common one - another program already holding the dongle - and telling
        someone their dongle is "ready to use" on a screen that exists because
        it was not would be worse than saying nothing.
        """
        diagnosis = doctor.diagnose()
        if not diagnosis.ok:
            steps = "\n".join(
                f"{n}.  {step}" for n, step in enumerate(diagnosis.remedy, 1)
            )
            return diagnosis.headline, steps
        return (
            "The dongle is set up correctly, but could not be started",
            "1.  Another program is probably using it. Close any other radio "
            "software and try again.\n"
            "2.  If nothing else is running, unplug the dongle, plug it back "
            "in, and restart BetterSDR.",
        )

    def _toolbar(self, level: Level) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar")
        bar.setFixedHeight(38)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 10, 4)

        title = QLabel("BetterSDR")
        title.setStyleSheet("color: #e6edf3; font-weight: 600; font-size: 13px;")
        layout.addWidget(title)
        layout.addSpacing(16)

        self._nav = QButtonGroup(self)
        self._nav.setExclusive(True)
        for index, name in enumerate(("Discover", "Listen", "Aircraft")):
            button = QPushButton(name)
            button.setObjectName("nav")
            button.setCheckable(True)
            button.setChecked(index == 0)
            self._nav.addButton(button, index)
            layout.addWidget(button)
        self._nav.idClicked.connect(self._show_page)

        layout.addStretch(1)

        self._levels = QButtonGroup(self)
        self._levels.setExclusive(True)
        for candidate in Level:
            button = QPushButton(candidate.label)
            button.setObjectName("level")
            button.setCheckable(True)
            button.setToolTip(candidate.description)
            button.setChecked(candidate is level)
            self._levels.addButton(button, int(candidate))
            layout.addWidget(button)
        self._levels.idClicked.connect(self._level_changed)
        return bar

    def _level_changed(self, value: int) -> None:
        level = Level(value)
        for page in (self.discover, self.view, self.aircraft):
            if page is not None:
                page.set_level(level)
        if self.settings is not None:
            # Remembered immediately rather than at close: somebody who has
            # reached Expert should not be met by Simple tomorrow morning, and
            # a crash is exactly the session where that would happen.
            self.settings["level"] = level.name.lower()
            self.settings.save()

    def _show_page(self, index: int) -> None:
        """Switch views, and stop the ones leaving so they cost nothing.

        Every page polls the engine on a timer. A hidden page that kept
        polling would repaint widgets nobody can see and, worse, keep asking a
        scan for progress after the user has walked away from it.

        Stopping comes first, and that is not tidiness. A page can have the
        radio on loan - the aircraft screen parks it at 1090 MHz - and the
        page arriving reads the radio's state to set itself up. Started first,
        the listening screen took the band plan's aircraft entry and applied
        it to the FM station underneath.
        """
        if self._stack is None:
            return
        pages = (self.discover, self.view, self.aircraft)
        for position, page in enumerate(pages):
            if page is not None and position != index:
                page.stop()
        arriving = pages[index] if 0 <= index < len(pages) else None
        if arriving is not None:
            arriving.start()
        self._stack.setCurrentIndex(index)
        button = self._nav.button(index)
        if button is not None:
            button.setChecked(True)

    def _listen_to(self, signal: Signal) -> None:
        """Somebody pressed Listen on a card."""
        if self.view is None:
            return
        if self.engine is not None and self.engine.scanning:
            self.engine.stop_scan()
        # Switch first, then apply: showing the page calls the view's start(),
        # and applying the signal before that would have the view's own
        # start-up overwrite the mode the classifier chose.
        self._show_page(1)
        self.view.show_signal(signal)

    def _tune_to(self, entry) -> None:
        """Somebody pressed a favourite or a recently played chip.

        The same order as `_listen_to` and for the same reason: showing the
        page runs the view's own start-up, which would otherwise overwrite
        the mode the chip carried with the band plan's.
        """
        if self.view is None:
            return
        if self.engine is not None and self.engine.scanning:
            self.engine.stop_scan()
        self._show_page(1)
        self.view.tune_to(
            int(entry.frequency_hz), entry.mode, float(entry.bandwidth_hz)
        )

    def _show_status(self) -> None:
        if self.engine is None:
            self.statusBar().showMessage("No radio connected")
            return
        gain = self.engine.gain
        rate = self.engine.sample_rate
        parts = [
            f"{rate / 1e6:.1f} MS/s"
            if rate >= 1_000_000
            else f"{rate / 1e3:.0f} kS/s"
        ]
        if gain is not None:
            parts.append(f"gain {gain.gain_db:.1f} dB")
            if gain.overloaded:
                parts.append("front end overloaded - try a shorter antenna")
        recording = self.engine.recording
        if recording.active:
            parts.append("recording")
        if recording.message:
            parts.append(recording.message)
        text = "   ".join(parts)
        if text != self._status_text:
            self._status_text = text
            self.statusBar().showMessage(text)

    # -- lifecycle ---------------------------------------------------------

    def showEvent(self, event) -> None:
        if self._stack is not None:
            self._show_page(self._stack.currentIndex())
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        for page in (self.discover, self.view, self.aircraft):
            if page is not None:
                page.stop()
        if self.view is not None:
            self.view.remember()
        self.bookmarks.save()
        # `leave` before `save`: the visit in progress is worth counting, and
        # the whole point of the recent list is to have last night's station
        # on the landing screen tomorrow morning.
        self.history.leave()
        self.history.save()
        if self.engine is not None:
            # Stops the radio *and* closes any recording with a valid header.
            self.engine.stop()
        super().closeEvent(event)


__all__ = ["MainWindow", "ProblemView"]
