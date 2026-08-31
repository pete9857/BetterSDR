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

from ..core import doctor, native
from ..core.bookmarks import BookmarkStore
from ..core.engine import Engine
from ..core.history import History
from ..core.settings import Settings
from ..scan.classifier import Signal
from .aircraft_view import AircraftView
from .discover_view import DiscoverView
from .learn_view import LearnView
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

    # The nav bar, in order, and the index of each page. Named rather than
    # numbered at the call sites: adding Learn made a literal `1` meaning the
    # listening screen the sort of thing that is right until it silently is
    # not.
    PAGES = ("Discover", "Listen", "Aircraft", "Learn")
    DISCOVER, LISTEN, AIRCRAFT, LEARN = range(len(PAGES))

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
        # One history, for the same reason there is one bookmark store: it
        # is written wherever the radio is tuned and read back by the
        # Recently played section, and two copies would have that section
        # offering a station the radio left minutes ago.
        self.history = history if history is not None else History.open()
        self.view: ListenView | None = None
        self.discover: DiscoverView | None = None
        self.aircraft: AircraftView | None = None
        self.learn = LearnView(level=level)
        self._stack: QStackedWidget | None = None
        # Which page a reader on an article should be returned to.
        # Recorded when they leave for the Learn screen, because Back
        # means the control they were looking at, and by then the
        # stack no longer knows which that was.
        self._before_learn = self.DISCOVER

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
                settings=settings,
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
            # The step buttons either side of the frequency readout walk the
            # Discover list, so the listening screen is told what that list
            # currently is rather than reaching across to ask. Primed here as
            # well as connected: a scan whose results are already on screen
            # when the user switches level, or reopens the window, has emitted
            # its last change long before this.
            self.discover.resultsChanged.connect(self.view.set_results)
            self.view.set_results(self.discover.listed)
            # Both screens with named controls on them offer the same
            # way in, and neither knows where it goes. This is the whole
            # crossing: a control's own name is the route to what it means.
            self.discover.helpRequested.connect(self._explain)
            self.view.helpRequested.connect(self._explain)
            self.learn.backRequested.connect(self._leave_learn)
            self._stack = QStackedWidget()
            self._stack.addWidget(self.discover)
            self._stack.addWidget(self.view)
            self._stack.addWidget(self.aircraft)
            self._stack.addWidget(self.learn)
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
        # Asked before the doctor, because the doctor diagnoses how the
        # dongle is bound to Windows - which is the wrong answer entirely
        # when the driver itself was never loaded.
        try:
            native.load()
        except native.DriverNotFoundError as exc:
            return "BetterSDR could not load the radio driver", str(exc)

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
        for index, name in enumerate(self.PAGES):
            button = QPushButton(name)
            button.setObjectName("nav")
            button.setCheckable(True)
            button.setChecked(index == 0)
            self._nav.addButton(button, index)
            layout.addWidget(button)
        self._nav.idClicked.connect(self._nav_clicked)

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
        for page in self._pages:
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
        pages = self._pages
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

    @property
    def _pages(self) -> tuple:
        """Every page, in nav order. `None` where the radio never started."""
        return (self.discover, self.view, self.aircraft, self.learn)

    def _nav_clicked(self, index: int) -> None:
        """A nav button, as opposed to the app switching pages on its own.

        Pressing Learn always arrives at the home page, even from an article
        that is already open. Somebody who reaches for the tab has a browsing
        question rather than the one specific question that brought them here
        from a control - and an article left over from twenty minutes ago is a
        Learn tab that appears to contain one entry.
        """
        if index == self.LEARN:
            self.learn.show_home()
        self._show_page(index)

    def _explain(self, topic: str) -> None:
        """Somebody clicked the name of a control.

        The page they were on is remembered first, because Back has to return
        them to the control they were looking at. Anything else makes the
        explanation a detour that costs them their place, which is exactly the
        thing that stops people ever clicking the second one.
        """
        if self._stack is None:
            return
        self._before_learn = self._stack.currentIndex()
        # Only on the way to a real article. A topic with no article behind it
        # is a mistake caught by the tests, and the right thing to do with it
        # at runtime is nothing at all - not to take the reader off the screen
        # they were using and strand them on a glossary they did not ask for.
        if self.learn.show_topic(topic, from_app=True):
            self._show_page(self.LEARN)

    def _leave_learn(self) -> None:
        """Back, from an article that was opened from a control."""
        target = self._before_learn
        self._show_page(target if target != self.LEARN else self.DISCOVER)

    def _listen_to(self, signal: Signal) -> None:
        """Somebody pressed Listen on a card."""
        if self.view is None:
            return
        if self.engine is not None and self.engine.scanning:
            self.engine.stop_scan()
        # Switch first, then apply: showing the page calls the view's start(),
        # and applying the signal before that would have the view's own
        # start-up overwrite the mode the classifier chose.
        self._show_page(self.LISTEN)
        self.view.show_signal(signal)

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
        for page in self._pages:
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
