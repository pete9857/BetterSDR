"""Application shell: the window, the level selector, and startup diagnosis.

The startup path matters more than it looks. If the dongle is missing or still
bound to the Windows TV driver, the honest thing is a screen that says what to
do about it in plain English - not a stack trace and not an empty spectrum
that leaves the user wondering whether the radio is simply quiet.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core import doctor
from ..core.engine import Engine
from .levels import Level
from .listen_view import ListenView

WINDOW_STYLE = """
QMainWindow, QWidget#root { background: #0b0e13; }
QStatusBar { background: #10151c; color: #8b98a5; }
QStatusBar::item { border: none; }
QWidget#toolbar { background: #10151c; border-bottom: 1px solid #1d232b; }
QPushButton#level {
    background: #161b22; color: #8b98a5;
    border: 1px solid #2b323b; padding: 3px 14px;
}
QPushButton#level:checked { background: #5ad1ff; color: #0b0e13; font-weight: 600; }
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
        self, engine: Engine | None = None, level: Level = Level.STANDARD
    ) -> None:
        super().__init__()
        self.setWindowTitle("BetterSDR")
        self.resize(1180, 760)
        self.setStyleSheet(WINDOW_STYLE)
        self.engine = engine
        self.view: ListenView | None = None

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
        else:
            self.view = ListenView(engine, level=level)
            layout.addWidget(self.view, 1)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self._show_status()

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
        if self.view is not None:
            self.view.set_level(Level(value))

    def _show_status(self) -> None:
        if self.engine is None:
            self.statusBar().showMessage("No radio connected")
            return
        gain = self.engine.gain
        parts = [f"{self.engine.sample_rate / 1e6:.1f} MS/s"]
        if gain is not None:
            parts.append(f"gain {gain.gain_db:.1f} dB")
            if gain.overloaded:
                parts.append("front end overloaded - try a shorter antenna")
        self.statusBar().showMessage("   ".join(parts))

    # -- lifecycle ---------------------------------------------------------

    def showEvent(self, event) -> None:
        if self.view is not None:
            self.view.start()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        if self.view is not None:
            self.view.stop()
        if self.engine is not None:
            self.engine.stop()
        super().closeEvent(event)


__all__ = ["MainWindow", "ProblemView"]
