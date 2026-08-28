"""One found signal, as a card in the discovery list.

This is the screen the whole project is arguing for, so the card has to do
three things at once: say what the signal is in words a beginner already
knows, let them hear it in one click, and - the part that separates this from
a scanner plugin - show its reasoning when asked. The "What is this?" expander
is not a debugging aid; it is how somebody learns what a band plan is without
being told they are being taught.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...scan.classifier import Signal, Strength
from .icons import glyph

BARS = 4
BAR_ON = QColor("#5ad1ff")
BAR_OFF = QColor("#2b323b")

CARD_STYLE = """
QFrame#card { background: #10151c; border: 1px solid #1d232b; border-radius: 6px; }
QFrame#card:hover { border-color: #2b323b; }
QLabel#cardIcon { font-size: 24px; }
QLabel#cardTitle { color: #e6edf3; font-size: 15px; font-weight: 600; }
QLabel#cardWhere { color: #8b98a5; font-size: 12px; }
QLabel#cardWhy { color: #6d7b89; font-size: 11px; }
QLabel#cardAbout { color: #8b98a5; font-size: 12px; }
QLabel#cardHd {
    color: #0b0e13; background: #5ad1ff; border-radius: 3px;
    padding: 0px 5px; font-size: 10px; font-weight: 600;
}
QLabel#cardGuess {
    color: #0b0e13; background: #c9a33a; border-radius: 3px;
    padding: 0px 5px; font-size: 10px; font-weight: 600;
}
QPushButton#listen {
    background: #5ad1ff; color: #0b0e13; border: none;
    border-radius: 4px; padding: 6px 18px; font-weight: 600;
}
QPushButton#listen:hover { background: #7cdcff; }
QPushButton#explain {
    background: transparent; color: #6d7b89; border: none;
    padding: 0px; text-align: left; font-size: 11px;
}
QPushButton#explain:hover { color: #8b98a5; }
"""


class StrengthBars(QWidget):
    """Four bars, the way every phone shows signal strength."""

    def __init__(self, strength: Strength, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._strength = strength
        self.setFixedSize(30, 22)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolTip(f"{strength.label} signal")

    def set_strength(self, strength: Strength) -> None:
        self._strength = strength
        self.setToolTip(f"{strength.label} signal")
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        width = self.width() / (BARS * 1.6)
        gap = width * 0.6
        for index in range(BARS):
            # Rising heights, so strength reads at a glance without the label.
            height = self.height() * (0.3 + 0.7 * (index + 1) / BARS)
            lit = index < int(self._strength)
            painter.fillRect(
                QRectF(
                    index * (width + gap),
                    self.height() - height,
                    width,
                    height,
                ),
                BAR_ON if lit else BAR_OFF,
            )


class SignalCard(QFrame):
    """One signal: what it is, how strong, and a button to hear it."""

    listenRequested = QtSignal(object)

    def __init__(self, signal: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.signal = signal
        self.setObjectName("card")
        self.setStyleSheet(CARD_STYLE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)
        outer.addLayout(self._row(signal))

        self.about = QLabel(
            f"{signal.description}\n\nWhy we think so: {signal.explanation}"
        )
        self.about.setObjectName("cardAbout")
        self.about.setWordWrap(True)
        self.about.setVisible(False)

        self.explain = QPushButton("What is this?")
        self.explain.setObjectName("explain")
        self.explain.setCursor(Qt.CursorShape.PointingHandCursor)
        self.explain.clicked.connect(self._toggle)
        outer.addWidget(self.explain)
        outer.addWidget(self.about)

    def _row(self, signal: Signal) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        icon = QLabel(glyph(signal.icon))
        icon.setObjectName("cardIcon")
        icon.setFixedWidth(34)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(icon)

        text = QVBoxLayout()
        text.setSpacing(1)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title = QLabel(signal.label)
        title.setObjectName("cardTitle")
        title_row.addWidget(title)
        if signal.hd is not None and signal.hd.present:
            badge = QLabel("HD")
            badge.setObjectName("cardHd")
            badge.setToolTip(signal.hd.summary)
            title_row.addWidget(badge)
        if not signal.certain:
            # Said out loud rather than hidden in the confidence number. An
            # honest "not sure" costs nothing; a confident wrong answer costs
            # the user's trust in every other line on the screen.
            guess = QLabel("BEST GUESS")
            guess.setObjectName("cardGuess")
            guess.setToolTip(signal.explanation)
            title_row.addWidget(guess)
        title_row.addStretch(1)
        text.addLayout(title_row)

        where = QLabel(f"{signal.display_frequency}  -  {signal.strength.label}")
        where.setObjectName("cardWhere")
        text.addWidget(where)
        row.addLayout(text, 1)

        row.addWidget(StrengthBars(signal.strength))

        listen = QPushButton("Listen")
        listen.setObjectName("listen")
        listen.setCursor(Qt.CursorShape.PointingHandCursor)
        listen.clicked.connect(lambda: self.listenRequested.emit(self.signal))
        row.addWidget(listen)
        return row

    def _toggle(self) -> None:
        showing = not self.about.isVisible()
        self.about.setVisible(showing)
        self.explain.setText("Hide" if showing else "What is this?")


__all__ = ["SignalCard", "StrengthBars"]
