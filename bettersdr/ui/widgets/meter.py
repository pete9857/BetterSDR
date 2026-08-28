"""Signal strength meter.

Shows the power in the tuned channel, which is the number that answers "is
anything actually there?". It reads in dBFS because that is what the rest of
the app is calibrated in, but the bar itself is the part a beginner uses - the
scale is there for when they start caring.

The peak marker decays rather than sticking, so a short transmission stays
visible for a moment after it ends without permanently misreporting the level.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

FLOOR_DBFS = -80.0
CEILING_DBFS = 0.0
# Roughly one second to fall the full scale at a 30 Hz refresh.
PEAK_DECAY_DB_PER_UPDATE = 2.5

BACKGROUND = QColor("#0b0e13")
TRACK = QColor("#161b22")
BORDER = QColor("#2b323b")
TEXT = QColor("#8b98a5")
PEAK = QColor("#e6edf3")
MUTED = QColor("#3d4650")


class SignalMeter(QWidget):
    """A horizontal bar reading channel power in dBFS."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level_dbfs = FLOOR_DBFS
        self._peak_dbfs = FLOOR_DBFS
        self._squelch_open: bool | None = None
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, level_dbfs: float, squelch_open: bool | None = None) -> None:
        self._level_dbfs = float(level_dbfs)
        self._squelch_open = squelch_open
        self._peak_dbfs = max(
            self._peak_dbfs - PEAK_DECAY_DB_PER_UPDATE, self._level_dbfs
        )
        self.update()

    @staticmethod
    def _fraction(level_dbfs: float) -> float:
        span = CEILING_DBFS - FLOOR_DBFS
        return min(1.0, max(0.0, (level_dbfs - FLOOR_DBFS) / span))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)

        bar = QRectF(2, 2, self.width() - 78, self.height() - 16)
        painter.setPen(BORDER)
        painter.setBrush(TRACK)
        painter.drawRoundedRect(bar, 3, 3)

        gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        gradient.setColorAt(0.0, QColor("#2f6f4f"))
        gradient.setColorAt(0.6, QColor("#5ad1ff"))
        gradient.setColorAt(0.85, QColor("#f0c14a"))
        gradient.setColorAt(1.0, QColor("#e0533f"))

        filled = QRectF(bar)
        filled.setWidth(bar.width() * self._fraction(self._level_dbfs))
        painter.setPen(Qt.PenStyle.NoPen)
        # A closed squelch means audio is muted, so the bar greys out. Without
        # this the meter shows a signal while the speakers are silent, which
        # reads as a bug.
        painter.setBrush(MUTED if self._squelch_open is False else gradient)
        painter.drawRoundedRect(filled, 3, 3)

        peak_x = bar.left() + bar.width() * self._fraction(self._peak_dbfs)
        painter.setPen(PEAK)
        painter.drawLine(int(peak_x), int(bar.top()) + 1, int(peak_x), int(bar.bottom()))

        font = QFont("Segoe UI")
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(TEXT)
        painter.drawText(
            QRectF(bar.right() + 6, bar.top(), 70, bar.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"{self._level_dbfs:.0f} dBFS",
        )

        font.setPointSize(7)
        painter.setFont(font)
        for level in range(int(FLOOR_DBFS), int(CEILING_DBFS) + 1, 20):
            x = bar.left() + bar.width() * self._fraction(level)
            # Nudge the end labels inward rather than letting them clip. A
            # scale reading "80" where it means "-80" is worse than one
            # slightly off-centre from its tick.
            left = max(0.0, min(x - 14, self.width() - 28))
            painter.drawText(
                QRectF(left, bar.bottom() + 1, 28, 12),
                Qt.AlignmentFlag.AlignCenter,
                str(level),
            )
        painter.end()


__all__ = ["CEILING_DBFS", "FLOOR_DBFS", "SignalMeter"]
