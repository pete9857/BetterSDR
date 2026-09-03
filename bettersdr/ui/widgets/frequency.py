"""The big digit-wise frequency readout.

Scrolling the wheel over a single digit changes that digit's decade. It looks
like a cosmetic flourish and is actually the fastest tuning control on any SDR
app: no dialog, no step-size setting to find first, and the granularity you
want is wherever you put the pointer. Copying it is not optional if an SDR#
user is to feel at home.

**And a wheel is not a thing everybody has.** On a laptop the wheel is a
two-finger gesture on a trackpad that a beginner may never have used
deliberately, which left the app's primary tuning control unreachable for
exactly the audience it was written for. So each digit is also two buttons:
clicking its upper half winds it up and its lower half winds it down, by the
same decade the wheel would have moved. The hovered half is the one that
lights up, so what a click is about to do is visible before it happens. The
step buttons either side of the readout are the third way in, and they are
the one that needs no aim.

The arithmetic is module-level functions rather than methods so it can be
tested without a display.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

# 1.766 GHz needs ten digits, and a fixed width stops the readout jumping
# about as the frequency changes.
DIGITS = 10
GROUP = 3
ACTIVE_COLOUR = QColor("#e6edf3")
LEADING_ZERO_COLOUR = QColor("#3d4650")
SEPARATOR_COLOUR = QColor("#5a6672")
HIGHLIGHT_COLOUR = QColor(90, 209, 255, 46)
UNIT_COLOUR = QColor("#8b98a5")


def digit_step_hz(index: int) -> int:
    """How much digit `index` is worth, counting from the left."""
    if not 0 <= index < DIGITS:
        raise ValueError(f"digit index must be 0..{DIGITS - 1}, got {index}")
    return 10 ** (DIGITS - 1 - index)


def nudge_digit(value_hz: int, index: int, steps: int) -> int:
    """Add `steps` to one digit's decade, without touching the others.

    Carrying is deliberately allowed: winding 999 up by one should give 1000,
    which is what anyone expects from a wheel. Clamping happens at the tuning
    range, not here.
    """
    return int(value_hz) + steps * digit_step_hz(index)


def half_at(top: float, height: float, y: float) -> int:
    """Which half of a digit cell a point is in: +1 above, -1 below.

    The boundary belongs to the lower half, arbitrarily but consistently, so
    that a click exactly on the middle pixel always does the same thing.
    """
    return 1 if y < top + height / 2.0 else -1


def format_digits(value_hz: int) -> str:
    """Zero-padded to a fixed width so the display never reflows."""
    return f"{max(0, int(value_hz)):0{DIGITS}d}"


class FrequencyDisplay(QWidget):
    """A large frequency readout whose digits are individually scrollable."""

    valueChanged = Signal(int)

    def __init__(
        self,
        value_hz: int = 98_500_000,
        minimum_hz: int = 500_000,
        maximum_hz: int = 1_766_000_000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.minimum_hz = int(minimum_hz)
        self.maximum_hz = int(maximum_hz)
        self._value_hz = int(value_hz)
        self._hover_digit: int | None = None
        # Which half of the hovered digit the pointer is in, so the highlight
        # says which way a click will go rather than merely which digit it
        # will land on.
        self._hover_half = 1
        self._digit_rects: list[QRectF] = []

        self._font = QFont("Consolas")
        self._font.setPointSize(30)
        self._font.setBold(True)
        self._unit_font = QFont("Segoe UI")
        self._unit_font.setPointSize(11)

        self.setMouseTracking(True)
        self.setMinimumHeight(58)
        # Wide enough for every digit it can ever show. Without this the
        # readout is Expanding with no floor, so a narrow window squeezes it
        # and the digits are painted over the buttons either side - which
        # looks like a rendering fault rather than a window that is too small.
        metrics = QFontMetricsF(self._font)
        separators = (DIGITS - 1) // GROUP
        self.setMinimumWidth(
            int(
                DIGITS * metrics.horizontalAdvance("0")
                + separators * metrics.horizontalAdvance(".")
                + QFontMetricsF(self._unit_font).horizontalAdvance(" MHz")
                + 24
            )
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip(
            "Click the top half of a digit to wind it up and the bottom half "
            "to wind it down - or scroll the wheel over it. The buttons "
            "either side move a whole channel at a time."
        )

    # -- value -------------------------------------------------------------

    @property
    def value_hz(self) -> int:
        return self._value_hz

    def set_value(self, value_hz: int, notify: bool = False) -> None:
        clamped = max(self.minimum_hz, min(self.maximum_hz, int(value_hz)))
        if clamped == self._value_hz:
            return
        self._value_hz = clamped
        self.update()
        if notify:
            self.valueChanged.emit(self._value_hz)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b0e13"))

        painter.setFont(self._font)
        metrics = QFontMetricsF(self._font)
        digit_width = metrics.horizontalAdvance("0")
        separator_width = metrics.horizontalAdvance(".")

        text = format_digits(self._value_hz)
        # Separators fall between groups counted from the right.
        separators = {i for i in range(1, DIGITS) if (DIGITS - i) % GROUP == 0}
        total = DIGITS * digit_width + len(separators) * separator_width
        unit_width = QFontMetricsF(self._unit_font).horizontalAdvance(" MHz")

        x = (self.width() - total - unit_width) / 2.0
        baseline_top = (self.height() - metrics.height()) / 2.0
        self._digit_rects = []

        leading = True
        for index, character in enumerate(text):
            if character != "0":
                leading = False
            rect = QRectF(x, baseline_top, digit_width, metrics.height())
            self._digit_rects.append(rect)

            if index == self._hover_digit:
                # Only the half under the pointer, because that half is a
                # button and the other one is a different button.
                half = QRectF(rect)
                half.setHeight(rect.height() / 2.0)
                if self._hover_half < 0:
                    half.moveTop(rect.top() + rect.height() / 2.0)
                painter.fillRect(half.adjusted(-1, 2, 1, -2), HIGHLIGHT_COLOUR)
            painter.setPen(LEADING_ZERO_COLOUR if leading else ACTIVE_COLOUR)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, character)
            x += digit_width

            if index + 1 in separators:
                painter.setPen(SEPARATOR_COLOUR)
                painter.drawText(
                    QRectF(x, baseline_top, separator_width, metrics.height()),
                    Qt.AlignmentFlag.AlignCenter,
                    ".",
                )
                x += separator_width

        painter.setFont(self._unit_font)
        painter.setPen(UNIT_COLOUR)
        painter.drawText(
            QRectF(x + 6, baseline_top, unit_width + 12, metrics.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Hz",
        )
        painter.end()

    # -- interaction -------------------------------------------------------

    def _digit_at(self, position) -> int | None:
        for index, rect in enumerate(self._digit_rects):
            if rect.left() <= position.x() <= rect.right():
                return index
        return None

    def _half_at(self, index: int, position) -> int:
        rect = self._digit_rects[index]
        return half_at(rect.top(), rect.height(), position.y())

    def mouseMoveEvent(self, event) -> None:
        hover = self._digit_at(event.position())
        half = self._hover_half
        if hover is not None:
            half = self._half_at(hover, event.position())
        if hover != self._hover_digit or half != self._hover_half:
            self._hover_digit = hover
            self._hover_half = half
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        """A click is a nudge of the digit under it, in the half clicked.

        The whole of the tuning story for somebody with no wheel: the readout
        is ten controls rather than a label, and which one a click lands on
        is the decade it moves.
        """
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        index = self._digit_at(event.position())
        if index is None:
            super().mousePressEvent(event)
            return
        steps = self._half_at(index, event.position())
        self._hover_digit = index
        self._hover_half = steps
        self.set_value(nudge_digit(self._value_hz, index, steps), notify=True)
        event.accept()

    def leaveEvent(self, event) -> None:
        self._hover_digit = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        index = self._digit_at(event.position())
        if index is None:
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() // 120 or (
            1 if event.angleDelta().y() > 0 else -1
        )
        self.set_value(nudge_digit(self._value_hz, index, int(steps)), notify=True)
        event.accept()


__all__ = [
    "DIGITS",
    "FrequencyDisplay",
    "digit_step_hz",
    "format_digits",
    "half_at",
    "nudge_digit",
]
