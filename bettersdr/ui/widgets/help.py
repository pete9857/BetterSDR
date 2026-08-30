"""Making a control's own name the way in to what it means.

The Learn tab would be a reference manual nobody opens if the only route to it
were the Learn tab. The route that matters is the other one: somebody is
looking at a control called "Threshold", does not know what a threshold is,
and clicks the word. That is the moment the explanation is wanted, and it is
the only moment at which the app can be sure of it.

Two shapes, because the control column has two shapes of row:

`HelpLabel` is the caption of a labelled row - "RF gain", "Bandwidth". The
whole caption is the link, because the caption is what the reader is looking
at and puzzled by.

`HelpButton` is a small question mark for rows that have no caption. A check
box carries its own text and clicking that text has to keep toggling it, so
the affordance goes beside it rather than on it.

Three rules hold for both:

**Nothing looks clickable unless something is behind it.** Both are built
through `attach`, which asks `learn.has()` first and hands back the plain
widget when the answer is no. A label that underlines on hover and then does
nothing is worse than one that never offered.

**The affordance has to be visible without shouting.** These sit beside forty
controls; a row of bright links would compete with the controls themselves.
So the resting state is exactly the ordinary caption colour and only hover
lights up - which is also why hover is handled here rather than in a
stylesheet, since Qt has no hover selector for the anchors inside a QLabel.

**A tooltip says what will happen.** "What does this mean?" rather than the
article's own summary: the promise being made is that clicking opens an
explanation, and a tooltip that gave the explanation would make the click
pointless.
"""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from ..learn import get, has

# The caption colour of every other row in the panel. A help label at rest is
# indistinguishable from a caption that is not one, deliberately: discovering
# that a word is clickable is the reward for pointing at it, not a demand made
# of every reader on every row.
REST = "#8b98a5"
HOVER = "#5ad1ff"

TOOLTIP = "What does this mean?"

HELP_BUTTON_STYLE = """
QPushButton#helpButton {
    background: transparent; color: #4d5865; border: none;
    padding: 0px; font-size: 11px; font-weight: 700;
}
QPushButton#helpButton:hover { color: #5ad1ff; }
"""


def tooltip_for(topic: str) -> str:
    """The hover text: the promise, plus what is being explained.

    The article's own title is included because a caption is often shorter
    than the thing it names - "Threshold" is the squelch threshold on one row
    and the gain rider's threshold on another, and the tooltip is where that
    ambiguity gets resolved before the click rather than after it.
    """
    article = get(topic)
    if article is None:
        return TOOLTIP
    return f"{TOOLTIP}  -  {article.title}"


class HelpLabel(QLabel):
    """A row caption that is also the way in to what the row means."""

    helpRequested = QtSignal(str)

    def __init__(
        self, text: str, topic: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._caption = text
        self._topic = topic
        self.setTextFormat(Qt.TextFormat.RichText)
        # Links only. `TextSelectableByMouse` would let a drag across the
        # caption start a selection instead of arming the click.
        self.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip_for(topic))
        self.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.linkActivated.connect(lambda _href: self.helpRequested.emit(self._topic))
        self._paint(REST)

    @property
    def topic(self) -> str:
        return self._topic

    def _paint(self, colour: str) -> None:
        """Redraw the caption in one colour.

        The underline is always there and always the caption's own colour, so
        the row's height and baseline never move between resting and hovered -
        a label that grew an underline on hover would nudge the field beside
        it by a pixel, forty rows down a scrolling column.
        """
        self.setText(
            f'<a href="{html.escape(self._topic)}" '
            f'style="color: {colour}; text-decoration: none; '
            f'border-bottom: 1px dotted {colour};">'
            f"{html.escape(self._caption)}</a>"
        )

    def enterEvent(self, event) -> None:
        self._paint(HOVER)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._paint(REST)
        super().leaveEvent(event)


class HelpButton(QPushButton):
    """A question mark, for a row whose own text cannot be the link."""

    helpRequested = QtSignal(str)

    def __init__(self, topic: str, parent: QWidget | None = None) -> None:
        super().__init__("?", parent)
        self._topic = topic
        self.setObjectName("helpButton")
        self.setStyleSheet(HELP_BUTTON_STYLE)
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip_for(topic))
        # Fixed and narrow. It shares a row with a check box that has to keep
        # every pixel it had, and a button that claimed its natural width
        # would push captions off the end of a 272 px column.
        self.setFixedWidth(14)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clicked.connect(lambda: self.helpRequested.emit(self._topic))

    @property
    def topic(self) -> str:
        return self._topic


def label_for(text: str, topic: str) -> QLabel:
    """The caption for a row: a `HelpLabel` if anything explains it.

    Returning a plain `QLabel` rather than raising is the deliberate half of
    this. A topic with no article is a mistake, and it is caught by
    `tests/test_learn.py`, which walks every topic every view names - but the
    right behaviour at runtime is a caption that is merely ordinary, not a
    window that will not open.
    """
    if topic and has(topic):
        return HelpLabel(text, topic)
    plain = QLabel(text)
    plain.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return plain


__all__ = [
    "HOVER",
    "REST",
    "TOOLTIP",
    "HelpButton",
    "HelpLabel",
    "label_for",
    "tooltip_for",
]
