"""The right-hand control column: grouped rows that appear with the level.

Phase 3 takes the listening screen from eight controls to about forty, which
is more than fits on any laptop and far more than belongs in one flat list. So
controls are grouped under headings and the whole column scrolls, and every
row still carries the minimum level at which it appears - promoting a control
stays a one-word change at its call site.

A section hides itself when every row in it is hidden. Without that, Simple
mode would be a screen of headings with nothing underneath them, which reads
as an app that has broken rather than one that is being quiet.

**The viewport is never given a stylesheet.** A stylesheet on a
`QScrollArea`'s viewport drags every descendant through the stylesheet style,
and `QLabel` is a `QFrame`, so each label starts painting a border it never
asked for. The content widget is named and styled instead, and the viewport
gets a palette.

**No row may dictate the column width.** A combo box asks to be as wide as
its longest entry, and the sound-card list contains names like "Speakers
(Realtek(R) Audio)" - which demanded 278 px of a 250 px column. The scroll
area cannot shrink a child below its minimum, and horizontal scrolling is
off, so the surplus was simply cut off the right-hand edge: buttons lost the
end of their centred captions and every spin box lost its arrows. Every field
is passed through `fit_to_column` on the way in, which is what keeps the
column honest.

**And the column is then measured rather than guessed.** `fit_to_column`
stops one field running away with the width; it does not make a fixed 272 px
enough for the widest *row*, which is a caption plus a field plus the space
between them. Measured at Expert, with the scrollbar the column always has,
that came to 305 px - so the fixed width cut 33 px off every spin box's
arrows and the right-hand edge of every combo. `fit_to_contents` asks the
built layout what it needs and adopts that as the minimum the column may
ever be, which is a number that cannot drift when a caption is reworded.
The panel is then a splitter pane rather than a fixed one: the minimum is
what keeps it honest, and the user is free to give it more.

**A row can say what it means.** Passing `topic=` makes the caption a link
into the Learn tab - see `widgets/help.py` - and the panel gathers every one
of them onto a single `helpRequested`, so the listening screen connects once
rather than forty times. It is one more word at the call site, exactly like
`level`, and for the same reason: explaining a control belongs where the
control is declared, not in a second list that will drift out of step with
this one.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..learn import has
from ..levels import Level
from .help import HelpButton, HelpLabel, label_for

# The width the column opens at, before `fit_to_contents` measures what the
# rows actually need. It is a floor rather than a ceiling now: a panel
# narrower than this looks like a strip whatever fits in it.
PANEL_WIDTH = 272
# How much wider than its own minimum the column is allowed to be dragged.
# Not a technical limit - it is that a control column half the window wide
# is a worse screen than a spectrum, and nothing in here gets better with
# more room.
MAX_PANEL_WIDTH = 520
# How little a combo box may shrink to. Small enough that the sound-card list
# cannot widen the panel, large enough that a mode name is still readable.
MIN_FIELD_CHARS = 6
BACKGROUND = "#10151c"
# Where a wrapped control remembers the row it was put in. A Qt property
# rather than an attribute, because the widget is a C++ object whose Python
# wrapper is not guaranteed to be the same one twice.
ROW_PROPERTY = "panelRow"

PANEL_STYLE = """
QWidget#panelContent { background: #10151c; }
QWidget#panelContent QLabel { color: #8b98a5; background: transparent; }
QLabel#sectionTitle {
    color: #5ad1ff; font-size: 10px; font-weight: 700;
    border-top: 1px solid #1d232b;
    padding: 12px 0 4px 0; margin-top: 6px;
}
QLabel#sectionTitle[firstSection="true"] {
    border-top: none; padding-top: 2px; margin-top: 0px;
}
QWidget#panelContent QCheckBox {
    color: #8b98a5; background: transparent; padding: 2px 0;
}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background: #161b22; color: #e6edf3;
    border: 1px solid #2b323b; border-radius: 3px; padding: 2px 6px;
    min-height: 18px;
}
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
    color: #5a6672; border-color: #1d232b;
}
QPushButton#panelButton {
    background: #1b222c; color: #cbd5e0;
    border: 1px solid #2b323b; border-radius: 3px;
    padding: 5px 10px; min-height: 18px;
}
QPushButton#panelButton:hover { background: #232b37; }
QPushButton#panelButton:checked { background: #5ad1ff; color: #0b0e13; }
QPushButton#panelButton:disabled { color: #4a5460; border-color: #1d232b; }
QWidget#panelContent QSlider::groove:horizontal {
    background: #1b222c; border: 1px solid #2b323b;
    border-radius: 2px; height: 4px;
}
QWidget#panelContent QSlider::sub-page:horizontal {
    background: #5ad1ff; border-radius: 2px;
}
QWidget#panelContent QSlider::handle:horizontal {
    background: #e6edf3; border: none; border-radius: 6px;
    width: 12px; margin: -5px 0;
}
"""


def set_row_visible(widget: QWidget, shown: bool) -> None:
    """Show or hide the whole row a control sits in.

    `add_wide` may wrap a control in a row of its own to carry the question
    mark that explains it, and `setVisible` on the control alone then hides
    half a row. Every caller that hides a control for a reason of its own
    goes through here, and a control that was never wrapped is simply itself.
    """
    row = widget.property(ROW_PROPERTY)
    (row if isinstance(row, QWidget) else widget).setVisible(shown)


def fit_to_column(widget: QWidget) -> QWidget:
    """Stop one field from setting the width of the whole column.

    A combo box's minimum width is its longest entry, and the sound-card list
    is full of names nobody chose. Capping the contents length decouples the
    two: the box still fills whatever the column gives it, and elides what
    does not fit rather than pushing the panel wider than it is allowed to be.

    Every field is then told to expand, so they all end at the same place. A
    column of boxes each sized to its own contents has a ragged right edge
    that reads as carelessness even when nobody can say what is wrong with it.
    """
    if isinstance(widget, QComboBox):
        widget.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        widget.setMinimumContentsLength(MIN_FIELD_CHARS)
    elif not isinstance(widget, QAbstractSpinBox):
        return widget
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
    widget.setSizePolicy(policy)
    return widget


class Section:
    """One heading and the rows under it."""

    def __init__(
        self,
        title: str,
        level: Level,
        layout: QFormLayout,
        first: bool = False,
        on_help: Callable[[str], None] | None = None,
    ) -> None:
        self.level = level
        self._layout = layout
        self._on_help = on_help
        self._rows: list[tuple[QWidget, Level]] = []
        # Every help affordance in this section, so a caller can check what a
        # built panel actually offers without walking the widget tree.
        self.topics: list[str] = []

        self.header = QLabel(title.upper())
        self.header.setObjectName("sectionTitle")
        # A rule above each heading, except the first, where a line under the
        # top of the panel would read as a stray border rather than a divider.
        self.header.setProperty("firstSection", "true" if first else "false")
        layout.addRow(self.header)

    def _armed(self, widget: HelpLabel | HelpButton, topic: str) -> None:
        widget.helpRequested.connect(self._on_help)
        self.topics.append(topic)

    def add(
        self,
        label: str,
        widget: QWidget,
        level: Level | None = None,
        topic: str = "",
    ) -> QWidget:
        """A labelled row. `level` defaults to the section's own.

        `topic` makes the caption itself the way in to what it means, which is
        the whole reason this is here rather than in a lookup table keyed on
        the caption text: two rows are called "Threshold" and they are not the
        same threshold.
        """
        at = self.level if level is None else level
        caption = label_for(label, topic)
        if isinstance(caption, HelpLabel) and self._on_help is not None:
            self._armed(caption, topic)
        self._layout.addRow(caption, fit_to_column(widget))
        self._rows.append((caption, at))
        self._rows.append((widget, at))
        return widget

    def add_wide(
        self, widget: QWidget, level: Level | None = None, topic: str = ""
    ) -> QWidget:
        """A row with no caption, for buttons and checkboxes that read alone.

        A check box carries its own text, and clicking that text has to keep
        toggling it - so the way in to the explanation is a question mark
        beside the row rather than the row itself. The returned widget is
        still the one that was passed in; only what the layout holds changes.
        """
        at = self.level if level is None else level
        fit_to_column(widget)
        row: QWidget = widget
        if topic and has(topic) and self._on_help is not None:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(4)
            line.addWidget(widget, 1)
            button = HelpButton(topic)
            self._armed(button, topic)
            line.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._layout.addRow(row)
        # The container, not the child: hiding only the check box would leave
        # an empty row the height of a control at every level below its own.
        self._rows.append((row, at))
        # And the caller is given the way back to it, because a view that
        # hides a control for its own reasons - HD Radio on a build with no
        # decoder - would otherwise leave the question mark beside it
        # floating in the column with nothing to its left.
        widget.setProperty(ROW_PROPERTY, row)
        return widget

    def apply_level(self, level: Level) -> bool:
        """Show what belongs at this level. Returns whether anything is left."""
        visible = False
        for widget, minimum in self._rows:
            shown = level >= minimum
            widget.setVisible(shown)
            visible = visible or shown
        self.header.setVisible(visible)
        return visible


class ControlPanel(QScrollArea):
    """A scrolling column of `Section`s."""

    # Somebody clicked a control's name wanting to know what it means. The
    # panel gathers every row's affordance onto this one signal, so the view
    # above it connects once and knows nothing about which rows offer it.
    helpRequested = QtSignal(str)

    def __init__(self, width: int = PANEL_WIDTH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # What the column asks for when it is first laid out. A splitter takes
        # its opening sizes from the size hints of its panes, so this is how
        # the panel says how wide it would like to be without also insisting
        # on it - `fit_to_contents` raises it to whatever the rows need, and
        # `set_preferred_width` puts back whatever the user last dragged.
        self._preferred = int(width)
        self.setMinimumWidth(int(width))
        self.setMaximumWidth(MAX_PANEL_WIDTH)
        self.setWidgetResizable(True)
        # A scroll area is horizontally Expanding by default, which in a
        # splitter means it takes its share of every pixel the window grows
        # by - at Simple, where three controls are showing, that had the
        # column half as wide again as it needed to be. Preferred keeps it at
        # the width it asked for and gives the spectrum the rest; dragging
        # the handle still overrides it, which is the whole point of a
        # splitter.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # A palette, not a stylesheet - see the module docstring.
        palette = self.viewport().palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
        self.viewport().setPalette(palette)
        self.viewport().setAutoFillBackground(True)

        content = QWidget()
        content.setObjectName("panelContent")
        content.setStyleSheet(PANEL_STYLE)
        outer = QVBoxLayout(content)
        outer.setContentsMargins(10, 6, 10, 12)
        outer.setSpacing(0)

        self._form = QFormLayout()
        self._form.setHorizontalSpacing(10)
        self._form.setVerticalSpacing(7)
        self._form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        outer.addLayout(self._form)
        outer.addStretch(1)
        self.setWidget(content)

        self._sections: list[Section] = []

    # -- width -------------------------------------------------------------

    def sizeHint(self):  # noqa: N802 - Qt's name
        """As tall as a scroll area likes, and exactly as wide as asked for.

        A `QScrollArea` left to itself hints at the size of everything it
        contains, which in a splitter would open this pane over half the
        window and push the spectrum into a strip.
        """
        hint = super().sizeHint()
        hint.setWidth(self._preferred)
        return hint

    def fit_to_contents(self) -> int:
        """Adopt the width the widest level actually needs, and return it.

        Measured at Expert, because that is the most this column can ever be
        asked to show and a hidden row is not in a layout's minimum: measure
        at Simple and Expert is cut off at the right-hand edge, which is the
        fault this exists to end arriving by a new route. Measuring with
        *every* row visible is not the same thing and is worse - it includes
        the rows this level does not have, and sized the column 90 px wider
        than anything it can display.

        The caller sets the level it actually wants immediately afterwards;
        every one of them does so anyway, because the level is what the view
        was constructed with.
        """
        content = self.widget()
        if content is None:
            return self._preferred
        self.set_level(Level.EXPERT)
        needed = content.minimumSizeHint().width() + self._scrollbar_width()
        needed += 2 * self.frameWidth()
        self._preferred = max(self._preferred, needed)
        self.setMinimumWidth(self._preferred)
        self.setMaximumWidth(max(MAX_PANEL_WIDTH, self._preferred))
        self.updateGeometry()
        return self._preferred

    def set_preferred_width(self, width: int) -> int:
        """Open at `width`, or at the minimum if that is wider. Returns it."""
        self._preferred = max(self.minimumWidth(), min(self.maximumWidth(), int(width)))
        self.updateGeometry()
        return self._preferred

    @property
    def preferred_width(self) -> int:
        return self._preferred

    def _scrollbar_width(self) -> int:
        """The scrollbar's own width, which the column always loses.

        Always, not sometimes: forty rows do not fit on any screen this app
        runs on, so the vertical bar is permanently there and the viewport is
        permanently that much narrower than the pane.

        The hint, never the current width: a widget that has not been shown
        yet is 100 px wide by default, whatever it is, and asking a scrollbar
        how wide it is before anybody has seen it made the column 88 px too
        wide - which is the same class of mistake as measuring the rows
        before a level has hidden any of them.
        """
        return self.verticalScrollBar().sizeHint().width()

    def section(self, title: str, level: Level = Level.STANDARD) -> Section:
        section = Section(
            title,
            level,
            self._form,
            first=not self._sections,
            on_help=self.helpRequested.emit,
        )
        self._sections.append(section)
        return section

    def set_level(self, level: Level) -> None:
        for section in self._sections:
            section.apply_level(level)

    @property
    def topics(self) -> tuple[str, ...]:
        """Every topic this panel offers a way in to, in the order built.

        Exists for the test that asserts each one actually has an article
        behind it. A control that looks like a link and does nothing is the
        one failure this feature can have that nobody would report.
        """
        return tuple(topic for section in self._sections for topic in section.topics)


def on_change(widget: QWidget, slot: Callable[..., None]) -> None:
    """Connect whichever "the user changed this" signal a widget has.

    Saves every call site from remembering that a check box toggles, a spin
    box changes its value and a combo box changes its index.
    """
    for name in ("toggled", "valueChanged", "currentIndexChanged", "clicked"):
        signal = getattr(widget, name, None)
        if signal is not None:
            signal.connect(slot)
            return
    raise TypeError(f"{type(widget).__name__} has no change signal to connect")


__all__ = [
    "MAX_PANEL_WIDTH",
    "MIN_FIELD_CHARS",
    "ROW_PROPERTY",
    "PANEL_STYLE",
    "PANEL_WIDTH",
    "ControlPanel",
    "Section",
    "fit_to_column",
    "on_change",
    "set_row_visible",
]
