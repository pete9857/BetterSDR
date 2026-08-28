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
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..levels import Level

# Wide enough for the longest label plus a field at a usable size, once
# `fit_to_column` has stopped the combo boxes asking for more. Measured, not
# guessed: the widest two-column row is "Offset tuning" at 70 + 12 + 118.
PANEL_WIDTH = 272
# How little a combo box may shrink to. Small enough that the sound-card list
# cannot widen the panel, large enough that a mode name is still readable.
MIN_FIELD_CHARS = 6
BACKGROUND = "#10151c"

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
    ) -> None:
        self.level = level
        self._layout = layout
        self._rows: list[tuple[QWidget, Level]] = []

        self.header = QLabel(title.upper())
        self.header.setObjectName("sectionTitle")
        # A rule above each heading, except the first, where a line under the
        # top of the panel would read as a stray border rather than a divider.
        self.header.setProperty("firstSection", "true" if first else "false")
        layout.addRow(self.header)

    def add(self, label: str, widget: QWidget, level: Level | None = None) -> QWidget:
        """A labelled row. `level` defaults to the section's own."""
        at = self.level if level is None else level
        caption = QLabel(label)
        caption.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._layout.addRow(caption, fit_to_column(widget))
        self._rows.append((caption, at))
        self._rows.append((widget, at))
        return widget

    def add_wide(self, widget: QWidget, level: Level | None = None) -> QWidget:
        """A row with no caption, for buttons and checkboxes that read alone."""
        at = self.level if level is None else level
        self._layout.addRow(fit_to_column(widget))
        self._rows.append((widget, at))
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

    def __init__(self, width: int = PANEL_WIDTH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(width)
        self.setWidgetResizable(True)
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

    def section(self, title: str, level: Level = Level.STANDARD) -> Section:
        section = Section(title, level, self._form, first=not self._sections)
        self._sections.append(section)
        return section

    def set_level(self, level: Level) -> None:
        for section in self._sections:
            section.apply_level(level)


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
    "MIN_FIELD_CHARS",
    "PANEL_STYLE",
    "PANEL_WIDTH",
    "ControlPanel",
    "Section",
    "fit_to_column",
    "on_change",
]
