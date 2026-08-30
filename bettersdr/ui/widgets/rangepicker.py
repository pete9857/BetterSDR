"""The whole tunable dial, as a list of things to point the receiver at.

The band chips on the discovery screen are a deliberately short list: the
handful of allocations that are worth putting in front of somebody who has
never used a receiver, because they are busy, legal to listen to and
interesting on the first try. That is the right list for Simple and Standard
and the wrong one for Expert, where "beginner-friendly must not mean
capability-capped" is the whole argument - an expert wanting to know what is
on 150.8-156 MHz should not have to leave for SDR# to find out.

So at Expert the chips are replaced by this: every stretch of dial the dongle
can reach, from 500 kHz to 1.766 GHz, with no holes in it. Where the band plan
has a band, the band; where it only knows who the stretch is licensed to, that
- "Business and public safety", "Federal government" - and where it knows
neither, an honest "Unallocated". Each is labelled with its own span, because
that is the fact that distinguishes the two stretches both called Federal
government, and because the span is what a reader needs in order to decide
whether it is the one they meant.

Several may be selected at once. That is not a convenience: a scanner user
watching their local public-safety traffic wants 150.8-156 MHz *and* 453 MHz
*and* the 700 MHz pairs, and asking them to choose one and sweep three times
is asking them to do the app's job. What comes out is `sweep_ranges`, which is
the selection turned into something a sweep can be planned from.

No engine, no device and no threads, the same as every other widget here. It
emits what was chosen; the discovery screen decides what to do about it.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtCore import Signal as QtSignal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.device import MAX_TUNE_HZ, MIN_TUNE_HZ
from ...core.frontend import DEFAULT_SAMPLE_RATE, safe_sample_rate
from ...scan import bandplan
from ...scan.classifier import format_bandwidth, format_frequency
from ...scan.sweeper import plan_steps
from .icons import glyph

# How long one step of a sweep costs, end to end: retune, settle, dwell and
# measure. Not a guess - the FM band is 12 steps and three passes of it take
# 5.1 s, and the AM band is 9 steps and three passes take 4.9 s, which are
# 0.142 and 0.181 seconds a step. It is only ever used to say "about a
# minute", so the spread between the two does not matter.
SECONDS_PER_STEP = 0.15

# Tall enough to show several ranges without scrolling and short enough that
# the list of what was found is still the biggest thing on the screen. The
# list below it is the point of the screen; this is how you aim it.
PICKER_HEIGHT_PX = 190

PICKER_STYLE = """
QWidget#rangePicker { background: #0b0e13; }
QScrollArea#rangeScroll {
    border: 1px solid #1d232b; border-radius: 4px; background: #0e1218;
}
QWidget#rangeList { background: #0e1218; }
QCheckBox#range {
    color: #8b98a5; font-size: 12px; padding: 2px 6px; spacing: 7px;
}
QCheckBox#range:hover { color: #e6edf3; }
QCheckBox#range:checked { color: #e6edf3; }
QCheckBox#range[band="true"] { color: #b9c6d3; }
QCheckBox#range[band="true"]:checked { color: #e6edf3; font-weight: 600; }
QCheckBox::indicator {
    width: 13px; height: 13px; border-radius: 3px;
    border: 1px solid #2b323b; background: #10151c;
}
QCheckBox::indicator:hover { border-color: #3d4650; }
QCheckBox::indicator:checked { background: #5ad1ff; border-color: #5ad1ff; }
QLabel#rangeSummary { color: #6d7b89; font-size: 11px; }
QPushButton#rangeAction {
    background: transparent; color: #5ad1ff; border: none;
    padding: 2px 6px; font-size: 11px;
}
QPushButton#rangeAction:hover { color: #7cdcff; }
QPushButton#rangeAction:disabled { color: #3d4650; }
"""


def span_label(segment: bandplan.Segment) -> str:
    """"88 MHz-108 MHz: FM Radio" - the span first, then what is there.

    The span leads because it is the half that is always true and always
    distinct. Two stretches of dial are both named "Federal government"; no
    two start in the same place.
    """
    return (
        f"{format_frequency(segment.start_hz)}-"
        f"{format_frequency(segment.end_hz)}: {segment.name}"
    )


def effective_rate(segment: bandplan.Segment) -> int:
    """The window this stretch of dial will actually be swept through.

    What it asked for, narrowed by the guard that keeps the window clear of
    0 Hz. The engine applies the same guard again when it plans the sweep, so
    this is not a second opinion - it is the same one, asked early enough to
    decide which stretches may be merged with which.
    """
    return safe_sample_rate(
        segment.start_hz,
        preferred_hz=int(segment.sample_rate_hz or DEFAULT_SAMPLE_RATE),
    )


def ranges_for(
    segments: Iterable[bandplan.Segment],
) -> tuple[tuple[int, int, int | None], ...]:
    """A selection turned into ranges to sweep, merged by the window each gets."""
    return bandplan.sweep_ranges(segments, rate_for=effective_rate)


def step_count(ranges: tuple[tuple[int, int, int | None], ...]) -> int:
    """How many tuner steps one pass over a selection comes to.

    Planned exactly as the engine will plan it, through the window each range
    will actually get - `safe_sample_rate` narrows the AM band to 240 kHz and
    that is nine times as many steps per megahertz. An estimate derived from
    the total width instead would be out by a factor of ten on any selection
    containing the bottom of the dial.
    """
    return sum(
        len(
            plan_steps(
                low_hz,
                high_hz,
                safe_sample_rate(
                    low_hz, preferred_hz=int(rate or DEFAULT_SAMPLE_RATE)
                ),
            )
        )
        for low_hz, high_hz, rate in ranges
    )


def duration_phrase(steps: int) -> str:
    """How long a pass will take, in the units somebody would say it in."""
    seconds = steps * SECONDS_PER_STEP
    if seconds < 90:
        return f"about {max(1, round(seconds))}s a pass"
    return f"about {round(seconds / 60)} min a pass"


def summarise(segments: tuple[bandplan.Segment, ...]) -> str:
    """What the current selection amounts to, said out loud under the list.

    Selecting the whole dial is a legitimate thing to ask for and it is a
    two-and-a-half minute sweep. Somebody who has just ticked sixty boxes has
    no way to know that, and a progress bar that crawls for two minutes looks
    exactly like an application that has hung - so the cost is stated before
    the button is pressed rather than discovered afterwards.
    """
    if not segments:
        return "Nothing selected. Tick a range to scan it."
    ranges = ranges_for(segments)
    width = sum(high - low for low, high, _ in ranges)
    steps = step_count(ranges)
    chosen = (
        segments[0].name if len(segments) == 1 else f"{len(segments)} ranges"
    )
    return (
        f"{chosen}  -  {format_bandwidth(width)} of dial  -  "
        f"{steps} steps, {duration_phrase(steps)}"
    )


class RangePicker(QWidget):
    """A scrolling, multiple-choice list of everywhere the receiver can go."""

    selectionChanged = QtSignal()

    def __init__(
        self,
        low_hz: int = MIN_TUNE_HZ,
        high_hz: int = MAX_TUNE_HZ,
        region: str = bandplan.DEFAULT_REGION,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.segments: tuple[bandplan.Segment, ...] = bandplan.coverage(
            low_hz, high_hz, region
        )
        self._boxes: dict[str, QCheckBox] = {}
        self._quiet = False
        self._build()
        self._refresh_summary()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self.setObjectName("rangePicker")
        self.setStyleSheet(PICKER_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self.area = QScrollArea()
        self.area.setObjectName("rangeScroll")
        self.area.setWidgetResizable(True)
        self.area.setMaximumHeight(PICKER_HEIGHT_PX)
        self.area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        holder = QWidget()
        holder.setObjectName("rangeList")
        column = QVBoxLayout(holder)
        column.setContentsMargins(6, 6, 6, 6)
        column.setSpacing(1)
        for segment in self.segments:
            box = QCheckBox(f"{self._icon(segment)}  {span_label(segment)}")
            box.setObjectName("range")
            box.setCursor(Qt.CursorShape.PointingHandCursor)
            # A band is a place the app has something to offer and an
            # allocation is only a statement of who owns the space, so they
            # are not weighted the same. Both are listed, because the
            # difference is about what to expect, not about what is allowed.
            box.setProperty("band", "true" if segment.band is not None else "false")
            box.setToolTip(segment.description or span_label(segment))
            box.toggled.connect(self._toggled)
            column.addWidget(box)
            self._boxes[segment.key] = box
        column.addStretch(1)
        self.area.setWidget(holder)
        outer.addWidget(self.area)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.summary = QLabel("")
        self.summary.setObjectName("rangeSummary")
        row.addWidget(self.summary)
        row.addStretch(1)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("rangeAction")
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear)
        row.addWidget(self.clear_button)
        self.all_button = QPushButton("Select the whole dial")
        self.all_button.setObjectName("rangeAction")
        self.all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.all_button.clicked.connect(self.select_all)
        row.addWidget(self.all_button)
        outer.addLayout(row)

    @staticmethod
    def _icon(segment: bandplan.Segment) -> str:
        """A band wears its own icon; everything else wears one neutral mark.

        Giving an allocation the fallback aerial would say the app has
        something for it, which is exactly the distinction this list has to
        keep: it can tune there and it has nothing prepared for it.
        """
        return glyph(segment.icon) if segment.band is not None else "·"

    # -- what is chosen ----------------------------------------------------

    @property
    def selection(self) -> tuple[bandplan.Segment, ...]:
        """Everything ticked, in frequency order."""
        return tuple(
            segment
            for segment in self.segments
            if self._boxes[segment.key].isChecked()
        )

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(segment.key for segment in self.selection)

    def ranges(self) -> tuple[tuple[int, int, int | None], ...]:
        """The selection as ranges to sweep, merged and each with its window."""
        return ranges_for(self.selection)

    def set_keys(self, keys: object) -> None:
        """Tick exactly these, silently. Unknown keys are ignored.

        Silently because this is how a remembered selection is restored, and
        a restore that emitted would read as sixty clicks - which for a
        running monitor session means sixty restarts.
        """
        wanted = (
        {str(key) for key in keys}
        if isinstance(keys, (list, tuple))
        else set()
    )
        self._quiet = True
        try:
            for key, box in self._boxes.items():
                box.setChecked(key in wanted)
        finally:
            self._quiet = False
        self._refresh_summary()

    def set_band(self, band: bandplan.Band | None) -> None:
        """Tick the entry for one band and nothing else, silently.

        The bridge between the two pickers: arriving at Expert with FM Radio
        chosen on the chips should show FM Radio ticked here, not an empty
        list and a disabled Scan button.
        """
        self.set_keys(() if band is None else (bandplan.Segment.of(band).key,))

    def clear(self) -> None:
        self._set_all(False)

    def select_all(self) -> None:
        self._set_all(True)

    def _set_all(self, checked: bool) -> None:
        self._quiet = True
        try:
            for box in self._boxes.values():
                box.setChecked(checked)
        finally:
            self._quiet = False
        self._refresh_summary()
        self.selectionChanged.emit()

    def _toggled(self, _checked: bool) -> None:
        if self._quiet:
            return
        self._refresh_summary()
        self.selectionChanged.emit()

    def _refresh_summary(self) -> None:
        selection = self.selection
        self.summary.setText(summarise(selection))
        self.clear_button.setEnabled(bool(selection))
        self.all_button.setEnabled(len(selection) < len(self.segments))


__all__ = [
    "PICKER_HEIGHT_PX",
    "SECONDS_PER_STEP",
    "RangePicker",
    "duration_phrase",
    "effective_rate",
    "ranges_for",
    "span_label",
    "step_count",
    "summarise",
]
