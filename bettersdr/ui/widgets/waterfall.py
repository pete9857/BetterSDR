"""The waterfall: a scrolling history of the spectrum.

The scrolling is the part worth being careful about. A history of 400 rows by
4096 bins is 6.5 MB, and shifting that array once per frame at 30 Hz would
spend 200 MB/s of memory bandwidth doing nothing useful. So rows are written
into a ring at a rolling index and never moved.

The trick that makes the ring displayable without a copy is storing every row
twice, `rows` apart. Any window of `rows` consecutive entries in the doubled
array is then a contiguous slice, so the oldest-to-newest view is a plain
NumPy view - no `np.roll`, no per-frame allocation.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QTransform

from . import colormaps
from .axes import BlankAxis
from .viewspan import PanZoom

DEFAULT_ROWS = 400
# Nothing has been received yet, so start below any plausible signal rather
# than at zero, which would paint the whole history at full brightness.
EMPTY_DB = -140.0

# The tuned-frequency marker. Two lines, not one: a dark backing under a
# bright core. Every other marker in the app is drawn on a background the app
# chose, but this one is drawn on the picture itself - and the picture is at
# its brightest precisely where the marker goes, because the thing being
# listened to is the strongest signal on screen. White alone disappears into a
# strong station; black alone disappears into an empty band.
MARKER_COLOUR = "#ffffff"
MARKER_OUTLINE = QColor(0, 0, 0, 190)
MARKER_WIDTH = 1
MARKER_OUTLINE_WIDTH = 3


class WaterfallHistory:
    """A rolling spectrum history with no per-frame copying."""

    def __init__(self, rows: int = DEFAULT_ROWS, bins: int = 1024) -> None:
        self._allocate(rows, bins)

    def _allocate(self, rows: int, bins: int) -> None:
        if rows < 1 or bins < 1:
            raise ValueError(f"rows and bins must be positive, got {rows}x{bins}")
        self.rows = int(rows)
        self.bins = int(bins)
        self._data = np.full((self.rows * 2, self.bins), EMPTY_DB, dtype=np.float32)
        # Index of the newest row. Starts at the end so the first push lands
        # at 0 and the history fills forwards.
        self._newest = self.rows - 1
        self.pushed = 0

    def clear(self) -> None:
        self._data.fill(EMPTY_DB)
        self._newest = self.rows - 1
        self.pushed = 0

    def resize(self, rows: int | None = None, bins: int | None = None) -> None:
        """Change the shape. History is discarded - it no longer lines up."""
        rows = self.rows if rows is None else int(rows)
        bins = self.bins if bins is None else int(bins)
        if rows != self.rows or bins != self.bins:
            self._allocate(rows, bins)

    def push(self, row_db: np.ndarray) -> None:
        """Add one spectrum row as the newest line.

        A row of the wrong width resizes the history rather than raising: the
        FFT size is a live display setting, and a user changing it should see
        the waterfall restart, not the app fall over.
        """
        row = np.asarray(row_db, dtype=np.float32)
        if row.size != self.bins:
            self.resize(bins=row.size)
        self._newest = (self._newest + 1) % self.rows
        self._data[self._newest] = row
        self._data[self._newest + self.rows] = row
        self.pushed += 1

    def image(self) -> np.ndarray:
        """The history oldest-first, as a view. Row 0 is the oldest line."""
        start = self._newest + 1
        return self._data[start : start + self.rows]


class WaterfallWidget(PanZoom, pg.PlotWidget):
    """The waterfall pane, sharing its x axis with the spectrum above it.

    Sharing the axis now means sharing a *view* of it as well: the same
    click, drag and wheel as the spectrum, handled by the same code, because
    two stacked panes that scrolled differently would be one picture cut in
    half rather than two views of one window.
    """

    tuneRequested = Signal(float)
    viewChanged = Signal(float, float)

    def __init__(
        self,
        rows: int = DEFAULT_ROWS,
        colour_map: str = colormaps.DEFAULT_NAME,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent=parent, axisItems={"left": BlankAxis()})
        self._init_pan_zoom()
        self.history = WaterfallHistory(rows=rows)
        self._center_hz = 0.0
        self._span_hz = 1.0
        self._frames_per_row = 1
        self._pending: list[np.ndarray] = []

        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setBackground("#0b0e13")
        plot = self.getPlotItem()
        plot.showAxis("bottom", False)
        plot.setContentsMargins(0, 0, 0, 0)
        plot.getViewBox().setDefaultPadding(0.0)

        self._image = pg.ImageItem(axisOrder="row-major")
        plot.addItem(self._image)

        # Solid rather than dashed, unlike the spectrum's cursor above it,
        # because this one has to be followed down a moving picture rather
        # than read against a still background - and because the column it
        # covers carries no data anyway: `psd.py` nulls the DC bin, which is
        # exactly this frequency. That null is what the marker replaces. It
        # was already visible as a thin black line down the middle, which
        # reads as a fault in the display rather than as the place the radio
        # is listening.
        self._cursor_outline = self._marker(MARKER_OUTLINE, MARKER_OUTLINE_WIDTH, 9)
        self._cursor = self._marker(MARKER_COLOUR, MARKER_WIDTH, 10)

        self.set_colour_map(colour_map)
        self.set_range(-90.0, -20.0)

    def _marker(self, colour, width: int, z: float) -> pg.InfiniteLine:
        """One vertical hairline above the image, drawn but never touched.

        Not movable, so it accepts no hover or mouse events of its own and
        cannot come between a press and the pan, click-to-tune and wheel zoom
        the pane behind it owns.
        """
        line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(colour, width=width))
        line.setZValue(z)
        line.setVisible(False)
        self.getPlotItem().addItem(line)
        return line

    # -- appearance --------------------------------------------------------

    def set_colour_map(self, name: str) -> None:
        table = colormaps.lookup_table(name)
        self.colour_map = name
        self._image.setLookupTable(table)

    def set_range(self, floor_db: float, ceiling_db: float) -> None:
        """The dB window mapped across the colour ramp."""
        if ceiling_db <= floor_db:
            ceiling_db = floor_db + 1.0
        self.floor_db = float(floor_db)
        self.ceiling_db = float(ceiling_db)
        self._image.setLevels((self.floor_db, self.ceiling_db))

    def set_tuned(self, tuned_hz: float) -> None:
        """Where the radio is listening, marked down the height of the pane.

        Told rather than derived from `push`, for the same reason the ribbon
        is: the centre of the captured window and the frequency being listened
        to are two different questions that happen to share an answer today.

        It is what makes panning aimable. Drag the picture sideways and the
        marker travels with the dial, so the distance between it and the
        signal under the pointer is visible rather than something to be worked
        out from the axis.
        """
        self._cursor.setPos(float(tuned_hz))
        self._cursor_outline.setPos(float(tuned_hz))
        self._cursor.setVisible(True)
        self._cursor_outline.setVisible(True)

    def set_speed(self, frames_per_row: int) -> None:
        """Average N display frames into one line, to slow the scroll.

        Scroll speed is decoupled from the frame rate on purpose: a slow
        waterfall should not mean a sluggish spectrum above it.
        """
        self._frames_per_row = max(1, int(frames_per_row))
        self._pending.clear()

    # -- data --------------------------------------------------------------

    def push(self, spectrum_db: np.ndarray, center_hz: float, span_hz: float) -> None:
        if center_hz != self._center_hz or span_hz != self._span_hz:
            self._center_hz, self._span_hz = center_hz, span_hz
            self.history.clear()
        # Before the early return below: the pane must follow a retune on the
        # frame it happens, not on whichever later one completes a slow row.
        self.set_window(center_hz, span_hz)

        self._pending.append(np.asarray(spectrum_db, dtype=np.float32))
        if len(self._pending) < self._frames_per_row:
            return
        row = (
            self._pending[0]
            if self._frames_per_row == 1
            else np.mean(self._pending, axis=0)
        )
        self._pending.clear()
        self.history.push(row)
        self._redraw()

    def _redraw(self) -> None:
        image = self.history.image()
        self._image.setImage(image, autoLevels=False)
        # Map image columns onto real frequency so the pane lines up with the
        # spectrum above without either of them knowing about the other.
        low = self._center_hz - self._span_hz / 2.0
        transform = QTransform()
        transform.translate(low, 0.0)
        transform.scale(self._span_hz / max(1, image.shape[1]), 1.0)
        self._image.setTransform(transform)
        # Until the history fills, show only the rows that hold real data.
        # Otherwise the pane opens as a large black rectangle that looks like
        # a fault rather than like a waterfall that has just started.
        filled = min(self.history.rows, max(1, self.history.pushed))
        self.setYRange(image.shape[0] - filled, image.shape[0], padding=0.0)


__all__ = [
    "DEFAULT_ROWS",
    "EMPTY_DB",
    "MARKER_COLOUR",
    "MARKER_OUTLINE",
    "WaterfallHistory",
    "WaterfallWidget",
]
