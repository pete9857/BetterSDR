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
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTransform

from . import colormaps
from .axes import BlankAxis

DEFAULT_ROWS = 400
# Nothing has been received yet, so start below any plausible signal rather
# than at zero, which would paint the whole history at full brightness.
EMPTY_DB = -140.0


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


class WaterfallWidget(pg.PlotWidget):
    """The waterfall pane, sharing its x axis with the spectrum above it."""

    tuneRequested = Signal(float)

    def __init__(
        self,
        rows: int = DEFAULT_ROWS,
        colour_map: str = colormaps.DEFAULT_NAME,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent=parent, axisItems={"left": BlankAxis()})
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
        self.set_colour_map(colour_map)
        self.set_range(-90.0, -20.0)

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
        self.setXRange(low, low + self._span_hz, padding=0.0)
        # Until the history fills, show only the rows that hold real data.
        # Otherwise the pane opens as a large black rectangle that looks like
        # a fault rather than like a waterfall that has just started.
        filled = min(self.history.rows, max(1, self.history.pushed))
        self.setYRange(image.shape[0] - filled, image.shape[0], padding=0.0)

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.getPlotItem().vb.mapSceneToView(event.position())
            self.tuneRequested.emit(float(point.x()))
        super().mousePressEvent(event)


__all__ = ["DEFAULT_ROWS", "EMPTY_DB", "WaterfallHistory", "WaterfallWidget"]
