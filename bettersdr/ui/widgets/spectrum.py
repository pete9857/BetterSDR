"""The spectrum analyser pane, plus the band-plan ribbon above it.

The ribbon is not decoration. It is the best passive teaching device in the
app: a beginner who tunes around sees "Aircraft", "Weather Radio", "Marine
VHF" slide past under the cursor and learns the layout of the spectrum without
reading anything. It is also straight SDR# parity, so it earns its space
twice.

Levels are the calibrated dBFS that `dsp/psd.py` produces, which is what lets
the y axis carry real numbers instead of arbitrary units.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from ...dsp.psd import PeakHold, noise_floor_db
from ...scan import bandplan
from .axes import AXIS_WIDTH, BlankAxis, FrequencyAxis

BACKGROUND = "#0b0e13"
GRID_ALPHA = 0.18
TRACE_COLOUR = "#5ad1ff"
PEAK_COLOUR = "#f0a24a"
PASSBAND_COLOUR = QColor(90, 209, 255, 40)


class BandRibbon(pg.PlotWidget):
    """Coloured allocation blocks aligned to the spectrum's frequency axis."""

    def __init__(self, region: str = bandplan.DEFAULT_REGION, parent=None) -> None:
        super().__init__(parent=parent, axisItems={"left": BlankAxis()})
        self.region = region
        self._span: tuple[float, float] | None = None
        self._items: list[pg.GraphicsObject] = []

        self.setBackground(BACKGROUND)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setFixedHeight(24)
        plot = self.getPlotItem()
        plot.showAxis("bottom", False)
        plot.setContentsMargins(0, 0, 0, 0)
        plot.getViewBox().setDefaultPadding(0.0)
        self.setYRange(0.0, 1.0, padding=0.0)

    def set_span(self, low_hz: float, high_hz: float) -> None:
        if self._span == (low_hz, high_hz):
            return
        self._span = (low_hz, high_hz)
        plot = self.getPlotItem()
        for item in self._items:
            plot.removeItem(item)
        self._items.clear()

        for band in bandplan.overlapping(low_hz, high_hz, self.region):
            colour = QColor(band.colour)
            block = pg.LinearRegionItem(
                values=(max(band.start_hz, low_hz), min(band.end_hz, high_hz)),
                brush=QColor(colour.red(), colour.green(), colour.blue(), 110),
                pen=pg.mkPen(colour, width=1),
                movable=False,
            )
            block.setZValue(-10)
            plot.addItem(block)
            self._items.append(block)

            # Label at the visible centre of the block, not the band's own
            # centre, so a band running off the edge still reads.
            middle = (max(band.start_hz, low_hz) + min(band.end_hz, high_hz)) / 2.0
            visible = min(band.end_hz, high_hz) - max(band.start_hz, low_hz)
            if visible < (high_hz - low_hz) * 0.06:
                continue  # too narrow on screen for text to fit honestly
            label = pg.TextItem(band.name, color="#e6edf3", anchor=(0.5, 0.5))
            label.setPos(middle, 0.5)
            plot.addItem(label)
            self._items.append(label)

        self.setXRange(low_hz, high_hz, padding=0.0)


class SpectrumWidget(pg.PlotWidget):
    """Live spectrum with peak hold and a draggable passband."""

    tuneRequested = Signal(float)
    bandwidthChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent, axisItems={"bottom": FrequencyAxis("bottom")})
        self._center_hz = 0.0
        self._tuned_hz = 0.0
        self._bandwidth_hz = 200_000.0
        self._peak = PeakHold()
        self._show_peak = True
        self._updating_passband = False

        self.setBackground(BACKGROUND)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        plot = self.getPlotItem()
        plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        plot.setLabel("left", "dBFS")
        plot.getViewBox().setDefaultPadding(0.0)
        plot.getAxis("left").setTextPen("#8b98a5")
        plot.getAxis("left").setWidth(AXIS_WIDTH)
        plot.getAxis("bottom").setTextPen("#8b98a5")

        self._peak_curve = plot.plot(pen=pg.mkPen(PEAK_COLOUR, width=1))
        self._curve = plot.plot(pen=pg.mkPen(TRACE_COLOUR, width=1))

        self._passband = pg.LinearRegionItem(
            brush=PASSBAND_COLOUR,
            pen=pg.mkPen("#5ad1ff", width=1),
            movable=True,
        )
        self._passband.setZValue(-5)
        plot.addItem(self._passband)
        self._passband.sigRegionChanged.connect(self._passband_dragged)

        self._cursor = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine)
        )
        plot.addItem(self._cursor)

        self.setYRange(-110.0, -10.0, padding=0.0)

    # -- data --------------------------------------------------------------

    def update_spectrum(
        self, spectrum_db: np.ndarray, center_hz: float, sample_rate: float
    ) -> None:
        bins = spectrum_db.size
        if bins == 0:
            return
        if center_hz != self._center_hz:
            self._center_hz = center_hz
            self._peak.reset()

        freqs = center_hz + (np.arange(bins) - bins // 2) * (sample_rate / bins)
        self._curve.setData(freqs, spectrum_db)
        if self._show_peak:
            self._peak_curve.setData(freqs, self._peak.update(spectrum_db))

        self.setXRange(freqs[0], freqs[-1], padding=0.0)

    def set_peak_hold(self, enabled: bool) -> None:
        self._show_peak = bool(enabled)
        self._peak.reset()
        if not enabled:
            self._peak_curve.setData([], [])

    def reset_peak_hold(self) -> None:
        self._peak.reset()

    def auto_range(self, spectrum_db: np.ndarray) -> None:
        """Fit the y axis to the current noise floor and peak.

        Anchored on the floor rather than the mean, so one strong station does
        not push everything else off the bottom of the display.
        """
        if spectrum_db.size == 0:
            return
        floor = noise_floor_db(spectrum_db)
        peak = float(np.max(spectrum_db))
        self.setYRange(floor - 8.0, max(peak + 8.0, floor + 30.0), padding=0.0)

    # -- passband ----------------------------------------------------------

    def set_passband(self, tuned_hz: float, bandwidth_hz: float) -> None:
        self._tuned_hz = float(tuned_hz)
        self._bandwidth_hz = float(bandwidth_hz)
        self._updating_passband = True
        try:
            half = self._bandwidth_hz / 2.0
            self._passband.setRegion((tuned_hz - half, tuned_hz + half))
            self._cursor.setPos(tuned_hz)
        finally:
            self._updating_passband = False

    def _passband_dragged(self) -> None:
        if self._updating_passband:
            return
        low, high = self._passband.getRegion()
        centre = (low + high) / 2.0
        width = abs(high - low)
        if abs(centre - self._tuned_hz) > 1.0:
            self.tuneRequested.emit(float(centre))
        if abs(width - self._bandwidth_hz) > 1.0:
            self.bandwidthChanged.emit(float(width))

    # -- interaction -------------------------------------------------------

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click to tune. Single click is reserved for the passband."""
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.getPlotItem().vb.mapSceneToView(event.position())
            self.tuneRequested.emit(float(point.x()))
        super().mouseDoubleClickEvent(event)


__all__ = ["BandRibbon", "FrequencyAxis", "SpectrumWidget"]
