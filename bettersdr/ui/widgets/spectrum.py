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

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF

from ...dsp.psd import PeakHold, noise_floor_db
from ...scan import bandplan
from ..levels import Level
from .axes import AXIS_WIDTH, BlankAxis, FrequencyAxis
from .viewspan import EDGE_GRAB_PX, PanZoom

BACKGROUND = "#0b0e13"
GRID_ALPHA = 0.18
TRACE_COLOUR = "#5ad1ff"
PEAK_COLOUR = "#f0a24a"
PASSBAND_COLOUR = QColor(90, 209, 255, 40)


# A channel has to be this much of the visible span before the grid of
# channels is drawn at all. A marine window at 2.4 MS/s holds 96 of them; 96
# hairlines teach nobody anything, so the grid waits until the user has zoomed
# in far enough for it to be a ruler rather than a texture.
CELL_FRACTION = 0.012
# Channel names are drawn a size down. The lane is half the ribbon, the names
# are repetitive, and the eye is meant to land on the one under the passband.
CHANNEL_POINT_SIZE = 8
# Clear space either side of a name, in pixels, before it counts as touching
# its neighbour.
LABEL_GAP_PX = 10
# How solid the backing behind the tuned name is. Not opaque, because the
# block underneath is how the eye finds it in the first place.
LABEL_BACKING_ALPHA = 200

RIBBON_HEIGHT = 34
# The channel lane is the bottom half, and it only exists when there is
# something to put in it - a band with no channels keeps the full-height
# stripe the ribbon has always drawn.
CHANNEL_LANE = 0.5

ALLOCATION_COLOUR = QColor("#4a5058")

# Which name survives a collision. The one the receiver is actually on always
# does: a ribbon that drops the name of the band you are listening to because
# a neighbour was fractionally wider has got the priority exactly backwards.
RANK_TUNED = 0
RANK_BAND = 1
RANK_ALLOCATION = 2
RANK_CHANNEL = 3


class ChannelCell(NamedTuple):
    """One named channel, as a block to draw on the ribbon."""

    start_hz: float
    end_hz: float
    name: str
    tuned: bool


class Label(NamedTuple):
    """A name asking to be drawn at `hz`, `half_hz` of text wide either side."""

    hz: float
    half_hz: float
    text: str
    rank: int


def channel_cells(
    bands: Sequence[bandplan.Band],
    low_hz: float,
    high_hz: float,
    tuned_hz: float,
) -> tuple[ChannelCell, ...]:
    """The channel blocks worth drawing across a window.

    Pure, so the rules can be read in a test rather than in a screenshot.

    One threshold and one exception. The grid appears once a channel is a
    reasonable fraction of the window, and **the channel the receiver is
    actually on is drawn whatever its width** - because at 2.4 MS/s no marine
    channel comes close to fitting, and that is exactly the moment somebody
    most needs telling that they are listening to Channel 16.
    """
    span = high_hz - low_hz
    if span <= 0:
        return ()
    cells: list[ChannelCell] = []
    for band in bands:
        if not band.channels:
            continue
        # The same width `Band.channel` claims either side of a frequency, so
        # the block on screen is exactly the ground the name covers.
        width = float(band.raster_hz or band.bandwidth_hz)
        gridded = width >= span * CELL_FRACTION
        here = band.channel(tuned_hz) if band.contains(tuned_hz) else None
        for channel in band.channels:
            start = channel.frequency_hz - width / 2.0
            end = channel.frequency_hz + width / 2.0
            if end < low_hz or start > high_hz:
                continue
            tuned = channel is here
            if not (gridded or tuned):
                continue
            cells.append(ChannelCell(start, end, channel.name, tuned))
    return tuple(cells)


def without_collisions(labels: Sequence[Label]) -> tuple[Label, ...]:
    """The names that can be drawn without touching each other.

    Also pure, and the reason it exists is a real overlap: at 162.550 MHz the
    weather channels and the federal allocation next to them both wanted a
    name, 260 kHz apart in a 2.4 MHz window, and the ribbon read "Federal
    governmentWeather Radio". A fraction-of-the-window rule cannot see that,
    because it never compares two labels with each other - and the one the
    fraction rule would have dropped is the band being listened to.

    Highest rank first, then left to right, so the answer does not depend on
    the order the caller happened to build the list in.
    """
    kept: list[Label] = []
    for label in sorted(labels, key=lambda item: (item.rank, item.hz)):
        if any(
            abs(label.hz - other.hz) < label.half_hz + other.half_hz
            for other in kept
        ):
            continue
        kept.append(label)
    return tuple(sorted(kept, key=lambda item: item.hz))


class BandRibbon(pg.PlotWidget):
    """Coloured allocation blocks aligned to the spectrum's frequency axis.

    Two lanes. The top one is the band plan - the stripe that says "2 m
    Amateur" - and from Standard upwards it also names the stretches no band
    covers, so the ribbon stops going blank over half the dial. The bottom one
    appears when the band under the cursor has named channels, and says which
    of them the receiver is sitting on.
    """

    def __init__(
        self,
        region: str = bandplan.DEFAULT_REGION,
        level: Level = Level.STANDARD,
        parent=None,
    ) -> None:
        super().__init__(parent=parent, axisItems={"left": BlankAxis()})
        self.region = region
        self.level = level
        self._span: tuple[float, float, float] | None = None
        self._drawn: tuple | None = None
        self._items: list[pg.GraphicsObject] = []

        self.setBackground(BACKGROUND)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setFixedHeight(RIBBON_HEIGHT)
        plot = self.getPlotItem()
        plot.showAxis("bottom", False)
        plot.setContentsMargins(0, 0, 0, 0)
        plot.getViewBox().setDefaultPadding(0.0)
        self.setYRange(0.0, 1.0, padding=0.0)

    def set_level(self, level: Level) -> None:
        self.level = level
        self._redraw()

    def set_span(self, low_hz: float, high_hz: float, tuned_hz: float) -> None:
        """The stretch of dial on screen, and where inside it the radio is.

        The two used to be one number: the display always showed the whole
        window, so the middle of the span was the tuned frequency by
        construction. Panning breaks that - the middle of a panned view is
        wherever the user dragged to - and the ribbon would then highlight
        whichever channel happened to be in the centre of the screen rather
        than the one being listened to, which is the one piece of information
        this lane exists to carry.
        """
        self._span = (low_hz, high_hz, tuned_hz)
        self._redraw()

    # -- drawing -----------------------------------------------------------

    def _redraw(self) -> None:
        if self._span is None:
            return
        low_hz, high_hz, tuned_hz = self._span
        width_px = self._view_width_px()
        # The width is part of the key because what fits depends on it: the
        # same span in a narrower window holds fewer names, and a resize that
        # did not redraw would leave them overlapping.
        key = (low_hz, high_hz, tuned_hz, int(self.level), int(width_px))
        if key == self._drawn:
            return
        self._drawn = key

        plot = self.getPlotItem()
        for item in self._items:
            plot.removeItem(item)
        self._items.clear()

        span = high_hz - low_hz
        hz_per_px = span / width_px if span > 0 else 1.0

        bands = bandplan.overlapping(low_hz, high_hz, self.region)
        cells = channel_cells(bands, low_hz, high_hz, tuned_hz)
        lane = (CHANNEL_LANE, 1.0) if cells else (0.0, 1.0)
        text_y = (CHANNEL_LANE + 1.0) / 2.0 if cells else 0.5

        # Names are collected first and drawn last, because whether one can be
        # drawn depends on the others. The two lanes are resolved separately -
        # they are different rows on the screen, so a channel name and a band
        # name at the same frequency are not a collision.
        upper: list[Label] = []
        lower: list[Label] = []
        styles: dict[int, tuple[str, int, bool]] = {}

        def claim(into, text, left, right, rank, colour, point_size=0):
            """Ask for `text` centred in `left`..`right`, if there is room.

            Measured in the font it will be drawn in rather than guessed as a
            fraction of the window: "Federal government" and "2 m" have no
            fraction in common, and the fraction rule that predated this both
            hid short names that fitted and drew long ones across each other.

            A name wider than its own block is normally a name for its
            neighbour, so it is dropped - except for the one being listened
            to, which is the name the user is actually looking for. Weather
            Radio is 150 kHz of a 2.4 MHz window and half as wide again as
            its stripe. That one gets a backing, because a name overhanging
            its block has the block's own edges drawn through its letters.
            """
            width_hz = self._text_width_px(text, point_size) * hz_per_px
            overhangs = right - left < width_hz
            if overhangs and rank != RANK_TUNED:
                return
            label = Label(
                hz=(left + right) / 2.0,
                half_hz=(width_hz + LABEL_GAP_PX * hz_per_px) / 2.0,
                text=text,
                rank=rank,
            )
            styles[id(label)] = (colour, point_size, overhangs)
            into.append(label)

        for band in bands:
            colour = QColor(band.colour)
            left = max(band.start_hz, low_hz)
            right = min(band.end_hz, high_hz)
            self._block(
                left,
                right,
                brush=QColor(colour.red(), colour.green(), colour.blue(), 110),
                pen=pg.mkPen(colour, width=1),
                span=lane,
            )
            claim(
                upper,
                band.name,
                left,
                right,
                RANK_TUNED if band.contains(tuned_hz) else RANK_BAND,
                "#e6edf3",
            )

        if self.level >= Level.STANDARD:
            # Dashed and grey on purpose: this is not a band, it is a note
            # about who owns the silence, and it must not look like somewhere
            # the app is offering to take you.
            for allocation in bandplan.overlapping_allocations(
                low_hz, high_hz, self.region
            ):
                left = max(allocation.start_hz, low_hz)
                right = min(allocation.end_hz, high_hz)
                self._block(
                    left,
                    right,
                    brush=QColor(74, 80, 88, 70),
                    pen=pg.mkPen(
                        ALLOCATION_COLOUR, width=1, style=Qt.PenStyle.DashLine
                    ),
                    span=lane,
                )
                claim(
                    upper,
                    allocation.name,
                    left,
                    right,
                    RANK_TUNED if allocation.contains(tuned_hz) else RANK_ALLOCATION,
                    "#8b98a5",
                )

        for cell in cells:
            self._block(
                max(cell.start_hz, low_hz),
                min(cell.end_hz, high_hz),
                brush=QColor(90, 209, 255, 55 if cell.tuned else 12),
                pen=pg.mkPen("#5ad1ff" if cell.tuned else "#2b323b", width=1),
                span=(0.0, CHANNEL_LANE),
            )
            # Measured against the channel itself rather than its visible part:
            # a channel half off the edge of the window is half off the edge of
            # the radio's window too, and sliding its name inward would point
            # at a frequency that is not the one it names. The tuned one skips
            # the fitting test entirely - at 2.4 MS/s no marine channel is as
            # wide as its own name, and that is the moment somebody most needs
            # telling that they are listening to Channel 16.
            claim(
                lower,
                cell.name,
                cell.start_hz,
                cell.end_hz,
                RANK_TUNED if cell.tuned else RANK_CHANNEL,
                "#5ad1ff" if cell.tuned else "#b6c2cf",
                point_size=CHANNEL_POINT_SIZE,
            )

        for label in without_collisions(upper):
            colour, point_size, backed = styles[id(label)]
            self._label(label.text, label.hz, text_y, colour, point_size, backed)
        for label in without_collisions(lower):
            colour, point_size, backed = styles[id(label)]
            self._label(
                label.text, label.hz, CHANNEL_LANE / 2.0, colour, point_size, backed
            )

        self.setXRange(low_hz, high_hz, padding=0.0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # `GraphicsView.setCentralItem` resizes during `super().__init__`, so
        # this runs once before the ribbon has any attributes of its own -
        # and `PlotWidget.__getattr__` forwards the miss to the plot item,
        # which raises rather than returning None.
        if getattr(self, "_span", None) is None:
            return
        # Narrowing the widget changes which names fit, and the cache key
        # carries the width, so this is a real redraw rather than a repeat.
        self._redraw()

    def _view_width_px(self) -> float:
        """How many pixels the frequency axis is drawn across.

        The view box is the honest answer once the widget has been laid out;
        the widget's own width less the axis gutter is the fallback for the
        first draw, which happens before Qt has shown anything.
        """
        width = float(self.getPlotItem().getViewBox().width())
        if width > 1.0:
            return width
        return max(float(self.width() - AXIS_WIDTH), 1.0)

    def _font(self, point_size: int) -> QFont:
        font = QFont(self.font())
        if point_size:
            font.setPointSize(point_size)
        return font

    def _text_width_px(self, text: str, point_size: int = 0) -> float:
        return float(QFontMetricsF(self._font(point_size)).horizontalAdvance(text))

    def _block(self, start_hz, end_hz, brush, pen, span) -> None:
        block = pg.LinearRegionItem(
            values=(start_hz, end_hz),
            brush=brush,
            pen=pen,
            movable=False,
            span=span,
        )
        block.setZValue(-10)
        self.getPlotItem().addItem(block)
        self._items.append(block)

    def _label(
        self,
        text: str,
        hz: float,
        y: float,
        colour: str,
        point_size: int = 0,
        backed: bool = False,
    ) -> None:
        fill = None
        if backed:
            ground = QColor(BACKGROUND)
            ground.setAlpha(LABEL_BACKING_ALPHA)
            fill = pg.mkBrush(ground)
        label = pg.TextItem(text, color=colour, anchor=(0.5, 0.5), fill=fill)
        label.setFont(self._font(point_size))
        label.setPos(hz, y)
        self.getPlotItem().addItem(label)
        self._items.append(label)


class SpectrumWidget(PanZoom, pg.PlotWidget):
    """Live spectrum with peak hold, a passband, and zoom and pan across it.

    Click anywhere to tune, drag to pan, wheel to zoom - the same three
    gestures as the waterfall below, which is why they live in `PanZoom`
    rather than in either pane. The passband's *edges* still drag to set the
    bandwidth; its body no longer moves, because click-to-tune says where to
    listen far more directly than dragging a block does, and a movable body
    would swallow every press inside a passband that at 8x zoom fills the
    whole pane.
    """

    tuneRequested = Signal(float)
    bandwidthChanged = Signal(float)
    viewChanged = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent, axisItems={"bottom": FrequencyAxis("bottom")})
        self._init_pan_zoom()
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
        # The edges keep their own hover highlight - they are still handles -
        # but the body must not light up as though it were one, now that a
        # press inside it is a click on the spectrum behind it.
        self._passband.setAcceptHoverEvents(False)
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

        # The whole window is always drawn; what is on screen is a view of it,
        # which is what makes zooming free of the radio.
        self.set_window(center_hz, sample_rate)

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
        """An edge was dragged, so the bandwidth changed - and only that.

        The drag used to be able to retune as well, because the whole block
        was movable and moving it moved its centre. Clicking says where to
        listen now, so an edge is purely a bandwidth handle: it is measured
        from the frequency being listened to rather than from the other edge,
        and the passband stays centred where the radio is. A handle that also
        retuned would drag the passband out from under the pointer halfway
        through setting a filter width.
        """
        if self._updating_passband:
            return
        low, high = self._passband.getRegion()
        half = self._bandwidth_hz / 2.0
        # Whichever edge moved furthest from where it was put is the one under
        # the pointer; the other one has not moved at all.
        edges = (abs(low - self._tuned_hz), abs(high - self._tuned_hz))
        moved = max(edges, key=lambda reach: abs(reach - half))
        width = 2.0 * moved
        if abs(width - self._bandwidth_hz) > 1.0:
            self.bandwidthChanged.emit(float(width))

    # -- interaction -------------------------------------------------------

    def _grabs_an_item(self, hz: float) -> bool:
        """Whether a press is on a passband edge rather than on the spectrum.

        In pixels, not hertz: the passband is 200 kHz wide on a broadcast
        station and 500 Hz on a CW signal, and a handle has to be the same
        size under the pointer either way. A press anywhere else is the
        pane's own, so click-to-tune and drag-to-pan work over the passband
        as well as beside it.
        """
        reach = EDGE_GRAB_PX * self._hz_per_px()
        low, high = self._passband.getRegion()
        return abs(hz - low) <= reach or abs(hz - high) <= reach


__all__ = ["BandRibbon", "FrequencyAxis", "SpectrumWidget"]
