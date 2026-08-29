"""The aircraft screen's map: where the sky above you actually is.

Nothing under this is downloaded. A tiled map means a network request per
tile, an attribution obligation and a service that can go away, in an
application whose whole claim is that it works off a dongle and a laptop with
nothing else plugged in - so the land behind the aircraft is 300 KB of
public-domain vectors compiled into the package, and the rule that survives
is the one that mattered: nothing is fetched while the app is running. See
`ui/basemap/`.

The receiver still does not know where it *is* - nothing in ADS-B tells it,
and asking a beginner for their latitude before they can see anything would
be the configuration screen this app exists to avoid. So the map frames
itself around what it has heard. The land is what makes that frame legible:
it is the difference between six dots in a rectangle and six aircraft over a
city you recognise. Water and land are two shades of the same dark rather
than a picture, and the places are named down to the size of a suburb,
because "somewhere over Puget Sound" is a worse answer than "over Bothell"
to somebody who lives there.

Everything that decides *where a thing goes* is a plain function at the top of
this file, tested without a window - the same split as `colormaps.py` and the
waterfall's ring buffer. What is left in the widget is paint calls.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pyqtgraph.functions import arrayToQPath
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from ...decode.adsb import Aircraft
from ..basemap import Basemap, load

# One nautical mile in degrees of latitude. A minute of arc, by definition,
# which is why aviation uses it and why no conversion constant is needed in
# the north-south direction.
NM_PER_DEGREE = 60.0
# How much room to leave around the outermost aircraft, as a fraction of the
# window. Without it an aircraft on the edge is half a triangle and its label
# is off the screen entirely.
MARGIN = 0.12
# The tightest the map will zoom, in nautical miles across. One aircraft on
# its own has no extent at all, and a map scaled to that would put it in the
# middle of a graticule ten metres wide.
MIN_SPAN_NM = 6.0
# How many positions of trail to keep behind each aircraft. A position
# arrives every second or two, so this is a couple of minutes of flight -
# long enough to see a turn, short enough not to become the whole picture.
TRAIL_POINTS = 90
# Round numbers, for the scale bar and the graticule. 1-2-5 is what a ruler
# does and what an eye reads without doing arithmetic.
_STEPS = (1.0, 2.0, 5.0)
# How far apart drawn points of the coastline may be, in pixels. Below about
# a pixel there is nothing to see for the work; above two or three the coast
# starts to look like a polygon.
LAND_DETAIL_PX = 1.5
# Water is the ground the whole map is painted on and land is one step up
# from it: the two are told apart by a shade, not by a contrast, because
# everything here is there to be recognised at a glance and then ignored.
# Anything competing with the aircraft for attention is a worse map.
WATER_COLOUR = "#0a121a"
LAND_COLOUR = "#18222c"
COAST_COLOUR = "#3d5468"
LAKE_EDGE_COLOUR = "#31465a"
BORDER_COLOUR = "#26323e"
CITY_COLOUR = "#46586a"
CITY_LABEL_COLOUR = "#6a8094"
# The dark outline that keeps an aircraft symbol legible against either.
SYMBOL_EDGE_COLOUR = "#080d13"
# The graticule is drawn as a wash rather than a colour, so that one number
# reads the same over water and over land.
GRATICULE_COLOUR = QColor(255, 255, 255, 13)
GRATICULE_LABEL_COLOUR = "#55677a"
# At most this many places on screen at once, and at most this many of them
# named. A city name drawn over another city name is worse than no name.
CITY_LIMIT = 40
CITY_LABELS = 16
# Where a place stops being a dot and becomes a slightly bigger dot. The list
# runs from New York to a town of five thousand, and drawing them all the
# same size says they are all the same kind of thing.
BIG_CITY_POPULATION = 100_000
# Low and high ends of the altitude colour ramp. Ground to the top of the
# airway structure: everything a receiver on a rooftop actually sees.
LOW_FT = 0.0
HIGH_FT = 40_000.0


def nice_number(value: float) -> float:
    """The nearest round 1, 2 or 5 at or below `value`."""
    if value <= 0.0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(value))
    for step in reversed(_STEPS):
        if step * power <= value:
            return step * power
    return power / 10.0


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles.

    Haversine rather than the flat approximation the projection uses: the
    projection only has to be right across one window, but a distance is a
    number somebody may check against something else.
    """
    radius = 60.0 * 180.0 / math.pi  # nautical miles per radian of arc
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    inner = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(inner)))


@dataclass(frozen=True)
class Projection:
    """Latitude and longitude onto pixels, for one particular window.

    Equirectangular about the centre of the view: longitude is squeezed by
    the cosine of the centre latitude and everything else is linear. Over the
    tens of miles a dongle on a windowsill can hear, that is accurate to
    within the width of the aircraft symbol, and it keeps the arithmetic
    invertible - a click on the map is a place, not a search.
    """

    centre_lat: float
    centre_lon: float
    # Pixels per degree of latitude. Longitude gets the same number scaled by
    # the cosine, which is what stops the map being stretched east-west.
    scale: float
    width: float
    height: float

    @property
    def squeeze(self) -> float:
        return max(math.cos(math.radians(self.centre_lat)), 1e-6)

    def to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        x = self.width / 2.0 + (lon - self.centre_lon) * self.squeeze * self.scale
        # Screen y grows downwards and latitude grows northwards.
        y = self.height / 2.0 - (lat - self.centre_lat) * self.scale
        return x, y

    def to_pixels(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The same arithmetic for whole arrays at once.

        The coastline is thousands of points and Python cannot afford to
        visit them one at a time five times a second - measured at 11.6 ms a
        frame doing exactly that, against 0.6 ms here.
        """
        x = self.width / 2.0 + (lons - self.centre_lon) * self.squeeze * self.scale
        y = self.height / 2.0 - (lats - self.centre_lat) * self.scale
        return x, y

    def to_position(self, x: float, y: float) -> tuple[float, float]:
        lat = self.centre_lat + (self.height / 2.0 - y) / self.scale
        lon = self.centre_lon + (x - self.width / 2.0) / (self.squeeze * self.scale)
        return lat, lon

    @property
    def span_nm(self) -> float:
        """How wide the window is, in nautical miles."""
        return self.width / self.scale * NM_PER_DEGREE


def fit(
    points: Sequence[tuple[float, float]],
    width: float,
    height: float,
    margin: float = MARGIN,
    min_span_nm: float = MIN_SPAN_NM,
) -> Projection | None:
    """A projection that puts every point on the window with room to spare.

    `None` when there is nothing to frame, which is the map's empty state and
    not an error: an aircraft is heard several times before it has sent the
    two halves of a position.
    """
    if not points or width <= 0.0 or height <= 0.0:
        return None
    lats = [lat for lat, _ in points]
    lons = [lon for _, lon in points]
    centre_lat = (min(lats) + max(lats)) / 2.0
    centre_lon = (min(lons) + max(lons)) / 2.0
    squeeze = max(math.cos(math.radians(centre_lat)), 1e-6)

    usable_w = max(width * (1.0 - 2.0 * margin), 1.0)
    usable_h = max(height * (1.0 - 2.0 * margin), 1.0)
    # The smallest scale that still fits both axes, then capped so a single
    # aircraft - or two flying in formation - does not zoom to absurdity.
    span_lat = max(max(lats) - min(lats), 1e-9)
    span_lon = max((max(lons) - min(lons)) * squeeze, 1e-9)
    scale = min(usable_h / span_lat, usable_w / span_lon)
    largest = width / (min_span_nm / NM_PER_DEGREE)
    return Projection(centre_lat, centre_lon, min(scale, largest), width, height)


def altitude_colour(altitude_ft: float | None, on_ground: bool = False) -> QColor:
    """Low and warm to high and cool, with the ground its own colour.

    Colour carries altitude because it is the one field that reads better as
    a picture than as a number: a stack of aircraft on the approach is
    obvious as a gradient and invisible as a column of digits.
    """
    if on_ground:
        return QColor("#8b98a5")
    if altitude_ft is None:
        return QColor("#6d7b89")
    fraction = min(1.0, max(0.0, (altitude_ft - LOW_FT) / (HIGH_FT - LOW_FT)))
    # Amber through green to the same blue the rest of the app uses for
    # things it is confident about.
    hue = 40.0 + fraction * 160.0
    return QColor.fromHsvF(hue / 360.0, 0.62, 1.0)


def scale_bar(projection: Projection) -> tuple[float, float]:
    """A round distance to draw, and how many pixels long it is."""
    quarter = projection.span_nm / 4.0
    distance = nice_number(quarter)
    pixels = distance / NM_PER_DEGREE * projection.scale
    return distance, pixels


def graticule_step(projection: Projection, lines: int = 4) -> float:
    """Spacing in degrees for the latitude and longitude lines."""
    span_deg = projection.height / projection.scale
    return max(nice_number(span_deg / lines), 0.001)


def update_trails(
    trails: dict[int, list[tuple[float, float]]],
    aircraft: Sequence[Aircraft],
    limit: int = TRAIL_POINTS,
) -> dict[int, list[tuple[float, float]]]:
    """Extend each aircraft's trail, and forget the ones that have gone.

    Pure enough to test without a window, which is the point: a trail that
    grew on every frame rather than on every move, or one that outlived the
    aircraft it belonged to, would look perfectly normal on screen for the
    first minute and wrong after ten.
    """
    seen = set()
    for plane in aircraft:
        if not plane.has_position:
            continue
        seen.add(plane.icao)
        trail = trails.setdefault(plane.icao, [])
        point = (float(plane.latitude), float(plane.longitude))
        # Only when it has actually moved: an aircraft reports its position
        # twice a second and a trail of identical points is a list that
        # grows for ever and draws nothing.
        if not trail or trail[-1] != point:
            trail.append(point)
            del trail[:-limit]
    for icao in [key for key in trails if key not in seen]:
        del trails[icao]
    return trails


def city_dot(population: int) -> float:
    """How wide to draw a place, in pixels.

    Two sizes rather than a scale: the list now runs from New York to a town
    of five thousand, and a continuous ramp over that range is either
    invisible at one end or a blob at the other. Two says only "this one is
    a city and that one is a suburb", which is all the eye needs to pick a
    landmark out of the dots.
    """
    return 4.0 if population >= BIG_CITY_POPULATION else 2.5


def format_degrees(value: float, axis: str) -> str:
    """A graticule label: `47.50 N`, not `47.500000`."""
    positive = "N" if axis == "lat" else "E"
    negative = "S" if axis == "lat" else "W"
    hand = positive if value >= 0 else negative
    return f"{abs(value):.2f}°{hand}"


def _joined(
    lines: Sequence[np.ndarray], shore: Sequence[np.ndarray | None] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Every line's longitudes and latitudes, and two ways of joining them.

    The first `connect` array traces each line whole, which is what a fill
    needs. The second traces only the steps the basemap marked as real
    shoreline, which is what the stroke needs: a ring cut out of a continent
    by the build's clipping box is part coast and part straight line along
    the box, and drawing the second would put a shoreline through Manitoba.
    `None` where the layer has no such marks and every step is real.
    """
    points = np.concatenate(lines)
    ends = np.cumsum([len(line) for line in lines]) - 1
    connect = np.ones(len(points), dtype=np.uint8)
    # A zero says "do not join this point to the next one", so the last point
    # of every line ends it. Without this the map is one continuous stroke
    # from Puget Sound to the Florida Keys.
    connect[ends] = 0
    edges = None
    if shore is not None and all(flags is not None for flags in shore):
        edges = np.concatenate(shore).astype(np.uint8)
        edges[ends] = 0
    return points[:, 0], points[:, 1], connect, edges


def polyline_path(projection: Projection, lines: Sequence[np.ndarray]) -> QPainterPath:
    """One painter path holding every line, projected into pixels.

    `lines` are (n, 2) arrays of longitude and latitude. `arrayToQPath`
    builds the path in C++ straight from the arrays: visiting a coastline
    point at a time in Python measured 11.6 ms a frame against 0.6 ms here.
    """
    lons, lats, connect, _ = _joined(lines)
    x, y = projection.to_pixels(lats, lons)
    return arrayToQPath(x, y, connect=connect)


# How coarse a version of each layer to draw, as the distance between points
# in degrees. Four levels cover every frame this map reaches, from a single
# aircraft six miles across to a whole coastline.
DETAIL_LEVELS = (0.0, 0.002, 0.008, 0.03)
# How far the window is rounded outwards before it is used as a cache key.
# The frame moves a little whenever an aircraft does, but which lines are on
# screen almost never changes, so rounding turns a per-frame search of every
# coastline into a dictionary lookup. Rounding *outwards* is what keeps it
# correct: the answer covers more than the window, never less.
WINDOW_BUCKET_DEG = 0.05
# Enough for a screen's worth of layers at a few zoom levels. Beyond that
# something is wrong and holding more of it will not help.
_CACHE_LIMIT = 48
_LAND_CACHE: dict[
    tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
] = {}


def detail_level(step_deg: float) -> int:
    """The coarsest stored detail that is still finer than `step_deg`."""
    level = 0
    for index, step in enumerate(DETAIL_LEVELS):
        if step <= step_deg:
            level = index
    return level


def land_arrays(
    basemap: Basemap, name: str, window: tuple[float, float, float, float], level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """The visible part of a layer, flattened once and kept.

    Returns longitudes, latitudes, the connect flags that trace every line
    whole, and the ones that trace only real shoreline. Held rather than
    recomputed because the expensive half is the search and the
    concatenation, neither of which the frame moving actually changes: with
    this the land costs the projection and nothing else.
    """
    west, south, east, north = window
    key = (
        name,
        level,
        math.floor(west / WINDOW_BUCKET_DEG),
        math.floor(south / WINDOW_BUCKET_DEG),
        math.ceil(east / WINDOW_BUCKET_DEG),
        math.ceil(north / WINDOW_BUCKET_DEG),
    )
    found = _LAND_CACHE.get(key)
    if found is not None:
        return found
    layer = basemap.layer(name)
    if layer is None:
        return None
    rounded = (
        key[2] * WINDOW_BUCKET_DEG,
        key[3] * WINDOW_BUCKET_DEG,
        key[4] * WINDOW_BUCKET_DEG,
        key[5] * WINDOW_BUCKET_DEG,
    )
    if layer.closed:
        rings = layer.visible_rings(*rounded, step_deg=DETAIL_LEVELS[level])
        lines = [ring for ring, _ in rings]
        shore: list[np.ndarray | None] | None = [flags for _, flags in rings]
    else:
        lines = layer.visible(*rounded, step_deg=DETAIL_LEVELS[level])
        shore = None
    if not lines:
        return None
    if len(_LAND_CACHE) >= _CACHE_LIMIT:
        _LAND_CACHE.clear()
    _LAND_CACHE[key] = _joined(lines, shore)
    return _LAND_CACHE[key]


class PlaneMap(QWidget):
    """A plan view of the aircraft that have reported where they are."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setAutoFillBackground(False)
        self._aircraft: tuple[Aircraft, ...] = ()
        # Where each aircraft has been, oldest first, keyed by ICAO address.
        # Held here rather than in the decoder because it is a display
        # memory: what the map has drawn, not what the receiver knows.
        self._trails: dict[int, list[tuple[float, float]]] = {}
        self._projection: Projection | None = None
        # Loaded once and shared: the loader is cached, and a missing data
        # file gives an empty basemap rather than an error, so the map is
        # drawn either way.
        self.basemap: Basemap = load()

    # -- data --------------------------------------------------------------

    def show_aircraft(self, aircraft: Sequence[Aircraft]) -> None:
        """Take a new sky. Cheap enough to call at the screen's refresh rate."""
        self._aircraft = tuple(aircraft)
        update_trails(self._trails, self._aircraft)
        self.update()

    def clear(self) -> None:
        self._aircraft = ()
        self._trails.clear()
        self._projection = None
        self.update()

    @property
    def plotted(self) -> int:
        """How many of the aircraft heard have said where they are."""
        return sum(1 for plane in self._aircraft if plane.has_position)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(WATER_COLOUR))

        points = [
            (float(plane.latitude), float(plane.longitude))
            for plane in self._aircraft
            if plane.has_position
        ]
        projection = fit(points, float(self.width()), float(self.height()))
        self._projection = projection
        if projection is None:
            self._paint_empty(painter)
            painter.end()
            return

        self._paint_land(painter, projection)
        self._paint_graticule(painter, projection)
        self._paint_cities(painter, projection)
        for plane in self._aircraft:
            if plane.has_position:
                self._paint_trail(painter, projection, plane)
        for plane in self._aircraft:
            if plane.has_position:
                self._paint_plane(painter, projection, plane)
        self._paint_scale(painter, projection)
        painter.end()

    def _paint_empty(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#6d7b89")))
        font = QFont(self.font())
        font.setPointSizeF(max(9.0, font.pointSizeF()))
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(24, 24, -24, -24),
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
            "No positions yet. An aircraft has to send two halves of a "
            "position before it can be placed, which takes a few seconds "
            "after it is first heard.",
        )

    def _window(self, projection: Projection) -> tuple[float, float, float, float]:
        """West, south, east, north of what is on screen."""
        north, west = projection.to_position(0.0, 0.0)
        south, east = projection.to_position(projection.width, projection.height)
        return west, south, east, north

    def _paint_land(self, painter: QPainter, projection: Projection) -> None:
        if self.basemap.empty:
            return
        # A pixel and a half, in degrees. Thinning by distance rather than by
        # a zoom level means one number covers every scale, and the whole
        # coastline of the United States costs a few thousand points at the
        # widest view instead of forty-five thousand.
        step_deg = LAND_DETAIL_PX / max(projection.scale, 1e-9)
        level = detail_level(step_deg)
        window = self._window(projection)
        # Land, then the lakes painted back out of it in the water colour, so
        # the Great Lakes are water rather than holes; then the shoreline
        # along the edge of each, then the state borders on top of both.
        for name, fill, stroke, width in (
            ("land", LAND_COLOUR, COAST_COLOUR, 1.1),
            ("lakes", WATER_COLOUR, LAKE_EDGE_COLOUR, 0.9),
            ("states", None, BORDER_COLOUR, 1.0),
        ):
            arrays = land_arrays(self.basemap, name, window, level)
            if arrays is None:
                continue
            lons, lats, connect, shore = arrays
            x, y = projection.to_pixels(lats, lons)
            if fill is not None:
                whole = arrayToQPath(x, y, connect=connect)
                # Odd-even, so a ring inside another ring is a hole: an
                # island in a lake is land again, and the arithmetic says so
                # without anybody having to sort rings by containment.
                whole.setFillRule(Qt.FillRule.OddEvenFill)
                # Without antialiasing, because the land arrives as a grid of
                # tiles and two antialiased fills that abut leave a hairline
                # of what is behind them showing along every join. A hard
                # edge has no seam, and the shoreline stroke below is drawn
                # antialiased directly over it wherever it is visible at all.
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                painter.fillPath(whole, QColor(fill))
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(stroke), width))
            painter.drawPath(
                arrayToQPath(x, y, connect=connect if shore is None else shore)
            )

    def _paint_cities(self, painter: QPainter, projection: Projection) -> None:
        cities = self.basemap.visible_cities(
            *self._window(projection), limit=CITY_LIMIT
        )
        if not cities:
            return
        font = QFont(self.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        # Largest first, so where two names collide the one that survives is
        # the one more likely to mean something to somebody.
        drawn: list[QRectF] = []
        named = 0
        for city in cities:
            x, y = projection.to_pixel(city.latitude, city.longitude)
            size = city_dot(city.population)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(CITY_COLOUR))
            painter.drawRect(QRectF(x - size / 2.0, y - size / 2.0, size, size))
            if named >= CITY_LABELS:
                continue
            width = metrics.horizontalAdvance(city.name)
            box = QRectF(x + 5.0, y - 8.0, width, 11.0)
            if any(box.intersects(other) for other in drawn):
                continue
            drawn.append(box)
            named += 1
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(CITY_LABEL_COLOUR)))
            painter.drawText(QPointF(x + 5.0, y + 3.0), city.name)

    def _paint_graticule(self, painter: QPainter, projection: Projection) -> None:
        step = graticule_step(projection)
        label_pen = QPen(QColor(GRATICULE_LABEL_COLOUR))
        line_pen = QPen(GRATICULE_COLOUR, 1)
        font = QFont(self.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)

        top_lat, left_lon = projection.to_position(0.0, 0.0)
        bottom_lat, _ = projection.to_position(0.0, projection.height)
        _, right_lon = projection.to_position(projection.width, 0.0)

        line = math.ceil(bottom_lat / step) * step
        while line <= top_lat:
            _, y = projection.to_pixel(line, projection.centre_lon)
            painter.setPen(line_pen)
            painter.drawLine(QPointF(0.0, y), QPointF(projection.width, y))
            painter.setPen(label_pen)
            painter.drawText(QPointF(6.0, y - 4.0), format_degrees(line, "lat"))
            line += step

        line = math.ceil(left_lon / step) * step
        # Longitude lines are closer together in pixels than latitude ones by
        # the cosine, so a spacing that reads comfortably down the side can
        # be a solid row of overlapping text along the bottom. The lines all
        # get drawn; only the labels are thinned.
        labelled_to = -1e9
        while line <= right_lon:
            x, _ = projection.to_pixel(projection.centre_lat, line)
            painter.setPen(line_pen)
            painter.drawLine(QPointF(x, 0.0), QPointF(x, projection.height))
            text = format_degrees(line, "lon")
            width = QFontMetricsF(font).horizontalAdvance(text) + 12.0
            if x >= labelled_to + width:
                painter.setPen(label_pen)
                painter.drawText(QPointF(x + 4.0, projection.height - 6.0), text)
                labelled_to = x
            line += step

    def _paint_trail(
        self, painter: QPainter, projection: Projection, plane: Aircraft
    ) -> None:
        trail = self._trails.get(plane.icao, [])
        if len(trail) < 2:
            return
        faded = QColor(altitude_colour(plane.altitude_ft, plane.on_ground))
        faded.setAlpha(90)
        painter.setPen(QPen(faded, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Trails are stored latitude first and the basemap longitude first,
        # because each follows the source it came from. Flipped here, once.
        track = np.asarray(trail, dtype=np.float64)[:, ::-1]
        painter.drawPath(polyline_path(projection, [track]))

    def _paint_plane(
        self, painter: QPainter, projection: Projection, plane: Aircraft
    ) -> None:
        x, y = projection.to_pixel(float(plane.latitude), float(plane.longitude))
        colour = altitude_colour(plane.altitude_ft, plane.on_ground)

        painter.save()
        painter.translate(x, y)
        # An aircraft that has not said which way it is going is drawn as a
        # dot rather than as an arrow pointing north, which would be a
        # heading the receiver never heard.
        if plane.track_deg is None:
            painter.setBrush(colour)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(-3.5, -3.5, 7.0, 7.0))
        else:
            painter.rotate(float(plane.track_deg))
            arrow = QPolygonF(
                [
                    QPointF(0.0, -8.0),
                    QPointF(5.5, 7.0),
                    QPointF(0.0, 4.0),
                    QPointF(-5.5, 7.0),
                ]
            )
            painter.setBrush(colour)
            painter.setPen(QPen(QColor(SYMBOL_EDGE_COLOUR), 1))
            painter.drawPolygon(arrow)
        painter.restore()

        font = QFont(self.font())
        font.setPointSizeF(8.5)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        detail = ""
        if plane.on_ground:
            detail = "on the ground"
        elif plane.altitude_ft is not None:
            detail = f"{plane.altitude_ft:,} ft"
        # An aircraft near the right-hand edge gets its label on the other
        # side of it. The margin `fit` leaves is for the symbol, not for
        # however long a callsign happens to be.
        widest = max(
            metrics.horizontalAdvance(plane.label),
            metrics.horizontalAdvance(detail),
        )
        left = x + 9.0
        if left + widest > projection.width - 4.0:
            left = x - 9.0 - widest
        painter.setPen(QPen(QColor("#cbd5e0")))
        painter.drawText(QPointF(left, y + 1.0), plane.label)
        if detail:
            painter.setPen(QPen(QColor("#8b98a5")))
            painter.drawText(QPointF(left, y + 12.0), detail)

    def _paint_scale(self, painter: QPainter, projection: Projection) -> None:
        distance, pixels = scale_bar(projection)
        if pixels < 8.0:
            return
        # Top right: the latitude labels run down the left and the longitude
        # labels along the bottom, and this is the corner nothing else wants.
        y = 26.0
        left = projection.width - pixels - 16.0
        painter.setPen(QPen(QColor("#8b98a5"), 1.5))
        painter.drawLine(QPointF(left, y), QPointF(left + pixels, y))
        painter.drawLine(QPointF(left, y - 4.0), QPointF(left, y + 4.0))
        painter.drawLine(
            QPointF(left + pixels, y - 4.0), QPointF(left + pixels, y + 4.0)
        )
        font = QFont(self.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)
        miles = (
            "1 nautical mile" if distance == 1.0 else f"{distance:g} nautical miles"
        )
        width = QFontMetricsF(font).horizontalAdvance(miles)
        painter.drawText(QPointF(left + pixels - width, y - 8.0), miles)


__all__ = [
    "MIN_SPAN_NM",
    "TRAIL_POINTS",
    "NM_PER_DEGREE",
    "PlaneMap",
    "Projection",
    "altitude_colour",
    "city_dot",
    "distance_nm",
    "fit",
    "format_degrees",
    "graticule_step",
    "detail_level",
    "land_arrays",
    "nice_number",
    "polyline_path",
    "scale_bar",
    "update_trails",
]
