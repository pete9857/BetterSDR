"""The aircraft map's background: coastlines, lakes, state lines, cities.

Compiled from Natural Earth by `tools/build_basemap.py` and committed as
`us.bsm`, about 230 KB. Nothing here downloads anything, and nothing here
parses a shapefile - the same bargain as `drivers/win-x64/`, where the cost
of carrying the data is paid once so that the running application never has
to depend on anything it cannot see.

The data answers the one question the map could not answer on its own.
An ADS-B receiver does not know where it is, so before this the display had
no fixed frame of reference at all: it framed itself on the aircraft and left
the user to recognise the arrangement. A coastline and a state border say
where the sky being drawn actually is, without anybody having to type in
their latitude.

**Format.** Eight magic bytes, the uncompressed length, then a zlib stream
holding:

    grid        float64     degrees per stored unit (1e-4, about 11 m)
    layers      uint32
    per layer:
        name    16 bytes    NUL-padded ASCII
        flags   uint32      bit 0: closed rings, to be filled
        lines   uint32
        counts  uint32 * lines          points in each line
        heads   int32 * 2 * lines       first point of each line, in units
        deltas  int16 * 2 * (points - lines)   steps along each line
        if bit 0:
            bytes   uint32              length of what follows
            shore   one bit per point, packed, most significant first
    cities      uint32
        lon     int32 * cities
        lat     int32 * cities
        pop     uint32 * cities
        names   uint32 length, then length-prefixed UTF-8 names

Steps rather than positions, because a coastline moves a few units at a time
and the high bytes of an int16 delta are almost all zero, which is what makes
the whole thing compress to a third of its size. Steps also mean the decoder
is a `cumsum` per line and nothing else - measured below a fiftieth of a
second for the lot, which is why the map can simply load it at startup.

**The shoreline bits are what let land be filled and still be drawn.** The
land layer is closed rings, because a fill needs them, and a ring cut out of
a continent by the build's clipping box is part real shoreline and part
straight lines along the box. One bit per point says which, and it is exactly
the `connect` array Qt is given: the same geometry is filled whole and
stroked in pieces, which is also the only way the edge of the fill and the
coastline are guaranteed to be in the same place.

A missing or unreadable file gives an empty basemap rather than an error.
The map is perfectly usable without it - it was, yesterday - and a beginner
whose install is missing a data file should get a plainer map, not a crash.
"""

from __future__ import annotations

import functools
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASEMAP_DIR = Path(__file__).resolve().parent
DEFAULT_REGION = "us"
MAGIC = b"BSDRMAP\x02"
# The longest run of points the map will ever skip when zoomed out. Beyond
# this a coastline stops being a coastline and starts being a polygon.
MAX_STRIDE = 32
# Bit 0 of a layer's flags word: closed rings, to be filled, carrying one
# shoreline bit per point.
FLAG_POLYGON = 1


@dataclass(frozen=True)
class Layer:
    """One set of lines - the land, the lakes, the state borders.

    `bounds` and `spacing` are computed once at load because both are asked
    for on every frame: the first decides whether a line is on screen at all,
    and the second decides how much of it is worth drawing.

    A `closed` layer is rings, to be filled. Its `shore` flags say which of
    each ring's steps are real coastline rather than the straight edges the
    build's clipping box cut through a continent - so one ring is filled
    whole and stroked in pieces, and the edge of the fill can never disagree
    with the shoreline drawn along it.
    """

    name: str
    lines: tuple[np.ndarray, ...]
    # west, south, east, north for each line.
    bounds: np.ndarray
    # Mean distance between consecutive points, in degrees, for each line.
    spacing: np.ndarray
    closed: bool = False
    # One flag per point, saying whether the step to the next point is real
    # shoreline. Empty for an open layer, where every step is.
    shore: tuple[np.ndarray, ...] = ()

    @property
    def points(self) -> int:
        return sum(len(line) for line in self.lines)

    def _kept(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        step_deg: float,
    ) -> list[tuple[int, np.ndarray | None]]:
        """Which lines cross the window, and which of their points to draw.

        `None` for the points means all of them. Shared by `visible` and
        `visible_rings`, so a ring and its shoreline flags cannot be thinned
        differently - which would slide the flags along the ring and stroke
        the clipping box instead of the coast.
        """
        if not self.lines:
            return []
        hit = (
            (self.bounds[:, 0] <= east)
            & (self.bounds[:, 2] >= west)
            & (self.bounds[:, 1] <= north)
            & (self.bounds[:, 3] >= south)
        )
        out: list[tuple[int, np.ndarray | None]] = []
        for index in np.flatnonzero(hit):
            line = self.lines[index]
            stride = 1
            if step_deg > 0.0 and self.spacing[index] > 0.0:
                stride = int(min(MAX_STRIDE, max(1, step_deg // self.spacing[index])))
            if stride == 1 or len(line) <= 2:
                out.append((int(index), None))
                continue
            keep = np.arange(0, len(line), stride)
            # The last point always survives: a coastline that stops a stride
            # short of its end leaves a gap wherever two lines meet, and a
            # ring that does it is no longer closed.
            if keep[-1] != len(line) - 1:
                keep = np.append(keep, len(line) - 1)
            out.append((int(index), keep))
        return out

    def visible(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        step_deg: float = 0.0,
    ) -> list[np.ndarray]:
        """The lines that cross this window, thinned to suit its scale.

        `step_deg` is how far apart drawn points may be - a pixel or two
        converted into degrees by the caller. Passing zero draws every point,
        which is right when zoomed in and wasteful when zoomed out: the whole
        coastline is 45,000 points and a 300 mile view has no use for more
        than a few thousand of them.
        """
        return [
            self.lines[index] if keep is None else self.lines[index][keep]
            for index, keep in self._kept(west, south, east, north, step_deg)
        ]

    def visible_rings(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        step_deg: float = 0.0,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """The same lines, each with its shoreline flags beside it.

        The flags are thinned by the same indices as the points they
        describe, so a run of coast that a wide view collapses into one step
        keeps the flag of the step it began with - the right answer at a
        scale where the whole run is under a pixel. `None` where the layer
        has no flags at all, which means every step is real.
        """
        out: list[tuple[np.ndarray, np.ndarray | None]] = []
        for index, keep in self._kept(west, south, east, north, step_deg):
            line = self.lines[index]
            flags = self.shore[index] if self.shore else None
            if keep is None:
                out.append((line, flags))
            else:
                out.append((line[keep], None if flags is None else flags[keep]))
        return out


@dataclass(frozen=True)
class City:
    """One place, with the population that decides whether it is drawn."""

    name: str
    longitude: float
    latitude: float
    population: int


@dataclass(frozen=True)
class Basemap:
    """Everything drawn behind the aircraft."""

    layers: dict[str, Layer]
    # Largest first, which is the order they should be drawn and dropped in.
    cities: tuple[City, ...] = ()

    def __post_init__(self) -> None:
        # The same places again as two columns, because the list is five
        # thousand long now that it reaches the size of a suburb, and a
        # window query five times a second should not be a walk down five
        # thousand dataclasses. Built here so it cannot disagree with
        # `cities`, whatever route the basemap was constructed by.
        columns = np.asarray(
            [(city.longitude, city.latitude) for city in self.cities], dtype=np.float64
        ).reshape(-1, 2)
        object.__setattr__(self, "_columns", columns)

    @property
    def empty(self) -> bool:
        return not self.layers and not self.cities

    def layer(self, name: str) -> Layer | None:
        return self.layers.get(name)

    def visible_cities(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        limit: int = 40,
    ) -> list[City]:
        """The largest places in this window, biggest first.

        Sorted by population at build time, so this takes the first `limit`
        of a masked comparison rather than sorting per frame - and taking the
        largest is also the right answer for a crowded window, where the
        alternative is a city name drawn on top of another city name.
        """
        if not self.cities:
            return []
        columns: np.ndarray = self._columns  # type: ignore[attr-defined]
        lons, lats = columns[:, 0], columns[:, 1]
        hit = (lons >= west) & (lons <= east) & (lats >= south) & (lats <= north)
        return [self.cities[index] for index in np.flatnonzero(hit)[:limit]]


def _decode(payload: bytes) -> Basemap:
    grid, layer_count = struct.unpack_from("<dI", payload, 0)
    at = 12
    layers: dict[str, Layer] = {}
    for _ in range(layer_count):
        name = payload[at : at + 16].split(b"\0")[0].decode("ascii")
        at += 16
        flags, count = struct.unpack_from("<II", payload, at)
        at += 8
        counts = np.frombuffer(payload, "<u4", count=count, offset=at)
        at += count * 4
        heads = np.frombuffer(payload, "<i4", count=count * 2, offset=at).reshape(-1, 2)
        at += count * 8
        total = int(counts.sum())
        steps = np.frombuffer(
            payload, "<i2", count=(total - count) * 2, offset=at
        ).reshape(-1, 2)
        at += (total - count) * 4

        closed = bool(flags & FLAG_POLYGON)
        bits = np.zeros(0, dtype=bool)
        if closed:
            (packed_length,) = struct.unpack_from("<I", payload, at)
            at += 4
            packed = np.frombuffer(payload, np.uint8, count=packed_length, offset=at)
            at += packed_length
            bits = np.unpackbits(packed)[:total].astype(bool)

        lines, bounds, spacing, shore = [], [], [], []
        cursor = mark = 0
        for index in range(count):
            length = int(counts[index])
            walk = steps[cursor : cursor + length - 1].astype(np.int64)
            cursor += length - 1
            if closed:
                shore.append(bits[mark : mark + length])
                mark += length
            units = np.empty((length, 2), dtype=np.int64)
            units[0] = heads[index]
            if length > 1:
                np.cumsum(walk, axis=0, out=units[1:])
                units[1:] += heads[index]
            line = (units * grid).astype(np.float32)
            lines.append(line)
            bounds.append(
                (
                    float(line[:, 0].min()),
                    float(line[:, 1].min()),
                    float(line[:, 0].max()),
                    float(line[:, 1].max()),
                )
            )
            steps_taken = np.abs(np.diff(line, axis=0)).max(axis=1)
            spacing.append(float(steps_taken.mean()) if steps_taken.size else 0.0)
        layers[name] = Layer(
            name=name,
            lines=tuple(lines),
            bounds=np.asarray(bounds, dtype=np.float64).reshape(-1, 4),
            spacing=np.asarray(spacing, dtype=np.float64),
            closed=closed,
            shore=tuple(shore),
        )

    (city_count,) = struct.unpack_from("<I", payload, at)
    at += 4
    lons = np.frombuffer(payload, "<i4", count=city_count, offset=at) * grid
    at += city_count * 4
    lats = np.frombuffer(payload, "<i4", count=city_count, offset=at) * grid
    at += city_count * 4
    pops = np.frombuffer(payload, "<u4", count=city_count, offset=at)
    at += city_count * 4
    (names_length,) = struct.unpack_from("<I", payload, at)
    at += 4
    blob = payload[at : at + names_length]

    cities, cursor = [], 0
    for index in range(city_count):
        size = blob[cursor]
        cursor += 1
        cities.append(
            City(
                name=blob[cursor : cursor + size].decode("utf-8"),
                longitude=float(lons[index]),
                latitude=float(lats[index]),
                population=int(pops[index]),
            )
        )
        cursor += size
    return Basemap(layers=layers, cities=tuple(cities))


@functools.lru_cache(maxsize=4)
def load(region: str = DEFAULT_REGION) -> Basemap:
    """The compiled basemap for a region, or an empty one if it is not there.

    Cached, because it is asked for by every map that opens and the answer
    never changes. Empty on any fault at all - a missing file, a truncated
    one, a file from a future version - for the same reason a corrupt
    settings file is replaced by defaults rather than reported: the map is
    the feature, and the land behind it is decoration that must not be able
    to stop the aircraft being drawn.
    """
    path = BASEMAP_DIR / f"{region}.bsm"
    try:
        raw = path.read_bytes()
        if not raw.startswith(MAGIC):
            return Basemap({})
        (length,) = struct.unpack_from("<I", raw, len(MAGIC))
        payload = zlib.decompress(raw[len(MAGIC) + 4 :])
        if len(payload) != length:
            return Basemap({})
        return _decode(payload)
    except Exception:  # noqa: BLE001 - decoration must not break the map
        return Basemap({})


__all__ = ["BASEMAP_DIR", "Basemap", "City", "Layer", "load"]
