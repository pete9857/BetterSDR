"""Compile the aircraft map's background from Natural Earth and the Census.

Run once, by hand, when the map data needs rebuilding. Its output -
`bettersdr/ui/basemap/us.bsm`, about 300 KB - is committed, so nothing at
runtime downloads, parses a shapefile or depends on this file.

    python tools/build_basemap.py

**Why Natural Earth and not OpenStreetMap.** OSM data is ODbL: attribution on
the map, and share-alike obligations on anything derived from the database.
Natural Earth is public domain, with no permission needed and no credit
required - it is credited in the README anyway. The Census files are works of
the United States government and are in the public domain for the same
purpose. This project already has one licensing boundary it has to be careful
about (the nrsc5 subprocess), and this is the version of that question that
can simply be avoided.

**Why land is a polygon and the coastline is no longer a layer of its own.**
Filling land needs closed rings, so the source is `ne_10m_land` rather than
`ne_10m_coastline`. Clipping a ring to a box introduces edges that are not
coastline at all - straight runs along the box - and stroking those would draw
a shoreline across the middle of Canada. So each ring carries one bit per
point saying whether the step to the next point is real shoreline, which is
exactly the `connect` array Qt wants and costs 6 KB for the whole country.
One geometry, filled whole and stroked in pieces, which is also the only way
the fill edge and the shoreline are guaranteed to coincide.

**Why the cities come from two places.** Natural Earth's populated places is
about 480 US entries and stops well above the size of a suburb - it has
Seattle and not Bothell. The Census gazetteer has every incorporated place in
the country, and the population estimates give each one a size to rank it by,
which is what decides who gets a label when two are close together. Natural
Earth still supplies everything outside the United States, which is what a
receiver near either border actually sees.

**Why the shapefiles are parsed here by hand.** The format is a 100-byte
header and a list of length-prefixed records, which is fifty lines of
`struct` - against pulling GDAL, fiona or geopandas into a build that
otherwise needs numpy and Qt. The parsing runs once on a developer's machine
and its output is checked in, so it is not in anybody's dependency tree.

The output format is documented in `bettersdr/ui/basemap/__init__.py`, which
is the only thing that reads it.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "build" / "naturalearth"
OUTPUT = ROOT / "bettersdr" / "ui" / "basemap"

BASE_URL = "https://naciscdn.org/naturalearth"

# Areas, drawn as fills - this is what gives the map land and water rather
# than a line drawing. Lakes are painted back over the land in the water
# colour, which is what makes the Great Lakes look like the Great Lakes.
POLYGON_LAYERS = (
    ("land", "10m/physical/ne_10m_land"),
    ("lakes", "10m/physical/ne_10m_lakes"),
)
# Drawn as lines. State boundaries are what tell an American where they are.
# Rivers and roads were tried and are clutter at these zooms - a map behind
# aircraft has to stay quieter than the aircraft.
LINE_LAYERS = (("states", "10m/cultural/ne_10m_admin_1_states_provinces_lines"),)
CITIES = "10m/cultural/ne_10m_populated_places_simple"

# Every incorporated place in the United States, with a position, and the
# Census Bureau's own estimate of how many people live in each. The two are
# joined on the state and place FIPS codes, which is what GEOID concatenates.
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_place_national.zip"
)
POPULATION_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2024/cities/totals/sub-est2024.csv"
)
# 162 is an incorporated place. The same file also carries county
# subdivisions and place-within-county parts, which would each name the same
# town again in a slightly different position.
PLACE_SUMLEV = "162"
POPULATION_COLUMN = "POPESTIMATE2024"

# Everything the United States covers, including the Aleutians on the far
# side of the date line. Clipping to boxes rather than to a polygon is
# deliberate: it keeps Canada and Mexico either side of the border, which is
# what a receiver near one actually sees out of the window.
#
# **The boxes must not overlap**, and the reason is the fill. Two clipped
# copies of the same island, drawn into one odd-even path, cancel where they
# overlap and leave a hole in the middle of the land. They are laid out to
# abut instead: the mainland box ends at 129 degrees west exactly where the
# Alaskan one begins.
#
# The north edge reaches well past the border because it is now visible. A
# coastline that stopped short only went missing; a *fill* that stops short
# draws southern British Columbia as ocean, and 53 degrees is far enough out
# that no aircraft this receiver can hear is anywhere near it.
BOXES = (
    (-129.0, 24.0, -59.0, 53.0),
    (-180.0, 51.0, -129.0, 71.6),
    (172.0, 51.0, 180.0, 56.0),
    (-161.0, 18.5, -154.0, 22.5),
)

# Areas are clipped to a grid of tiles this many degrees across rather than
# to the boxes whole, and the reason is that a fill has to be a closed ring.
# North America is one ring of 18,000 points, so a window anywhere near it
# fails to cull and the map draws the whole continent to show Puget Sound -
# measured at 17 ms a frame against 5. Cut into tiles it is thirty rings of
# six hundred, and the bounding-box test does its job. The tile edges cost a
# few points each and are marked as not-shoreline like any other cut, so
# nothing is drawn along them; the fills abut exactly, which is also why the
# map fills without antialiasing and strokes with it.
TILE_DEG = 5.0

MAGIC = b"BSDRMAP\x02"
# Degrees per stored unit: 1e-4 is about 11 m, which is finer than the source
# data and far finer than a pixel at any zoom this map reaches.
GRID = 1e-4
# Douglas-Peucker tolerance. At 1:10m the data is already coarser than this,
# so it removes almost nothing from the coast and a good deal from state
# borders, which are long straight lines stored as many points.
TOLERANCE = 0.0003
# The largest delta int16 can hold, with room to spare. A simplified border
# can be one segment several degrees long, so those get points put back.
DELTA_LIMIT = 30_000
# How close to a box edge a point has to be to count as sitting on it. The
# clipper puts its intersections exactly on the bound, so this only has to
# survive the arithmetic that computed them.
EDGE_EPSILON = 1e-9
# Bit 0 of a layer's flags word: the lines are closed rings, to be filled,
# and the layer carries one shoreline bit per point.
FLAG_POLYGON = 1

# Small US places, and larger places everywhere else. 5,000 reaches every
# suburb with a name anybody uses locally without filling the map with
# hamlets; Natural Earth's own list does not go below a city anyway.
MIN_US_POPULATION = 5_000
MIN_POPULATION = 50_000
# How far apart two places with the same name have to be before they are two
# places. A Census position is the centre of the whole place and a Natural
# Earth one is the middle of the built-up part, which can differ by a mile.
SAME_PLACE_DEG = 0.25
NAME_LIMIT = 40

# The Census writes a place's legal status into its name. Nobody says "Kent
# city", and "Nashville-Davidson metropolitan government (balance)" is not a
# label anybody can read off a map at eight point.
STATUS_WORDS = frozenset(
    {
        # Uppercase because that is how the gazetteer writes it, and because
        # a place that became a city since the gazetteer was published still
        # carries the old designation there.
        "CDP",
        "city",
        "town",
        "village",
        "borough",
        "municipality",
        "township",
        "government",
        "corporation",
        "metropolitan",
        "metro",
        "consolidated",
        "unified",
        "urban",
        "comunidad",
    }
)


# -- fetching --------------------------------------------------------------


def download(url: str, name: str) -> Path:
    """One source file, downloaded once and cached under `build/`."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / name
    if not local.exists():
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=300) as response:
            local.write_bytes(response.read())
    return local


def fetch(path: str) -> Path:
    """The zip for one Natural Earth layer."""
    return download(f"{BASE_URL}/{path}.zip", f"{path.rsplit('/', 1)[-1]}.zip")


def member(archive_path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.lower().endswith(suffix):
                return archive.read(name)
    raise KeyError(f"no {suffix} in {archive_path}")


# -- shapefile -------------------------------------------------------------


def polylines(data: bytes) -> list[np.ndarray]:
    """Every part of every polyline or polygon record, as (n, 2) lon/lat."""
    out: list[np.ndarray] = []
    offset = 100
    while offset < len(data):
        _, words = struct.unpack_from(">ii", data, offset)
        content = offset + 8
        shape_type = struct.unpack_from("<i", data, content)[0]
        if shape_type in (3, 5):  # PolyLine, Polygon
            parts_count, point_count = struct.unpack_from("<ii", data, content + 36)
            parts_at = content + 44
            parts = np.frombuffer(data, "<i4", count=parts_count, offset=parts_at)
            points = np.frombuffer(
                data,
                "<f8",
                count=point_count * 2,
                offset=parts_at + parts_count * 4,
            ).reshape(-1, 2)
            edges = [*parts.tolist(), point_count]
            for start, end in zip(edges[:-1], edges[1:], strict=False):
                if end - start >= 2:
                    out.append(np.array(points[start:end], dtype=float))
        offset = content + words * 2
    return out


def point_records(data: bytes) -> np.ndarray:
    found = []
    offset = 100
    while offset < len(data):
        _, words = struct.unpack_from(">ii", data, offset)
        content = offset + 8
        if struct.unpack_from("<i", data, content)[0] == 1:
            found.append(struct.unpack_from("<dd", data, content + 4))
        offset = content + words * 2
    return np.asarray(found, dtype=float)


def dbf_columns(data: bytes, wanted: set[str]) -> dict[str, list]:
    """The few attribute columns the cities need, out of the dBASE file."""
    count, header_len, record_len = struct.unpack_from("<IHH", data, 4)
    fields, at = [], 32
    while data[at] != 0x0D:
        raw = data[at : at + 32]
        fields.append(
            (raw[:11].split(b"\0")[0].decode("latin-1"), chr(raw[11]), raw[16])
        )
        at += 32
    out: dict[str, list] = {key: [] for key in wanted}
    for index in range(count):
        cursor = header_len + index * record_len + 1
        for name, kind, width in fields:
            if name in wanted:
                text = data[cursor : cursor + width].decode("latin-1").strip()
                if kind == "N":
                    try:
                        text = float(text)
                    except ValueError:
                        text = 0.0
                out[name].append(text)
            cursor += width
    return out


# -- geometry --------------------------------------------------------------


def inside(lon: float, lat: float) -> bool:
    return any(
        west <= lon <= east and south <= lat <= north
        for west, south, east, north in BOXES
    )


def overlaps(line: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    west, south, east, north = box
    return bool(
        line[:, 0].max() >= west
        and line[:, 0].min() <= east
        and line[:, 1].max() >= south
        and line[:, 1].min() <= north
    )


def clip(lines: list[np.ndarray]) -> list[np.ndarray]:
    """Keep the runs of each open line that fall inside the boxes.

    One vertex either side of each crossing is kept as well, so a border
    reaches the edge of the region instead of stopping just short of it and
    leaving a visible gap. Open lines only - a polygon has to keep its
    closure, and goes through `clip_polygon`.
    """
    kept: list[np.ndarray] = []
    for line in lines:
        flags = np.fromiter(
            (inside(lon, lat) for lon, lat in line), dtype=bool, count=len(line)
        )
        if not flags.any():
            continue
        padded = flags.copy()
        padded[:-1] |= flags[1:]
        padded[1:] |= flags[:-1]
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        starts = [0, *(edges + 1).tolist(), len(line)]
        for begin, end in zip(starts[:-1], starts[1:], strict=False):
            if end - begin >= 2 and padded[begin]:
                kept.append(line[begin:end])
    return kept


def clip_polygon(
    ring: np.ndarray, box: tuple[float, float, float, float]
) -> np.ndarray:
    """Sutherland-Hodgman against one box: a ring in, a closed ring out.

    Vectorised, because the North American land polygon is 200,000 points and
    this runs against each of the four boxes. Per clip edge, a vertex emits
    the crossing point wherever the boundary is crossed and then the next
    vertex when that one is inside - which is two interleaved arrays and a
    mask rather than a Python loop over the coast of a continent.

    The closure is implicit: the last point joins back to the first.
    """
    for axis, bound, keep_below in (
        (0, box[0], False),
        (0, box[2], True),
        (1, box[1], False),
        (1, box[3], True),
    ):
        if len(ring) < 3:
            return ring[:0]
        column = ring[:, axis]
        here = column <= bound if keep_below else column >= bound
        if here.all():
            continue
        if not here.any():
            return ring[:0]
        following = np.roll(ring, -1, axis=0)
        there = np.roll(here, -1)
        span = following[:, axis] - column
        with np.errstate(divide="ignore", invalid="ignore"):
            along = np.where(span != 0.0, (bound - column) / span, 0.0)
        crossing = ring + (following - ring) * along[:, None]
        crossing[:, axis] = bound
        out = np.empty((len(ring) * 2, 2))
        out[0::2] = crossing
        out[1::2] = following
        mask = np.empty(len(ring) * 2, dtype=bool)
        mask[0::2] = here != there
        mask[1::2] = there
        ring = out[mask]
    return ring


def real_edges(ring: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Which steps around a clipped ring are real shoreline.

    A step is not, when both of its ends sit on the same side of the clipping
    box: that is an edge the clipper drew, and stroking it would put a
    coastline across the middle of a continent. `flags[i]` describes the step
    from point i to point i + 1, so the last entry is about the step that
    closes the ring.
    """
    following = np.roll(ring, -1, axis=0)
    artificial = np.zeros(len(ring), dtype=bool)
    for axis, bound in ((0, box[0]), (0, box[2]), (1, box[1]), (1, box[3])):
        on_bound = np.abs(ring[:, axis] - bound) <= EDGE_EPSILON
        artificial |= on_bound & (np.abs(following[:, axis] - bound) <= EDGE_EPSILON)
    return ~artificial


def simplify(line: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker, iterative so a long coastline cannot blow the stack."""
    if len(line) < 3:
        return line
    keep = np.zeros(len(line), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(line) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start, end = line[first], line[last]
        span = end - start
        length = float(np.hypot(*span))
        middle = line[first + 1 : last]
        if length == 0.0:
            distance = np.hypot(*(middle - start).T)
        else:
            distance = (
                np.abs(
                    span[0] * (start[1] - middle[:, 1])
                    - (start[0] - middle[:, 0]) * span[1]
                )
                / length
            )
        worst = int(np.argmax(distance))
        if distance[worst] > tolerance:
            index = first + 1 + worst
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return line[keep]


def simplify_ring(
    ring: np.ndarray, flags: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify a closed ring without letting one run merge into its neighbour.

    Simplifying the whole ring at once would move a point across the join
    between real shoreline and a clipper's edge, and the one bit per point
    that says which is which would then be describing a different step. So
    each run of like steps is simplified on its own, which preserves every
    run's ends and therefore the ring's closure as well.

    Returns the ring with its first point repeated at the end, and one flag
    per point; the last flag is about nothing and is always false.
    """
    count = len(ring)
    changes = np.flatnonzero(flags != np.roll(flags, 1))
    if changes.size == 0:
        out = simplify(np.vstack((ring, ring[:1])), tolerance)
        out_flags = np.full(len(out), bool(flags[0]))
        out_flags[-1] = False
        return out, out_flags
    # Rotate so that index 0 begins a run, which turns a circular walk into a
    # linear one.
    start = int(changes[0])
    ring = np.roll(ring, -start, axis=0)
    flags = np.roll(flags, -start)
    closed = np.vstack((ring, ring[:1]))
    edges = [*np.flatnonzero(flags != np.roll(flags, 1)).tolist(), count]
    pieces: list[np.ndarray] = []
    values: list[bool] = []
    for begin, end in zip(edges[:-1], edges[1:], strict=False):
        piece = simplify(closed[begin : end + 1], tolerance)
        pieces.append(piece[:-1])
        values.append(bool(flags[begin]))
    out = np.vstack((*pieces, closed[-1:]))
    out_flags = np.concatenate(
        [
            np.full(len(piece), value)
            for piece, value in zip(pieces, values, strict=True)
        ]
        + [np.zeros(1, dtype=bool)]
    )
    return out, out_flags


def split_long(
    line: np.ndarray, flags: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Put points back wherever one step would overflow an int16 delta.

    Simplification turns a straight border into two points a long way apart -
    the 49th parallel is one segment across the whole continent - and the
    encoding stores steps, not positions. Splitting costs a handful of points
    and keeps the decoder a `cumsum`. An inserted point inherits the flag of
    the step it was inserted into, because it is still that same step.
    """
    limit = DELTA_LIMIT * GRID
    if len(line) < 2 or float(np.max(np.abs(np.diff(line, axis=0)))) <= limit:
        return line, flags
    out = [line[0]]
    grown: list[bool] = []
    for index, (previous, point) in enumerate(zip(line[:-1], line[1:], strict=False)):
        steps = int(float(np.max(np.abs(point - previous))) / limit) + 1
        for step in range(1, steps + 1):
            out.append(previous + (point - previous) * step / steps)
            grown.append(bool(flags[index]) if flags is not None else False)
    if flags is None:
        return np.asarray(out), None
    return np.asarray(out), np.asarray([*grown, False], dtype=bool)


def tiles() -> list[tuple[float, float, float, float]]:
    """The region boxes cut into a grid, so no ring is bigger than a tile."""
    out: list[tuple[float, float, float, float]] = []
    for west, south, east, north in BOXES:
        left = west
        while left < east:
            bottom = south
            while bottom < north:
                out.append(
                    (
                        left,
                        bottom,
                        min(left + TILE_DEG, east),
                        min(bottom + TILE_DEG, north),
                    )
                )
                bottom += TILE_DEG
            left += TILE_DEG
    return out


def polygon_lines(
    rings: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Clip every ring to the region, keeping its closure and its real edges."""
    lines: list[np.ndarray] = []
    connect: list[np.ndarray] = []
    grid = tiles()
    for ring in rings:
        for box in grid:
            if not overlaps(ring, box):
                continue
            clipped = clip_polygon(ring, box)
            if len(clipped) < 3:
                continue
            line, flags = simplify_ring(clipped, real_edges(clipped, box), TOLERANCE)
            if len(line) < 4:
                continue
            lines.append(line)
            connect.append(flags)
    return lines, connect


# -- encoding --------------------------------------------------------------


def encode_layer(
    name: str, lines: list[np.ndarray], connect: list[np.ndarray] | None = None
) -> bytes:
    counts, heads, deltas, bits = [], [], [], []
    for index, line in enumerate(lines):
        grown, flags = split_long(line, connect[index] if connect else None)
        quantised = np.round(grown / GRID).astype(np.int64)
        counts.append(len(quantised))
        heads.append(quantised[0])
        deltas.append(np.diff(quantised, axis=0))
        if connect is not None:
            bits.append(np.asarray(flags, dtype=bool))
    step = (
        np.concatenate(deltas).astype("<i2")
        if deltas
        else np.zeros((0, 2), dtype="<i2")
    )
    if deltas and np.abs(np.concatenate(deltas)).max() > 32_767:
        raise ValueError(f"{name}: a delta does not fit in an int16")
    packed = np.packbits(np.concatenate(bits)).tobytes() if bits else b""
    return b"".join(
        (
            name.encode("ascii").ljust(16, b"\0"),
            struct.pack("<II", FLAG_POLYGON if connect is not None else 0, len(lines)),
            np.asarray(counts, dtype="<u4").tobytes(),
            np.asarray(heads, dtype="<i4").tobytes(),
            step.tobytes(),
            struct.pack("<I", len(packed)) if connect is not None else b"",
            packed,
        )
    )


def encode_cities(cities: list[tuple[float, float, int, str]]) -> bytes:
    cities.sort(key=lambda city: -city[2])
    lons = np.round([city[0] for city in cities], 4) / GRID
    lats = np.round([city[1] for city in cities], 4) / GRID
    names = bytearray()
    for city in cities:
        encoded = city[3].encode("utf-8")[:NAME_LIMIT]
        names.append(len(encoded))
        names += encoded
    return b"".join(
        (
            struct.pack("<I", len(cities)),
            lons.astype("<i4").tobytes(),
            lats.astype("<i4").tobytes(),
            np.asarray([city[2] for city in cities], dtype="<u4").tobytes(),
            struct.pack("<I", len(names)),
            bytes(names),
        )
    )


# -- cities ----------------------------------------------------------------


def place_name(raw: str) -> str:
    """The name a person would use: `Kent city` is Kent.

    `Nashville-Davidson metropolitan government (balance)` is Nashville-
    Davidson, which is the same rule applied until it runs out of status
    words rather than a special case.
    """
    name = raw.replace("(balance)", "").replace("(pt.)", "").strip()
    words = name.split()
    while len(words) > 1 and words[-1] in STATUS_WORDS:
        words.pop()
    return " ".join(words)


def natural_earth_cities() -> list[tuple[float, float, int, str]]:
    """The places Natural Earth knows about, anywhere in the region."""
    archive = fetch(CITIES)
    coords = point_records(member(archive, ".shp"))
    table = dbf_columns(member(archive, ".dbf"), {"name", "pop_max"})
    return [
        (float(lon), float(lat), int(pop), str(city))
        for (lon, lat), city, pop in zip(
            coords, table["name"], table["pop_max"], strict=False
        )
        if inside(lon, lat) and pop >= MIN_POPULATION
    ]


def census_cities() -> list[tuple[float, float, int, str]]:
    """Every incorporated US place above `MIN_US_POPULATION`.

    The gazetteer says where a place is, the population estimates say how big
    it is, and GEOID is the state and place FIPS codes that the two share. A
    place with no estimate - almost always one dissolved or renamed since the
    gazetteer was published - is dropped rather than drawn with a size of
    zero, which would put it last in every ranking.
    """
    archive = download(GAZETTEER_URL, "gaz_place_national.zip")
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read(zipped.namelist()[0]).decode("latin-1").splitlines()
    header = [field.strip() for field in raw[0].split("\t")]

    estimates: dict[str, int] = {}
    text = download(POPULATION_URL, "sub-est.csv").read_text("latin-1").splitlines()
    for row in csv.DictReader(text):
        if row["SUMLEV"] == PLACE_SUMLEV:
            estimates[row["STATE"] + row["PLACE"]] = int(row[POPULATION_COLUMN])

    found: list[tuple[float, float, int, str]] = []
    for line in raw[1:]:
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) < len(header):
            continue
        record = dict(zip(header, fields, strict=False))
        population = estimates.get(record["GEOID"])
        if population is None or population < MIN_US_POPULATION:
            continue
        lon, lat = float(record["INTPTLONG"]), float(record["INTPTLAT"])
        if inside(lon, lat):
            found.append((lon, lat, population, place_name(record["NAME"])))
    return found


def merge_cities(
    census: list[tuple[float, float, int, str]],
    world: list[tuple[float, float, int, str]],
) -> list[tuple[float, float, int, str]]:
    """The Census list, plus everything Natural Earth has that it does not.

    Natural Earth is the only source for Vancouver, Tijuana and the rest, and
    for the few large US places that are not incorporated - Honolulu is a
    census-designated place and has no estimate of its own. Anything it names
    that the Census has already placed nearby is dropped, because two dots a
    mile apart carrying the same name is worse than either one alone.
    """
    placed: dict[str, list[tuple[float, float]]] = {}
    for lon, lat, _, name in census:
        placed.setdefault(name.casefold(), []).append((lon, lat))
    out = list(census)
    for entry in world:
        lon, lat, _, name = entry
        near = placed.get(name.casefold(), ())
        if any(
            abs(lon - other_lon) < SAME_PLACE_DEG
            and abs(lat - other_lat) < SAME_PLACE_DEG
            for other_lon, other_lat in near
        ):
            continue
        out.append(entry)
    return out


# -- building --------------------------------------------------------------


def build(region: str) -> Path:
    blobs: list[bytes] = []
    for name, path in POLYGON_LAYERS:
        lines, connect = polygon_lines(polylines(member(fetch(path), ".shp")))
        blobs.append(encode_layer(name, lines, connect))
        points = sum(len(line) for line in lines)
        shore = sum(int(flags.sum()) for flags in connect)
        print(
            f"  {name:<8} {len(lines):>5} rings {points:>7} points, "
            f"{shore:>7} of them shoreline"
        )
    for name, path in LINE_LAYERS:
        raw = clip(polylines(member(fetch(path), ".shp")))
        lines = [simplify(line, TOLERANCE) for line in raw]
        lines = [line for line in lines if len(line) >= 2]
        blobs.append(encode_layer(name, lines))
        points = sum(len(line) for line in lines)
        print(f"  {name:<8} {len(lines):>5} lines {points:>7} points")

    payload = bytearray(struct.pack("<dI", GRID, len(blobs)))
    for blob in blobs:
        payload += blob

    united_states = census_cities()
    chosen = merge_cities(united_states, natural_earth_cities())
    payload += encode_cities(chosen)
    print(
        f"  {'cities':<8} {len(chosen):>5} places, "
        f"{len(united_states)} of them from the Census"
    )

    body = zlib.compress(bytes(payload), 9)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"{region}.bsm"
    destination.write_bytes(MAGIC + struct.pack("<I", len(payload)) + body)
    print(
        f"\n  wrote {destination.relative_to(ROOT)} - "
        f"{destination.stat().st_size / 1024:.0f} KB "
        f"({len(payload) / 1024:.0f} KB uncompressed)"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="us")
    args = parser.parse_args(argv)
    print(f"Building the {args.region} basemap from Natural Earth and the Census")
    build(args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
