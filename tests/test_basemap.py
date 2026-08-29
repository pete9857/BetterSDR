"""Tests for the aircraft map's background data.

Two different things are tested here and they fail in different ways. The
codec is arithmetic - a wrong stride or a missed cumulative sum puts a
coastline in the wrong ocean, which is obvious. The *shipped file* is data,
and data fails quietly: a build run against the wrong region, or a layer
that silently came back empty, produces an application that looks fine until
somebody notices there is no land behind the aircraft.

So the last few tests assert things about `us.bsm` itself. They are the only
tests in this project that depend on a committed data file, and they exist
because nothing else would notice if it went wrong.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from bettersdr.ui.basemap import MAGIC, Basemap, City, Layer, load

SEATTLE = (-122.33, 47.60)


def make_layer(lines: list[np.ndarray], name: str = "coast") -> Layer:
    bounds = np.asarray(
        [
            (
                line[:, 0].min(),
                line[:, 1].min(),
                line[:, 0].max(),
                line[:, 1].max(),
            )
            for line in lines
        ],
        dtype=np.float64,
    ).reshape(-1, 4)
    spacing = np.asarray(
        [float(np.abs(np.diff(line, axis=0)).max(axis=1).mean()) for line in lines]
    )
    return Layer(name=name, lines=tuple(lines), bounds=bounds, spacing=spacing)


def straight(start_lon: float, count: int, step: float = 0.01) -> np.ndarray:
    lons = start_lon + np.arange(count) * step
    return np.column_stack((lons, np.full(count, 47.6))).astype(np.float32)


# -- culling ---------------------------------------------------------------


def test_a_line_outside_the_window_is_not_drawn():
    layer = make_layer([straight(-122.5, 20), straight(-70.0, 20)])
    visible = layer.visible(-123.0, 47.0, -122.0, 48.0)
    assert len(visible) == 1
    assert visible[0][0][0] == pytest.approx(-122.5)


def test_a_line_crossing_the_window_is_drawn_whole():
    """Clipping to the exact window would cost more than drawing the ends."""
    layer = make_layer([straight(-124.0, 400)])
    assert len(layer.visible(-123.0, 47.0, -122.9, 48.0)) == 1


def test_thinning_drops_points_but_never_the_ends():
    """A coastline that stops a stride short leaves a gap where lines meet."""
    line = straight(-122.5, 101)
    layer = make_layer([line])
    whole = layer.visible(-180, -90, 180, 90)[0]
    thinned = layer.visible(-180, -90, 180, 90, step_deg=0.1)[0]
    assert len(whole) == 101
    assert 5 < len(thinned) < 30
    assert thinned[0][0] == pytest.approx(line[0][0])
    assert thinned[-1][0] == pytest.approx(line[-1][0])


def test_a_wider_view_asks_for_fewer_points():
    layer = make_layer([straight(-122.5, 400)])
    counts = [
        len(layer.visible(-180, -90, 180, 90, step_deg=step)[0])
        for step in (0.0, 0.02, 0.1, 0.5)
    ]
    assert counts == sorted(counts, reverse=True)


def test_thinning_has_a_floor():
    """Past a point a coastline stops being a coastline and becomes a polygon."""
    layer = make_layer([straight(-122.5, 2_000)])
    assert len(layer.visible(-180, -90, 180, 90, step_deg=1_000.0)[0]) >= 60


# -- cities ----------------------------------------------------------------


def test_cities_come_back_largest_first_and_capped():
    cities = tuple(
        City(name=f"c{n}", longitude=-122.3, latitude=47.6, population=1_000 * n)
        for n in range(20, 0, -1)
    )
    book = Basemap(layers={}, cities=cities)
    found = book.visible_cities(-123.0, 47.0, -122.0, 48.0, limit=3)
    assert [city.population for city in found] == [20_000, 19_000, 18_000]


def test_a_city_outside_the_window_is_not_returned():
    book = Basemap(
        layers={},
        cities=(City("Seattle", -122.33, 47.60, 700_000),
                City("Miami", -80.2, 25.8, 400_000)),
    )
    found = book.visible_cities(-123.0, 47.0, -122.0, 48.0)
    assert [city.name for city in found] == ["Seattle"]


# -- the file --------------------------------------------------------------


def test_a_missing_file_is_an_empty_basemap():
    """The land is decoration; it must not be able to stop the map drawing."""
    book = load.__wrapped__("no-such-region")
    assert book.empty
    assert book.visible_cities(-180, -90, 180, 90) == []
    assert book.layer("coast") is None


def test_a_corrupt_file_is_an_empty_basemap(tmp_path, monkeypatch):
    import bettersdr.ui.basemap as basemap

    monkeypatch.setattr(basemap, "BASEMAP_DIR", tmp_path)
    (tmp_path / "broken.bsm").write_bytes(MAGIC + struct.pack("<I", 99) + b"nonsense")
    assert basemap.load.__wrapped__("broken").empty

    (tmp_path / "alien.bsm").write_bytes(b"NOTAMAP\x01" + b"\0" * 32)
    assert basemap.load.__wrapped__("alien").empty

    # The right magic and a valid stream, but the length does not agree.
    body = zlib.compress(b"\0" * 40, 9)
    (tmp_path / "short.bsm").write_bytes(MAGIC + struct.pack("<I", 999) + body)
    assert basemap.load.__wrapped__("short").empty


# -- the shipped data ------------------------------------------------------


def test_the_shipped_basemap_has_all_three_layers():
    book = load()
    assert not book.empty
    assert set(book.layers) == {"land", "lakes", "states"}
    assert book.layer("land").points > 30_000
    assert book.layer("states").points > 5_000


def test_the_areas_are_closed_and_the_borders_are_not():
    """A fill needs rings. A ring that lost its last point is a wedge of
    ocean painted over the middle of the country."""
    book = load()
    for name in ("land", "lakes"):
        layer = book.layer(name)
        assert layer.closed
        assert len(layer.shore) == len(layer.lines)
        for line, flags in zip(layer.lines, layer.shore, strict=True):
            assert len(flags) == len(line)
            assert line[0] == pytest.approx(line[-1], abs=1e-4)
    assert not book.layer("states").closed
    assert book.layer("states").shore == ()


def test_most_of_the_land_edge_is_real_shoreline():
    """The rest is where the build's clipping box cut a continent in half,
    and stroking those would draw a coastline across Manitoba."""
    land = load().layer("land")
    flags = np.concatenate(land.shore)
    assert 0.9 < float(flags.mean()) < 1.0


def test_thinning_a_ring_thins_its_shoreline_flags_with_it():
    """Flags that slid along the ring would stroke the clipping box."""
    land = load().layer("land")
    lon, lat = SEATTLE
    window = (lon - 1.0, lat - 1.0, lon + 1.0, lat + 1.0)
    for step in (0.0, 0.01, 0.2):
        rings = land.visible_rings(*window, step_deg=step)
        assert rings
        for ring, flags in rings:
            assert flags is not None
            assert len(flags) == len(ring)
            assert ring[0] == pytest.approx(ring[-1], abs=1e-4)


def test_the_shipped_basemap_is_the_united_states():
    """A build run against the wrong region would otherwise go unnoticed."""
    land = load().layer("land")
    west = min(float(line[:, 0].min()) for line in land.lines)
    east = max(float(line[:, 0].max()) for line in land.lines)
    south = min(float(line[:, 1].min()) for line in land.lines)
    north = max(float(line[:, 1].max()) for line in land.lines)
    assert west >= -180.0 and east <= 180.0
    assert 18.0 < south < 26.0  # the Florida Keys and Hawaii
    assert 70.0 < north < 72.0  # the Alaskan north coast


def test_the_coast_is_where_seattle_is():
    """Puget Sound is 30 miles of coastline; something must be near the city."""
    land = load().layer("land")
    lon, lat = SEATTLE
    near = land.visible(lon - 0.3, lat - 0.3, lon + 0.3, lat + 0.3)
    assert near
    closest = min(
        float(np.hypot(*(line - np.array([lon, lat])).T).min()) for line in near
    )
    assert closest < 0.15  # about ten miles


def test_the_cities_are_named_and_ordered():
    cities = load().cities
    assert len(cities) > 3_000
    assert cities[0].name == "New York"
    populations = [city.population for city in cities]
    assert populations == sorted(populations, reverse=True)
    assert all(city.population >= 5_000 for city in cities)
    assert not any(name.endswith((" city", " town", " CDP")) for name in
                   [city.name for city in cities])


def test_the_suburbs_are_there_and_not_only_the_cities():
    """Natural Earth alone has Seattle and stops well above Bothell, which is
    the whole reason the Census gazetteer is in the build at all."""
    named = {city.name for city in load().visible_cities(
        -122.6, 47.2, -121.8, 48.0, limit=200
    )}
    assert {"Seattle", "Bellevue", "Kent", "Bothell", "Renton"} <= named


def test_seattle_is_in_the_city_list_where_it_should_be():
    found = load().visible_cities(-122.6, 47.4, -122.0, 47.8)
    assert "Seattle" in [city.name for city in found]


def test_no_line_wraps_the_globe():
    """The Aleutians cross the date line, and a wrapped line draws across all
    of North America. Splitting them at the build is what stops it, and
    nothing on screen would say it had gone wrong.

    71 degrees is just wider than the widest clipping box, which the North
    American land ring fills from side to side; a wrapped line would be
    nearer 350."""
    for layer in load().layers.values():
        width = layer.bounds[:, 2] - layer.bounds[:, 0]
        assert width.max() < 71.0
