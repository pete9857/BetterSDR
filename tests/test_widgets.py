"""Tests for display-widget logic that needs no window.

Colour maps and the frequency readout's arithmetic are both pure functions
hiding inside Qt classes, and both are places where a quiet mistake would look
plausible on screen - a colour ramp that is not monotonic, or a digit that
carries into the wrong decade.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.core.bookmarks import Bookmark
from bettersdr.core.history import Station
from bettersdr.decode.adsb import Aircraft
from bettersdr.decode.pocsag import Page, PocsagState
from bettersdr.scan.classifier import Strength
from bettersdr.ui.levels import Level
from bettersdr.ui.widgets import aircraftcard, colormaps, pagerlog
from bettersdr.ui.widgets.frequency import (
    DIGITS,
    digit_step_hz,
    format_digits,
    nudge_digit,
)
from bettersdr.ui.widgets.planemap import (
    MIN_SPAN_NM,
    Projection,
    altitude_colour,
    distance_nm,
    fit,
    format_degrees,
    graticule_step,
    nice_number,
    scale_bar,
    update_trails,
)
from bettersdr.ui.widgets.quicktune import ago, quick_list, spent

# -- Colour maps -----------------------------------------------------------


@pytest.mark.parametrize("name", colormaps.NAMES)
def test_every_map_produces_a_full_table(name: str):
    table = colormaps.lookup_table(name)
    assert table.shape == (256, 3)
    assert table.dtype == np.uint8


@pytest.mark.parametrize("name", colormaps.NAMES)
def test_every_map_starts_dark_and_ends_bright(name: str):
    """An empty band must look empty and a strong signal must look strong."""
    table = colormaps.lookup_table(name).astype(int)
    assert table[0].sum() < table[-1].sum()


def test_classic_map_starts_at_black():
    assert list(colormaps.lookup_table("classic")[0]) == [0, 0, 0]


def test_grayscale_is_a_straight_ramp():
    table = colormaps.lookup_table("grayscale")
    assert np.all(table[:, 0] == table[:, 1])
    assert np.all(table[:, 1] == table[:, 2])
    assert np.all(np.diff(table[:, 0].astype(int)) >= 0)


def test_level_count_is_configurable():
    assert colormaps.lookup_table("viridis", levels=64).shape == (64, 3)


def test_unknown_map_is_rejected():
    with pytest.raises(ValueError, match="unknown colour map"):
        colormaps.lookup_table("ultraviolet")


def test_too_few_levels_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        colormaps.lookup_table("classic", levels=1)


# -- Frequency digits ------------------------------------------------------


def test_digit_steps_run_from_gigahertz_down_to_hertz():
    assert digit_step_hz(0) == 1_000_000_000
    assert digit_step_hz(DIGITS - 1) == 1
    assert digit_step_hz(3) == 1_000_000  # the megahertz digit


def test_digit_index_outside_the_display_is_rejected():
    with pytest.raises(ValueError, match="digit index"):
        digit_step_hz(DIGITS)


def test_nudging_a_digit_moves_only_its_decade():
    assert nudge_digit(98_500_000, 3, 1) == 99_500_000
    assert nudge_digit(98_500_000, 4, -1) == 98_400_000


def test_nudging_carries_the_way_a_wheel_should():
    """Winding 99.9 up by a tenth must give 100.0, not roll over to 90.0."""
    assert nudge_digit(99_900_000, 4, 1) == 100_000_000


def test_format_is_fixed_width_so_the_display_never_reflows():
    assert format_digits(98_500_000) == "0098500000"
    assert len(format_digits(1_766_000_000)) == DIGITS
    assert len(format_digits(0)) == DIGITS


# -- Signal icons ----------------------------------------------------------


def test_every_band_in_the_plan_has_a_glyph():
    """A band with no picture is the one that looks broken on the card."""
    from bettersdr.scan import bandplan
    from bettersdr.ui.widgets import icons

    for band in bandplan.load():
        assert band.icon in icons.GLYPHS, f"{band.name} uses icon {band.icon!r}"


def test_every_icon_the_classifier_can_invent_has_a_glyph():
    """The shape-only labels use icons no band plan entry needs to mention."""
    from bettersdr.ui.widgets import icons

    for name in ("wave", "chip", "walkie", "music", "question"):
        assert name in icons.GLYPHS


def test_an_unknown_icon_falls_back_rather_than_raising():
    from bettersdr.ui.widgets import icons

    assert icons.glyph("unicorn") == icons.FALLBACK
    assert icons.glyph("") == icons.FALLBACK


# -- Progressive disclosure ------------------------------------------------


def test_levels_are_ordered_so_comparisons_reveal_controls():
    assert Level.SIMPLE < Level.STANDARD < Level.EXPERT
    # This comparison is exactly what decides a control's visibility.
    assert Level.EXPERT >= Level.STANDARD
    assert not Level.SIMPLE >= Level.STANDARD


def test_every_level_describes_itself():
    for level in Level:
        assert level.label
        assert level.description


# -- Aircraft cards --------------------------------------------------------
#
# All of this is arithmetic and wording hiding inside a Qt class, and all of
# it is where a quiet mistake would look plausible: a heading named as the
# opposite compass point, or an aircraft with no altitude shown at zero feet.


def test_north_wraps_round_from_both_sides():
    assert aircraftcard.compass(0.0) == "north"
    assert aircraftcard.compass(359.0) == "north"
    assert aircraftcard.compass(22.0) == "north"
    assert aircraftcard.compass(338.0) == "north"


@pytest.mark.parametrize(
    ("degrees", "name"),
    [(90.0, "east"), (180.0, "south"), (270.0, "west"), (135.0, "south-east")],
)
def test_the_cardinal_points_are_not_swapped(degrees: float, name: str):
    assert aircraftcard.compass(degrees) == name


def test_an_aircraft_with_no_altitude_is_not_at_zero_feet():
    """The same rule as the classifier's "Unknown signal": nothing is better
    than a confident wrong answer."""
    assert aircraftcard.altitude_text(None) == ""


def test_altitude_says_whether_it_is_changing():
    assert aircraftcard.altitude_text(31_000) == "31,000 ft"
    assert aircraftcard.altitude_text(31_000, False, 1_800) == "31,000 ft, climbing"
    assert aircraftcard.altitude_text(9_000, False, -1_200) == "9,000 ft, descending"


def test_a_level_cruise_is_not_reported_as_a_climb():
    """A vertical rate wanders by a hundred feet a minute in level flight,
    and an aircraft permanently 'climbing' teaches the reader to ignore it."""
    assert aircraftcard.altitude_text(31_000, False, 100) == "31,000 ft"


def test_an_aircraft_on_the_ground_says_so_instead_of_its_altitude():
    assert aircraftcard.altitude_text(50, True, 0) == "On the ground"


def test_speed_keeps_whichever_half_it_has():
    assert aircraftcard.speed_text(451.2, 270.0) == "451 kt   heading west (270°)"
    assert aircraftcard.speed_text(None, 90.0) == "heading east (90°)"
    assert aircraftcard.speed_text(120.0, None) == "120 kt"
    assert aircraftcard.speed_text(None, None) == ""


def test_a_position_needs_both_halves():
    assert aircraftcard.position_text(47.6, None) == ""
    assert aircraftcard.position_text(None, -122.3) == ""


def test_the_hemispheres_are_the_right_way_round():
    assert aircraftcard.position_text(47.6062, -122.3321) == (
        "47.6062° N, 122.3321° W"
    )
    assert aircraftcard.position_text(-33.87, 151.21) == "33.8700° S, 151.2100° E"


def test_age_is_worded_the_way_people_say_it():
    assert aircraftcard.age_text(0.4) == "just now"
    assert aircraftcard.age_text(12.0) == "12 s ago"
    assert aircraftcard.age_text(180.0) == "3 min ago"


def test_strength_rises_with_the_level_and_never_reads_as_a_maybe():
    """Every message on screen has passed its checkword, so the weakest is a
    real aircraft rather than a guess - one bar, not none. The values are the
    range six aircraft actually arrived at, heard indoors."""
    assert aircraftcard.strength_from_rssi(-3.0) == Strength.STRONG
    assert aircraftcard.strength_from_rssi(-18.0) == Strength.GOOD
    assert aircraftcard.strength_from_rssi(-24.0) == Strength.FAIR
    assert aircraftcard.strength_from_rssi(-80.0) == Strength.WEAK


def _plane(**kwargs) -> Aircraft:
    fields = {"icao": 0x4B1234, "messages": 12, "age_s": 3.0, "rssi_dbfs": -24.0}
    return Aircraft(**(fields | kwargs))


def test_a_card_without_a_callsign_is_named_by_its_address():
    assert _plane().label == "4B1234"
    assert _plane(callsign="BAW49").label == "BAW49"


def test_the_summary_drops_what_the_aircraft_has_not_said():
    assert aircraftcard.summary_line(_plane()) == ""
    assert aircraftcard.summary_line(_plane(altitude_ft=31_000)) == "31,000 ft"
    assert aircraftcard.summary_line(
        _plane(altitude_ft=31_000, ground_speed_kt=451.0, track_deg=270.0)
    ) == "31,000 ft   ·   451 kt   heading west (270°)"


def test_the_provenance_line_grows_with_the_level():
    plane = _plane()
    assert aircraftcard.heard_line(plane, Level.SIMPLE) == "Heard 3 s ago"
    standard = aircraftcard.heard_line(plane, Level.STANDARD)
    assert "12 messages" in standard and "ICAO" not in standard
    expert = aircraftcard.heard_line(plane, Level.EXPERT)
    assert "ICAO 4B1234" in expert and "-24 dBFS" in expert


# -- Pager messages ---------------------------------------------------------


def _page(**kwargs) -> Page:
    fields = {
        "capcode": 1234568,
        "function": 3,
        "kind": "alphanumeric",
        "text": "CALL SWITCHBOARD",
        "baud": 1200,
        "received": 0.0,
    }
    return Page(**(fields | kwargs))


def test_a_capcode_is_padded_the_way_operators_write_it():
    assert pagerlog.capcode_text(1234568) == "1234568"
    assert pagerlog.capcode_text(45678) == "0045678"


def test_a_page_with_no_message_says_so_rather_than_showing_nothing():
    assert pagerlog.message_text(_page(kind="tone", text="")) == "Beep - no message"
    assert pagerlog.message_text(_page(text="")) == "Empty message"
    assert pagerlog.message_text(_page()) == "CALL SWITCHBOARD"


def test_the_detail_line_reports_lost_codewords_only_when_there_are_some():
    assert pagerlog.detail_text(_page()) == "1200 bps   alphanumeric"
    assert "1 lost codeword" in pagerlog.detail_text(_page(errors=1))
    assert "3 lost codewords" in pagerlog.detail_text(_page(errors=3))


def test_the_status_line_distinguishes_a_quiet_channel_from_no_decoder():
    assert pagerlog.status_text(None) == ""
    assert pagerlog.status_text(PocsagState()) == "Listening"
    heard = PocsagState(
        pages=(_page(),), baud=1200, codewords_ok=48, codewords_bad=16, batches=4
    )
    assert pagerlog.status_text(heard) == "1200 bps   1 message   75% of codewords intact"


# -- Favourites and recently played ----------------------------------------

KUOW = 94_900_000
KING = 98_100_000


def test_a_favourite_and_a_recent_on_one_frequency_appear_once():
    favourites = [Bookmark("KUOW", KUOW, favourite=True)]
    recent = [
        Station(frequency_hz=KUOW + 1_000, name="KUOW", last_heard=90.0),
        Station(frequency_hz=KING, name="KING", last_heard=80.0),
    ]
    chips = quick_list(favourites, recent, now=100.0)
    assert [c.frequency_hz for c in chips] == [KUOW, KING]
    assert chips[0].favourite and not chips[1].favourite


def test_favourites_are_never_crowded_out():
    favourites = [
        Bookmark(f"F{n}", 88_100_000 + n * 200_000, favourite=True) for n in range(9)
    ]
    recent = [Station(frequency_hz=KING, last_heard=10.0)]
    chips = quick_list(favourites, recent, now=100.0, limit=4)
    assert len(chips) == 9
    assert all(chip.favourite for chip in chips)


def test_the_strip_is_empty_before_anything_has_happened():
    assert quick_list([], [], now=0.0) == []


def test_wording():
    assert ago(5) == "just now"
    assert ago(600) == "10 minutes ago"
    assert ago(3600) == "an hour ago"
    assert ago(86_400) == "yesterday"
    assert ago(3 * 86_400) == "3 days ago"
    assert spent(30) == "30 seconds"
    assert spent(600) == "10 minutes"
    assert spent(3600) == "an hour"
    assert spent(4 * 3600) == "4 hours"


# -- The aircraft map ------------------------------------------------------

# Somewhere over Seattle, which is where the real aircraft in this project's
# notes were heard.
HOME_LAT, HOME_LON = 47.60, -122.33


def plane(lat=None, lon=None, icao=0xA1B2C3, **extra):
    return Aircraft(
        icao=icao, messages=10, age_s=1.0, latitude=lat, longitude=lon, **extra
    )


def test_nothing_to_frame_is_not_an_error():
    """An aircraft is heard several times before it has sent a position."""
    assert fit([], 800.0, 400.0) is None
    assert fit([(HOME_LAT, HOME_LON)], 0.0, 400.0) is None


def test_a_position_survives_the_round_trip():
    """The projection is invertible, so a click on the map is a place."""
    projection = fit([(HOME_LAT, HOME_LON), (47.8, -122.0)], 800.0, 400.0)
    x, y = projection.to_pixel(47.7, -122.2)
    lat, lon = projection.to_position(x, y)
    assert lat == pytest.approx(47.7, abs=1e-9)
    assert lon == pytest.approx(-122.2, abs=1e-9)


def test_north_is_up_and_east_is_right():
    """The sign test. Both errors draw a perfectly plausible map.

    A flipped latitude puts every arrival where a departure should be, and
    nothing else on the screen contradicts it.
    """
    projection = fit([(47.4, -122.6), (47.8, -122.0)], 800.0, 400.0)
    centre = projection.to_pixel(HOME_LAT, -122.3)
    north = projection.to_pixel(HOME_LAT + 0.1, -122.3)
    east = projection.to_pixel(HOME_LAT, -122.2)
    assert north[1] < centre[1]
    assert east[0] > centre[0]


def test_every_aircraft_lands_inside_the_window():
    points = [(47.4, -122.7), (47.9, -121.9), (47.6, -122.3)]
    projection = fit(points, 800.0, 400.0)
    for lat, lon in points:
        x, y = projection.to_pixel(lat, lon)
        assert 0.0 <= x <= 800.0
        assert 0.0 <= y <= 400.0


def test_one_aircraft_does_not_zoom_to_absurdity():
    """A single point has no extent, and a map fitted to it has no scale."""
    projection = fit([(HOME_LAT, HOME_LON)], 800.0, 400.0)
    assert projection.span_nm == pytest.approx(MIN_SPAN_NM)


def test_longitude_is_squeezed_by_the_latitude():
    """A degree of longitude is shorter than a degree of latitude up here.

    Without the cosine the map is stretched east-west by half at this
    latitude, which puts a north-south airway on screen as a diagonal.
    """
    projection = Projection(HOME_LAT, HOME_LON, 1000.0, 800.0, 400.0)
    east = projection.to_pixel(HOME_LAT, HOME_LON + 1.0)[0] - 400.0
    north = 200.0 - projection.to_pixel(HOME_LAT + 1.0, HOME_LON)[1]
    assert east < north
    assert east / north == pytest.approx(0.674, abs=0.01)


def test_a_degree_of_latitude_is_sixty_nautical_miles():
    """The definition, and the check that the units are what they claim."""
    assert distance_nm(47.0, -122.0, 48.0, -122.0) == pytest.approx(60.0, abs=0.1)
    assert distance_nm(47.0, -122.0, 47.0, -122.0) == 0.0


def test_the_scale_bar_measures_what_it_says():
    projection = fit([(47.4, -122.7), (47.9, -121.9)], 800.0, 400.0)
    distance, pixels = scale_bar(projection)
    # Whatever it says, that many pixels really is that far on this map.
    start = projection.to_position(100.0, 200.0)
    end = projection.to_position(100.0 + pixels, 200.0)
    assert distance_nm(*start, *end) == pytest.approx(distance, rel=0.02)


def test_round_numbers_are_round():
    assert nice_number(37.0) == 20.0
    assert nice_number(7.0) == 5.0
    assert nice_number(1.4) == 1.0
    assert nice_number(0.23) == 0.2
    assert nice_number(0.0) == 1.0


def test_the_graticule_gets_a_few_lines_at_any_zoom():
    for points in (
        [(47.5, -122.4), (47.7, -122.2)],
        [(40.0, -130.0), (50.0, -110.0)],
    ):
        projection = fit(points, 800.0, 400.0)
        step = graticule_step(projection)
        span = projection.height / projection.scale
        assert 2.0 <= span / step <= 12.0


def test_degrees_are_labelled_with_a_hemisphere():
    assert format_degrees(47.6, "lat") == "47.60°N"
    assert format_degrees(-122.33, "lon") == "122.33°W"


def test_altitude_is_a_colour_and_the_ground_is_its_own():
    low = altitude_colour(2_000.0)
    high = altitude_colour(38_000.0)
    assert low.name() != high.name()
    assert altitude_colour(0.0, on_ground=True).name() != low.name()
    # Not knowing the altitude is not the same as being on the ground.
    assert altitude_colour(None).name() != altitude_colour(0.0, True).name()


def test_a_trail_grows_only_when_the_aircraft_moves():
    """A position arrives twice a second; a trail of one place is not a trail."""
    trails = {}
    for _ in range(5):
        update_trails(trails, [plane(47.6, -122.3)])
    assert trails[0xA1B2C3] == [(47.6, -122.3)]
    update_trails(trails, [plane(47.61, -122.3)])
    assert len(trails[0xA1B2C3]) == 2


def test_a_trail_is_capped():
    trails = {}
    for step in range(40):
        update_trails(trails, [plane(47.6 + step * 0.01, -122.3)], limit=10)
    assert len(trails[0xA1B2C3]) == 10
    assert trails[0xA1B2C3][-1] == (pytest.approx(47.99), -122.3)


def test_a_trail_goes_when_its_aircraft_does():
    """Otherwise the map slowly fills with the paths of aircraft long gone."""
    trails = {}
    update_trails(trails, [plane(47.6, -122.3)])
    update_trails(trails, [])
    assert trails == {}


def test_an_aircraft_with_no_position_has_no_trail():
    trails = {}
    update_trails(trails, [plane(callsign="ASA123")])
    assert trails == {}
