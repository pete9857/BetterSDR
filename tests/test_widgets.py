"""Tests for display-widget logic that needs no window.

Colour maps and the frequency readout's arithmetic are both pure functions
hiding inside Qt classes, and both are places where a quiet mistake would look
plausible on screen - a colour ramp that is not monotonic, or a digit that
carries into the wrong decade.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.decode.adsb import Aircraft
from bettersdr.decode.pocsag import Page, PocsagState
from bettersdr.scan import bandplan, monitor, voice
from bettersdr.scan.classifier import Signal, Strength
from bettersdr.ui import results
from bettersdr.ui.levels import Level
from bettersdr.ui.listen_view import band_headline
from bettersdr.ui.widgets import (
    activitycard,
    aircraftcard,
    colormaps,
    pagerlog,
    rangepicker,
)
from bettersdr.ui.widgets.frequency import (
    DIGITS,
    digit_step_hz,
    format_digits,
    nudge_digit,
)
from bettersdr.ui.widgets.planemap import (
    HIT_RADIUS_PX,
    MIN_SPAN_NM,
    Projection,
    altitude_colour,
    distance_nm,
    fit,
    format_degrees,
    graticule_step,
    nearest_aircraft,
    nice_number,
    scale_bar,
    update_trails,
)
from bettersdr.ui.widgets.spectrum import (
    RANK_ALLOCATION,
    RANK_BAND,
    RANK_CHANNEL,
    RANK_TUNED,
    Label,
    channel_cells,
    without_collisions,
)
from bettersdr.ui.widgets.viewspan import (
    FULL,
    MAX_ZOOM,
    View,
    clamped,
    panned,
    slider_for_zoom,
    span,
    zoom_for_slider,
    zoomed,
)

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


# -- clicking an aircraft on the map ---------------------------------------
#
# Both mistakes this can make look entirely ordinary on screen: selecting the
# aircraft underneath the one aimed at, and selecting one that is nowhere near
# the pointer because nothing else was closer.


def _hit_setup():
    """A projection and two aircraft a long way apart on it."""
    projection = fit([(47.5, -122.5), (47.7, -122.1)], 800.0, 400.0)
    north = plane(47.7, -122.1, icao=0x111111)
    south = plane(47.5, -122.5, icao=0x222222)
    return projection, (north, south)


def test_a_click_on_an_aircraft_finds_that_aircraft():
    projection, aircraft = _hit_setup()
    for target in aircraft:
        x, y = projection.to_pixel(target.latitude, target.longitude)
        assert nearest_aircraft(projection, aircraft, x, y) == target.icao


def test_a_click_on_empty_sky_is_a_click_on_nothing():
    """Not "whichever is least far away" - the radius is what makes putting
    a selection down possible at all."""
    projection, aircraft = _hit_setup()
    x, y = projection.to_pixel(47.7, -122.1)
    assert nearest_aircraft(projection, aircraft, x, y + HIT_RADIUS_PX * 3) is None
    assert nearest_aircraft(projection, (), x, y) is None


def test_a_near_miss_still_counts():
    """The symbol is smaller than anything anyone can point at."""
    projection, aircraft = _hit_setup()
    x, y = projection.to_pixel(47.7, -122.1)
    assert nearest_aircraft(projection, aircraft, x + 8.0, y - 6.0) == 0x111111


def test_the_nearer_of_two_wins():
    projection, aircraft = _hit_setup()
    x1, y1 = projection.to_pixel(47.7, -122.1)
    x2, y2 = projection.to_pixel(47.5, -122.5)
    midway = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    towards_north = (
        midway[0] + (x1 - midway[0]) * 0.98,
        midway[1] + (y1 - midway[1]) * 0.98,
    )
    assert nearest_aircraft(projection, aircraft, *towards_north) == 0x111111


def test_an_aircraft_with_no_position_cannot_be_clicked():
    """It is not drawn, so it must not be selectable - otherwise a click on
    open water selects something invisible."""
    projection, _ = _hit_setup()
    silent = plane(callsign="ASA123", icao=0x333333)
    assert nearest_aircraft(projection, [silent], 400.0, 200.0) is None


# -- the Lookup button's URL -----------------------------------------------


def test_lookup_uses_the_address_because_every_aircraft_has_one():
    assert aircraftcard.lookup_url("A1B2C3") == (
        "https://flightaware.com/live/modes/A1B2C3/redirect"
    )


def test_a_callsign_narrows_the_lookup_to_todays_flight():
    assert aircraftcard.lookup_url("A1B2C3", "ASA123") == (
        "https://flightaware.com/live/modes/A1B2C3/ident/ASA123/redirect"
    )


def test_a_lookup_url_is_escaped_and_tidied():
    """A callsign comes off the air eight characters wide, and what is left
    after the padding is stripped is not guaranteed to be tidy."""
    assert aircraftcard.lookup_url("a1b2c3", " asa 123 ") == (
        "https://flightaware.com/live/modes/A1B2C3/ident/ASA%20123/redirect"
    )
    assert aircraftcard.lookup_url("A1B2C3", "   ") == (
        "https://flightaware.com/live/modes/A1B2C3/redirect"
    )


# -- Discover list: ordering and filtering ---------------------------------


def _found(freq_mhz: float, snr_db: float, label: str, icon: str = "wave") -> Signal:
    return Signal(
        frequency_hz=freq_mhz * 1e6,
        measured_hz=freq_mhz * 1e6,
        bandwidth_hz=12_500.0,
        peak_dbfs=-40.0,
        snr_db=snr_db,
        label=label,
        icon=icon,
        description="",
        mode="nfm",
        demod_bandwidth_hz=12_500.0,
        confidence=0.6,
        reasons=(),
    )


AIRBAND = (
    _found(118.3, 22.0, "Aircraft radio", "plane"),
    _found(119.9, 14.0, "Aircraft radio", "plane"),
    _found(120.1, 31.0, "Unmodulated carrier"),
    _found(121.5, 18.0, "Unmodulated carrier"),
    _found(122.7, 9.0, "Unmodulated carrier"),
)


def test_the_default_order_is_strongest_first():
    ordered = results.sort_signals(AIRBAND)
    assert [s.snr_db for s in ordered] == [31.0, 22.0, 18.0, 14.0, 9.0]


def test_frequency_order_runs_up_the_band():
    ordered = results.sort_signals(AIRBAND, "frequency")
    assert [s.frequency_hz for s in ordered] == sorted(
        s.frequency_hz for s in AIRBAND
    )


def test_grouping_by_type_keeps_each_kind_together():
    labels = [s.label for s in results.sort_signals(AIRBAND, "kind")]
    runs = [label for index, label in enumerate(labels) if index == 0
            or labels[index - 1] != label]
    assert len(runs) == len(set(runs))


def test_grouping_leads_with_the_kind_holding_the_strongest_signal():
    """Otherwise the airband leads with eighty-three carriers, not aircraft."""
    ordered = results.sort_signals(AIRBAND, "kind")
    assert ordered[0].label == "Unmodulated carrier"
    assert ordered[0].snr_db == 31.0


def test_an_unrecognised_order_is_the_default_rather_than_an_error():
    """It arrives from a settings file, and a stale key must not stop the app."""
    assert results.sort_signals(AIRBAND, "loudness") == results.sort_signals(AIRBAND)


@pytest.mark.parametrize("order", [key for _, key in results.SORTS])
def test_every_order_is_fully_determined(order: str):
    """Two equal signals swapping places would slam an open card shut."""
    tied = (
        _found(101.1, 20.0, "FM radio station", "music"),
        _found(99.5, 20.0, "FM radio station", "music"),
    )
    once = results.sort_signals(tied, order)
    assert once == results.sort_signals(tuple(reversed(tied)), order)


def test_every_order_keeps_every_signal():
    for _, order in results.SORTS:
        assert set(results.sort_signals(AIRBAND, order)) == set(AIRBAND)


def test_hiding_a_kind_removes_only_that_kind():
    listed = results.visible(AIRBAND, {"Unmodulated carrier"})
    assert {s.label for s in listed} == {"Aircraft radio"}
    assert len(listed) == 2


def test_filtering_leaves_the_order_it_was_given():
    ordered = results.sort_signals(AIRBAND, "frequency")
    listed = results.visible(ordered, {"Unmodulated carrier"})
    assert [s.frequency_hz for s in listed] == sorted(s.frequency_hz for s in listed)


def test_hiding_nothing_is_the_list_itself():
    assert results.visible(AIRBAND) == AIRBAND


def test_the_chips_lead_with_the_kind_holding_the_pile():
    kinds = results.summarise(AIRBAND)
    assert [k.label for k in kinds] == ["Unmodulated carrier", "Aircraft radio"]
    assert [k.count for k in kinds] == [3, 2]


def test_a_chip_carries_its_count_even_when_it_is_hiding_them():
    """A hidden kind must never look like a sweep that found nothing."""
    kinds = results.summarise(AIRBAND, {"Unmodulated carrier"})
    carriers = next(k for k in kinds if k.label == "Unmodulated carrier")
    assert not carriers.shown
    assert "3" in carriers.chip


def test_a_chip_takes_the_icon_of_what_it_stands_for():
    kinds = results.summarise(AIRBAND)
    assert {k.label: k.icon for k in kinds}["Aircraft radio"] == "plane"


def test_the_hidden_count_is_what_the_filter_is_holding_back():
    assert results.hidden_count(AIRBAND, {"Unmodulated carrier"}) == 3
    assert results.hidden_count(AIRBAND) == 0


def test_a_remembered_filter_naming_nothing_present_hides_nothing():
    """Last week's airband filter must not silently empty an FM scan."""
    assert results.visible(AIRBAND, {"Pager"}) == AIRBAND
    assert all(k.shown for k in results.summarise(AIRBAND, {"Pager"}))


# -- Stepping through the list from the listening screen -------------------


def test_next_and_previous_walk_the_list_in_the_order_it_is_shown():
    listed = results.sort_signals(AIRBAND, "frequency")
    here = listed[2].frequency_hz
    assert results.neighbour(listed, here, 1) is listed[3]
    assert results.neighbour(listed, here, -1) is listed[1]


def test_the_list_wraps_at_both_ends():
    """A car radio's seek button does, and stopping dead needs explaining."""
    listed = results.sort_signals(AIRBAND, "frequency")
    assert results.neighbour(listed, listed[-1].frequency_hz, 1) is listed[0]
    assert results.neighbour(listed, listed[0].frequency_hz, -1) is listed[-1]


def test_a_dial_a_few_hundred_hertz_off_is_still_on_that_signal():
    """Clicking a card leaves the radio near the centre the sweep measured."""
    listed = results.sort_signals(AIRBAND, "frequency")
    assert results.neighbour(listed, listed[1].frequency_hz + 900.0, 1) is listed[2]


def test_a_frequency_nowhere_in_the_list_enters_it_from_the_end_pressed():
    """The dial goes where it likes; the buttons must still mean something."""
    listed = results.sort_signals(AIRBAND, "frequency")
    assert results.neighbour(listed, 94_900_000.0, 1) is listed[0]
    assert results.neighbour(listed, 94_900_000.0, -1) is listed[-1]


def test_stepping_skips_whatever_the_filter_is_hiding():
    """The buttons walk the screen the user was looking at, not the sweep."""
    listed = results.visible(
        results.sort_signals(AIRBAND, "frequency"), {"Unmodulated carrier"}
    )
    stepped = results.neighbour(listed, listed[0].frequency_hz, 1)
    assert stepped.label == "Aircraft radio"


def test_an_empty_list_has_no_neighbour_in_either_direction():
    assert results.neighbour((), 98_100_000.0, 1) is None
    assert results.neighbour((), 98_100_000.0, -1) is None


def test_one_result_steps_to_itself_rather_than_to_nothing():
    only = results.sort_signals(AIRBAND)[:1]
    assert results.neighbour(only, only[0].frequency_hz, 1) is only[0]


# -- what the listening screen says about a frequency ----------------------


def test_the_header_describes_the_channel_the_dial_is_on():
    """The name is drawn on the ribbon; what is left up here is the why."""
    for level in Level:
        name, info = band_headline(156_800_000, level)
        assert name == "Marine VHF"
        assert "emergency" in info


def test_the_regulators_own_name_for_a_channel_waits_for_standard():
    simple = band_headline(156_800_000, Level.SIMPLE)[1]
    standard = band_headline(156_800_000, Level.STANDARD)[1]
    assert "Officially" not in simple
    assert standard.startswith(simple)
    assert "Distress, Safety and Calling" in standard


def test_a_band_with_no_named_channels_reads_as_it_always_did():
    name, info = band_headline(94_900_000, Level.STANDARD)
    assert name == "FM Radio"
    assert info.startswith("Music and talk")


def test_simple_is_told_nothing_is_here_rather_than_who_owns_it():
    name, info = band_headline(180_000_000, Level.SIMPLE)
    assert name == "Unallocated"
    assert info == "Nothing is normally broadcast here."


def test_standard_says_what_an_empty_stretch_is_licensed_for():
    name, info = band_headline(180_000_000, Level.STANDARD)
    assert name == "Unallocated"
    assert "television" in info


def test_expert_is_told_at_least_as_much_as_standard():
    """Nothing is ever removed by moving up a level."""
    for hz in (156_800_000, 180_000_000, 94_900_000, 27_185_000):
        standard = band_headline(hz, Level.STANDARD)
        assert band_headline(hz, Level.EXPERT) == standard


def test_a_frequency_the_dial_can_reach_always_says_something():
    """Every step of the dial, at Standard, has an answer that is not blank."""
    for hz in range(500_000, 1_766_000_000, 971_000):
        name, info = band_headline(hz, Level.STANDARD)
        assert name and info
        assert info != "Nothing is normally broadcast here."


# -- the channel blocks on the ribbon --------------------------------------


def _cells(centre_hz: float, span_hz: float, tuned_hz: float | None = None):
    low, high = centre_hz - span_hz / 2.0, centre_hz + span_hz / 2.0
    return channel_cells(
        bandplan.overlapping(low, high), low, high, tuned_hz or centre_hz
    )


def test_the_tuned_channel_is_drawn_even_when_nothing_else_fits():
    """A 2.4 MHz window holds 96 marine channels and no room for any name.

    That is the moment somebody most needs telling which one they are on, so
    the grid stays away and the one channel that matters is drawn anyway.
    """
    cells = _cells(156_800_000, 2_400_000)
    assert len(cells) == 1
    only = cells[0]
    assert only.name == "Channel 16"
    assert only.tuned


def test_zooming_in_brings_the_whole_grid_out():
    cells = _cells(156_800_000, 240_000)
    assert len(cells) > 8
    assert sum(cell.tuned for cell in cells) == 1
    assert [cell.name for cell in cells] == sorted(
        (cell.name for cell in cells),
        key=lambda name: [c.start_hz for c in cells if c.name == name][0],
    )


def test_a_channel_block_is_as_wide_as_the_ground_its_name_covers():
    """The block and `Band.channel` must claim the same span.

    A block drawn wider than the name applies to is a receiver pointing at a
    channel it is not on.
    """
    only = _cells(156_800_000, 2_400_000)[0]
    assert only.end_hz - only.start_hz == pytest.approx(25_000)
    assert (only.start_hz + only.end_hz) / 2.0 == pytest.approx(156_800_000)


def test_a_band_with_no_named_channels_draws_no_second_lane():
    assert _cells(94_900_000, 2_400_000) == ()


def test_a_dial_between_two_named_channels_still_marks_the_nearer_one():
    cells = _cells(156_800_000, 2_400_000, tuned_hz=156_806_000)
    assert [cell.name for cell in cells] == ["Channel 16"]


def test_nothing_is_drawn_for_a_window_with_no_width():
    assert channel_cells(bandplan.load(), 156_800_000, 156_800_000, 156_800_000) == ()


# -- which names survive when two want the same pixels ----------------------


def test_two_names_that_do_not_touch_are_both_kept():
    labels = [
        Label(hz=100.0, half_hz=10.0, text="left", rank=RANK_BAND),
        Label(hz=200.0, half_hz=10.0, text="right", rank=RANK_BAND),
    ]
    assert [label.text for label in without_collisions(labels)] == ["left", "right"]


def test_the_name_of_the_band_being_listened_to_wins():
    """The 162.550 MHz case, in miniature.

    A wide grey allocation next door used to shoulder the weather channels'
    own name off the ribbon, because whichever was drawn first simply stayed.
    """
    labels = [
        Label(hz=100.0, half_hz=60.0, text="Federal government", rank=RANK_ALLOCATION),
        Label(hz=140.0, half_hz=40.0, text="Weather Radio", rank=RANK_TUNED),
    ]
    assert [label.text for label in without_collisions(labels)] == ["Weather Radio"]


def test_a_band_name_outranks_a_channel_name():
    labels = [
        Label(hz=100.0, half_hz=30.0, text="Channel 16", rank=RANK_CHANNEL),
        Label(hz=110.0, half_hz=30.0, text="Marine VHF", rank=RANK_BAND),
    ]
    assert [label.text for label in without_collisions(labels)] == ["Marine VHF"]


def test_the_answer_does_not_depend_on_the_order_they_were_built_in():
    labels = [
        Label(hz=300.0, half_hz=20.0, text="third", rank=RANK_BAND),
        Label(hz=100.0, half_hz=20.0, text="first", rank=RANK_BAND),
        Label(hz=200.0, half_hz=20.0, text="second", rank=RANK_BAND),
    ]
    kept = without_collisions(labels)
    assert [label.text for label in kept] == ["first", "second", "third"]
    assert without_collisions(list(reversed(labels))) == kept


def test_a_name_dropped_for_touching_does_not_then_block_a_third():
    """Rejection is against what was kept, not against everything asked for.

    Otherwise one unlucky middle label takes its neighbour down with it.
    """
    labels = [
        Label(hz=100.0, half_hz=30.0, text="kept", rank=RANK_TUNED),
        Label(hz=120.0, half_hz=30.0, text="dropped", rank=RANK_BAND),
        Label(hz=150.0, half_hz=15.0, text="also kept", rank=RANK_BAND),
    ]
    assert [label.text for label in without_collisions(labels)] == [
        "kept",
        "also kept",
    ]


# -- zooming into the window and panning across it -------------------------

CENTRE = 98_100_000.0
RATE = 2_400_000.0


def _span(view: View):
    return span(CENTRE, RATE, view)


def test_the_default_view_is_the_whole_window():
    low, high = _span(FULL)
    assert (low, high) == (CENTRE - RATE / 2.0, CENTRE + RATE / 2.0)


@pytest.mark.parametrize("zoom", [1.0, 1.7, 8.0, MAX_ZOOM])
def test_a_view_is_that_many_times_narrower_than_the_window(zoom: float):
    low, high = _span(View(zoom, 0.0))
    assert high - low == pytest.approx(RATE / zoom)


@pytest.mark.parametrize("zoom", [1.0, 1.7, 8.0, MAX_ZOOM])
@pytest.mark.parametrize("offset", [-5.0, -0.3, 0.0, 0.3, 5.0])
def test_the_view_never_leaves_the_captured_window(zoom: float, offset: float):
    """There is nothing outside it to draw.

    A pane that scrolled into blackness at the edge would look like a
    receiver that had gone deaf, which is the worst thing a display can look
    like when the radio is working perfectly.
    """
    low, high = _span(clamped(zoom, offset))
    assert low >= CENTRE - RATE / 2.0 - 1e-6
    assert high <= CENTRE + RATE / 2.0 + 1e-6


def test_zoom_stops_at_the_whole_window_and_at_the_limit():
    assert clamped(0.2, 0.0).zoom == 1.0
    assert clamped(1e6, 0.0).zoom == MAX_ZOOM


def test_zooming_out_all_the_way_recovers_the_whole_window():
    """Including the pan: at 1x there is nowhere else for the view to be."""
    view = panned(View(16.0, 0.0), 900_000.0, RATE)
    assert view.offset > 0.0
    assert _span(zoomed(view, 1 / 64.0)) == _span(FULL)


@pytest.mark.parametrize("anchor", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_zooming_holds_still_whatever_is_under_the_pointer(anchor: float):
    """The wheel zooms about the cursor, which is only useful if it is exact.

    A zoom that drifts by a few percent per notch walks the signal the user
    was pointing at off the side of the pane in three notches, and the fault
    reads as the radio drifting rather than as the display doing arithmetic.
    """
    view = View(2.0, 0.1)
    low, high = _span(view)
    held = low + anchor * (high - low)

    for factor in (1.2, 1.2**5, 1 / 1.2):
        after = zoomed(view, factor, anchor)
        low, high = _span(after)
        assert low + anchor * (high - low) == pytest.approx(held, abs=1.0)


def test_a_zoom_that_would_leave_the_window_slides_back_inside_it():
    """Zooming out from the top of the window, under a pointer on the left.

    Holding that frequency where it is would put the new right-hand edge
    past the end of what was captured. The clamp wins: the view widens to
    what was asked for and stops at the edge of the window rather than
    following the pointer off the end of it.
    """
    view = zoomed(View(8.0, 0.4375), 1 / 4.0, anchor=0.0)
    low, high = _span(view)
    assert high == pytest.approx(CENTRE + RATE / 2.0)
    assert high - low == pytest.approx(RATE / 2.0)


def test_panning_moves_the_view_by_what_it_was_asked_for():
    low, high = _span(panned(View(4.0, 0.0), 100_000.0, RATE))
    assert low == pytest.approx(CENTRE - RATE / 8.0 + 100_000.0)
    assert high - low == pytest.approx(RATE / 4.0)


def test_panning_stops_at_the_edge_and_comes_straight_back():
    """A drag held against the edge must not build up somewhere to unwind.

    The pan is measured from the frequency the drag started on rather than
    accumulated, so what the clamp discards is discarded rather than stored:
    dragging back moves the view on the first pixel, not after undoing the
    distance the pointer travelled past the end.
    """
    at_edge = panned(View(4.0, 0.0), RATE, RATE)
    assert _span(at_edge)[1] == pytest.approx(CENTRE + RATE / 2.0)
    assert _span(panned(at_edge, -RATE / 8.0, RATE))[1] == pytest.approx(
        CENTRE + RATE / 2.0 - RATE / 8.0
    )


def test_panning_at_the_full_window_does_nothing():
    assert panned(FULL, 500_000.0, RATE) == FULL


def test_the_zoom_slider_is_the_same_ratio_at_both_ends():
    """A linear slider spends half its travel between 32x and 64x."""
    steps = [zoom_for_slider(value) for value in (0, 25, 50, 75, 100)]
    assert steps[0] == 1.0
    assert steps[-1] == MAX_ZOOM
    ratios = [b / a for a, b in zip(steps, steps[1:], strict=False)]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


@pytest.mark.parametrize("value", [0, 1, 37, 99, 100])
def test_the_slider_round_trips_through_the_zoom_it_sets(value: int):
    """Or the wheel and the slider fight each other one step at a time."""
    assert slider_for_zoom(zoom_for_slider(value)) == value


# -- the monitor's card -----------------------------------------------------


def test_how_long_ago_is_said_in_units_a_person_would_use():
    """"last heard 214s ago" is a stopwatch reading, not an answer.

    What the reader wants from this line is whether the channel is alive now,
    alive lately, or a note about earlier - so the units change with the scale
    rather than the number growing without bound.
    """
    assert activitycard.heard_phrase(0.0) == "just now"
    assert activitycard.heard_phrase(1.4) == "just now"
    assert activitycard.heard_phrase(9.0) == "9s ago"
    assert activitycard.heard_phrase(59.0) == "59s ago"
    assert activitycard.heard_phrase(214.0) == "4 min ago"
    assert activitycard.heard_phrase(7200.0) == "2 h ago"


def test_every_verdict_has_a_colour_and_only_voice_gets_the_accent():
    """A row of bright badges says nothing; one does.

    "This channel is static" is information the eye should skate over, so the
    greys outnumber the colours deliberately.
    """
    for kind in (
        voice.VOICE,
        voice.MUSIC,
        voice.DATA,
        voice.TONE,
        voice.NOISE,
        voice.SILENCE,
        voice.UNCLEAR,
    ):
        assert kind in activitycard.SOUND_COLOURS
    greys = [
        kind
        for kind, colour in activitycard.SOUND_COLOURS.items()
        if colour == "#6d7b89"
    ]
    assert len(greys) > len(activitycard.SOUND_COLOURS) - len(greys)


def test_an_activity_can_be_summarised_by_the_chip_code():
    """`results.summarise` is handed activities, not the signals inside them.

    It duck-types on `label` and `icon`, and the monitor's list must carry
    both - a chip counting a channel under a name its card no longer shows
    would hide the wrong rows.
    """
    signal = _found(155.0, 20.0, "Two-way radio", "walkie")
    activity = monitor.Activity(
        signal=signal,
        sightings=3,
        passes=4,
        first_heard=0.0,
        last_heard=1.0,
        snr_db=20.0,
        peak_snr_db=22.0,
        active=True,
        verdict=None,
        auditions=1,
        voice_heard=1,
        skipped=False,
        held=False,
        now=2.0,
    )
    assert activity.label == signal.label
    assert activity.icon == signal.icon
    kinds = results.summarise([activity])
    assert kinds[0].label == "Two-way radio"
    assert kinds[0].count == 1


# -- the whole-dial range picker ---------------------------------------------
#
# The Expert discovery screen replaces the band chips with every stretch of
# dial the dongle can reach. Three things in it are arithmetic hiding inside a
# widget, and all three are places a quiet mistake would look plausible: the
# label, the step count that decides whether a sweep is five seconds or two
# minutes, and the merge that turns ticked boxes into ranges.


def test_a_range_is_labelled_by_its_span_first():
    """Two stretches are both called Federal government; no two start alike."""
    fm = bandplan.Segment.of(
        next(band for band in bandplan.load() if band.name == "FM Radio")
    )
    assert rangepicker.span_label(fm) == "88 MHz-108 MHz: FM Radio"


def test_the_step_count_is_the_one_the_sweep_will_really_use():
    """A count derived from width alone is out by ten at the bottom of the
    dial: the AM band is swept through a 240 kHz window, not a 2.4 MHz one."""
    assert rangepicker.step_count(((88_000_000, 108_000_000, None),)) == 12
    assert rangepicker.step_count(((530_000, 1_700_000, 240_000),)) == 9
    assert rangepicker.step_count(
        ((530_000, 1_700_000, 240_000), (88_000_000, 108_000_000, None))
    ) == 21


def test_a_range_with_no_stated_window_still_gets_a_safe_one():
    beacons = next(
        segment
        for segment in bandplan.coverage(500_000, 1_766_000_000)
        if segment.start_hz == 500_000
    )
    assert beacons.sample_rate_hz is None
    assert rangepicker.effective_rate(beacons) < 2_400_000


def test_how_long_a_sweep_will_take_is_said_before_it_is_started():
    """A two-minute progress bar nobody was warned about reads as a hang."""
    assert "s a pass" in rangepicker.duration_phrase(12)
    assert "min a pass" in rangepicker.duration_phrase(991)


def test_the_summary_names_one_range_and_counts_several():
    segments = bandplan.coverage(500_000, 1_766_000_000)
    by_name = {segment.name: segment for segment in segments}
    assert rangepicker.summarise(()).startswith("Nothing selected")
    assert rangepicker.summarise((by_name["FM Radio"],)).startswith("FM Radio")
    two = rangepicker.summarise((by_name["FM Radio"], by_name["AM Radio"]))
    assert two.startswith("2 ranges")


def test_ranges_for_merges_what_touches_and_shares_a_window():
    segments = bandplan.coverage(500_000, 1_766_000_000)
    by_name = {segment.name: segment for segment in segments}
    assert rangepicker.ranges_for(
        [by_name["Business and public safety"], by_name["Marine VHF"]]
    ) == ((150_800_000, 162_025_000, 2_400_000),)
    assert rangepicker.ranges_for(
        [by_name["AM Radio"], by_name["Long-distance fixed links"]]
    ) == (
        (530_000, 1_700_000, 240_000),
        (1_700_000, 1_800_000, 2_400_000),
    )
