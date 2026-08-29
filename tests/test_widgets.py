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
from bettersdr.scan.classifier import Strength
from bettersdr.ui.levels import Level
from bettersdr.ui.widgets import aircraftcard, colormaps
from bettersdr.ui.widgets.frequency import (
    DIGITS,
    digit_step_hz,
    format_digits,
    nudge_digit,
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
