"""Tests for display-widget logic that needs no window.

Colour maps and the frequency readout's arithmetic are both pure functions
hiding inside Qt classes, and both are places where a quiet mistake would look
plausible on screen - a colour ramp that is not monotonic, or a digit that
carries into the wrong decade.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.ui.levels import Level
from bettersdr.ui.widgets import colormaps
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
