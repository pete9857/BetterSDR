"""Window width selection.

The rule these tests pin down came off the air rather than out of a
datasheet: a window that reaches below 0 Hz on the dial is dominated by the
V4 upconverter's oscillator leak, not by radio, and the 8-bit front end has
no headroom to spare for it.
"""

from __future__ import annotations

import pytest

from bettersdr.core.device import DEFAULT_SAMPLE_RATE, MAX_TUNE_HZ, MIN_TUNE_HZ
from bettersdr.core.frontend import (
    HF_EDGE_FRACTION,
    PROBE_SECONDS,
    SUPPORTED_SAMPLE_RATES,
    probe_bytes_for,
    safe_center_hz,
    safe_sample_rate,
)
from bettersdr.dsp.demod import AUDIO_RATE


def test_every_supported_rate_suits_both_the_dongle_and_the_demodulators():
    for rate in SUPPORTED_SAMPLE_RATES:
        assert rate % AUDIO_RATE == 0, f"{rate} is not a whole number of audio blocks"
        assert 225_001 <= rate <= 300_000 or 900_001 <= rate <= 3_200_000


def test_vhf_and_up_keep_the_full_window():
    for hz in (94_900_000, 162_550_000, 1_090_000_000):
        assert safe_sample_rate(hz) == DEFAULT_SAMPLE_RATE


@pytest.mark.parametrize("hz", [530_000, 710_000, 1_000_000, 1_700_000])
def test_the_am_band_never_gets_a_window_reaching_zero(hz):
    rate = safe_sample_rate(hz)
    assert hz - rate / 2 > 0, f"{hz} Hz would sample through 0 Hz at {rate}"


def test_the_window_keeps_a_margin_rather_than_just_clearing_zero():
    """Exactly touching 0 Hz is not good enough - the leak has width."""
    for hz in (600_000, 710_000, 1_100_000, 1_500_000):
        rate = safe_sample_rate(hz)
        assert hz - rate / 2 >= HF_EDGE_FRACTION * rate - 1


def test_it_only_ever_narrows():
    """A caller's preference is a ceiling, never a floor."""
    assert safe_sample_rate(94_900_000, preferred_hz=240_000) == 240_000
    assert safe_sample_rate(710_000, preferred_hz=240_000) == 240_000


def test_710_khz_loses_the_full_window_but_keeps_a_usable_one():
    rate = safe_sample_rate(710_000)
    assert rate < DEFAULT_SAMPLE_RATE
    assert rate >= 960_000


def test_the_lowest_tunable_frequency_still_gets_an_answer():
    """Below ~480 kHz nothing clears zero, and refusing to tune is worse."""
    rate = safe_sample_rate(MIN_TUNE_HZ)
    assert rate in SUPPORTED_SAMPLE_RATES


# -- the tuning guard ------------------------------------------------------


def test_a_frequency_the_dongle_can_reach_is_left_alone():
    for hz in (MIN_TUNE_HZ, 710_000, 98_500_000, MAX_TUNE_HZ):
        assert safe_center_hz(hz) == hz


def test_a_click_below_the_dongles_range_lands_on_its_lowest_frequency():
    """At the bottom of the AM band the window reaches below 500 kHz, so the
    spectrum offers frequencies the hardware cannot tune to. Asking for one
    used to kill the reader thread."""
    assert safe_center_hz(400_000) == MIN_TUNE_HZ
    assert safe_center_hz(0) == MIN_TUNE_HZ
    assert safe_center_hz(-1_000_000) == MIN_TUNE_HZ


def test_a_frequency_above_the_dongles_range_is_clamped_too():
    assert safe_center_hz(2_000_000_000) == MAX_TUNE_HZ


def test_the_guard_always_returns_something_tunable():
    for hz in (-5e6, 0, 1e3, 5e5, 1e8, 1.8e9, 1e12):
        assert MIN_TUNE_HZ <= safe_center_hz(hz) <= MAX_TUNE_HZ


def test_the_visible_am_window_reaches_below_what_can_be_tuned():
    """The reason the guard is needed at all, stated as a fact about the band.

    At the bottom of the AM broadcast band the widest safe window still shows
    frequencies under 500 kHz, so click-to-tune can ask for one.
    """
    low_edge = 530_000 - safe_sample_rate(530_000) / 2
    assert low_edge < MIN_TUNE_HZ


# -- the probe is sized in time, not bytes ---------------------------------


def test_the_probe_reproduces_the_measured_size_at_full_rate():
    assert probe_bytes_for(2_400_000) == 32_768


@pytest.mark.parametrize("rate", SUPPORTED_SAMPLE_RATES)
def test_a_probe_covers_the_same_span_of_time_at_every_rate(rate):
    """A byte count chosen on the FM band is a four-second freeze of the
    reader thread on the AM band: `choose_gain` reads twice per gain setting
    across up to 29 settings."""
    seconds = probe_bytes_for(rate) / 2 / rate
    assert 0.5 * PROBE_SECONDS <= seconds <= 1.5 * PROBE_SECONDS, rate


@pytest.mark.parametrize("rate", SUPPORTED_SAMPLE_RATES)
def test_a_whole_gain_sweep_stays_under_a_second(rate):
    """The reader is not reading while this runs, so it has to be quick."""
    steps = 29  # the R828D's gain table
    seconds = 2 * steps * probe_bytes_for(rate) / 2 / rate
    assert seconds < 1.0, f"{rate}: {seconds:.2f} s"


@pytest.mark.parametrize("rate", SUPPORTED_SAMPLE_RATES)
def test_a_probe_is_a_legal_usb_transfer(rate):
    assert probe_bytes_for(rate) % 512 == 0
