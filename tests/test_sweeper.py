"""Tests for the band sweep.

These run a whole scan against synthetic air that answers retuning, which is
the only way to put the parts that go wrong under test: step planning, tile
ownership at the overlaps, the deliberate tuner offset that keeps the dead DC
bin off real channels, and the persistence gate that decides what the user is
actually shown.

The acceptance criterion for this phase is "a cold scan of the FM band lists
real stations at correct frequencies with no phantom entries", so that is
tested here directly rather than only in pieces.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.scan.sweeper import (
    DEFAULT_OVERLAP,
    EDGE_GUARD,
    TUNE_OFFSET_HZ,
    Sweeper,
    plan_steps,
    run_sweep,
    usable_span,
)

from . import synth

RATE = 2_400_000
FM_LOW, FM_HIGH = 88_000_000, 108_000_000

# Real local frequencies, all on the US odd-tenths raster.
STATIONS = {
    94_900_000: synth.wfm_station(0.50),
    97_300_000: synth.wfm_station(0.35),
    101_100_000: synth.wfm_station(0.25),
    106_500_000: synth.wfm_station(0.18),
}


def sweep(air: synth.Air, **kwargs):
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, **kwargs)
    return sweeper, run_sweep(sweeper, air, settle_s=0.0, sleep=lambda _: None)


# -- Step planning ---------------------------------------------------------


def test_steps_cover_the_whole_range():
    steps = plan_steps(FM_LOW, FM_HIGH, RATE)
    tile = RATE * (1 - 0.25)

    assert steps[0] - tile / 2 <= FM_LOW
    assert steps[-1] + tile / 2 >= FM_HIGH


def test_steps_overlap_so_nothing_falls_between_them():
    steps = plan_steps(FM_LOW, FM_HIGH, RATE)
    gaps = np.diff(steps)

    assert np.all(gaps < RATE), "consecutive windows must overlap"
    assert np.all(gaps == gaps[0]), "and be evenly spaced"


def test_a_range_narrower_than_one_window_is_a_single_step():
    assert len(plan_steps(162_400_000, 162_550_000, RATE)) == 1


def test_the_fm_band_is_a_dozen_steps():
    """Which at 50 ms each is a pass in well under a second."""
    assert len(plan_steps(FM_LOW, FM_HIGH, RATE)) == 12


def test_a_descending_range_is_rejected():
    with pytest.raises(ValueError, match="ascend"):
        plan_steps(108_000_000, 88_000_000, RATE)


def test_an_impossible_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        plan_steps(FM_LOW, FM_HIGH, RATE, overlap=1.0)


# -- Driving the sweeper ---------------------------------------------------


def test_the_tuner_sits_off_the_tile_centre():
    """So the dead DC bin never lands on a channel of any standard raster."""
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE)
    assert sweeper.current_hz - sweeper.current_tile_hz == TUNE_OFFSET_HZ


def test_the_tuner_never_lands_on_a_channel_raster():
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE)
    for tile in sweeper.steps:
        tuned = tile + sweeper.tune_offset_hz
        for raster in (5_000, 10_000, 12_500, 25_000, 200_000):
            assert tuned % raster != 0, f"{tuned} sits on a {raster} Hz channel"


def test_a_dwell_is_a_whole_number_of_fft_frames():
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, fft_size=4096)
    assert sweeper.dwell_samples % 4096 == 0
    assert sweeper.dwell_samples > 0


def test_progress_runs_from_nothing_to_everything():
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, passes=2)
    assert sweeper.progress.fraction == 0.0

    run_sweep(sweeper, synth.Air(STATIONS), settle_s=0.0, sleep=lambda _: None)

    assert sweeper.complete
    assert sweeper.progress.fraction == 1.0


def test_feeding_a_finished_sweep_does_nothing():
    sweeper, _ = sweep(synth.Air(STATIONS), passes=1)
    before = len(sweeper.signals())
    sweeper.feed(synth.noise(4096))

    assert len(sweeper.signals()) == before


# -- The acceptance criterion ----------------------------------------------


def test_a_cold_scan_of_the_fm_band_finds_the_stations():
    _, result = sweep(synth.Air(STATIONS))
    found = {signal.frequency_hz for signal in result.signals}

    assert found == set(STATIONS), "every planted station, and nothing else"


def test_every_station_is_labelled_as_fm_radio():
    _, result = sweep(synth.Air(STATIONS))

    for signal in result.signals:
        assert signal.label == "FM Radio"
        assert signal.mode == "wfm"
        assert signal.certain


def test_a_quiet_band_reports_no_signals_at_all():
    """A list of phantoms is worse than an empty list, and less honest."""
    _, result = sweep(synth.Air({}))
    assert result.signals == ()


def test_signals_come_back_in_frequency_order():
    _, result = sweep(synth.Air(STATIONS))
    frequencies = [signal.frequency_hz for signal in result.signals]

    assert frequencies == sorted(frequencies)


def test_the_strongest_station_is_reported_strongest():
    _, result = sweep(synth.Air(STATIONS))
    assert result.strongest[0].frequency_hz == 94_900_000


def test_a_station_outside_the_scanned_range_is_not_reported():
    air = synth.Air({**STATIONS, 87_500_000: synth.wfm_station(0.5)})
    _, result = sweep(air)

    assert all(FM_LOW <= s.frequency_hz <= FM_HIGH for s in result.signals)


def test_a_narrow_station_on_a_tile_centre_is_still_found():
    """The DC blind spot: without the tuner offset this station vanishes."""
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, passes=2)
    tile = sweeper.steps[3]
    air = synth.Air({tile: synth.nfm_station(0.4)})
    result = run_sweep(sweeper, air, settle_s=0.0, sleep=lambda _: None)

    assert [s.measured_hz for s in result.signals] == [pytest.approx(tile, abs=2_000)]


def test_a_station_on_a_tile_boundary_is_reported_once():
    """Both neighbouring steps see it; only one of them owns it."""
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, passes=2)
    boundary = (sweeper.steps[2] + sweeper.steps[3]) / 2
    air = synth.Air({boundary: synth.wfm_station(0.5)})
    result = run_sweep(sweeper, air, settle_s=0.0, sleep=lambda _: None)

    assert len(result.signals) == 1


# -- Persistence across passes ---------------------------------------------


class Flicker(synth.Air):
    """Air where one station transmits during the first pass only."""

    def __init__(self, steady, fleeting_hz, steps):
        super().__init__(steady)
        self.fleeting_hz = fleeting_hz
        self.reads = 0
        self.steps = steps

    def read(self, samples):
        first_pass = self.reads < self.steps
        self.reads += 1
        if first_pass and abs(self.fleeting_hz - self.center) < self.rate / 2:
            saved = dict(self.stations)
            self.stations[self.fleeting_hz] = synth.wfm_station(0.5)
            try:
                return super().read(samples)
            finally:
                self.stations = saved
        return super().read(samples)


def test_a_signal_that_appears_once_never_reaches_the_list():
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, passes=3)
    air = Flicker(STATIONS, 99_500_000, len(sweeper.steps))
    result = run_sweep(sweeper, air, settle_s=0.0, sleep=lambda _: None)

    assert 99_500_000 not in {s.frequency_hz for s in result.signals}
    assert set(STATIONS) <= {s.frequency_hz for s in result.signals}


def test_a_single_pass_scan_still_reports_something():
    """Persistence must not mean nothing is ever shown on a one-pass scan."""
    _, result = sweep(synth.Air(STATIONS), passes=1)
    assert len(result.signals) == len(STATIONS)


# -- The stitched picture --------------------------------------------------


def test_the_stitched_spectrum_ascends_and_covers_the_range():
    _, result = sweep(synth.Air(STATIONS), passes=1)

    assert np.all(np.diff(result.frequencies) >= 0)
    assert result.frequencies[0] == pytest.approx(FM_LOW, abs=1_000)
    assert result.frequencies[-1] == pytest.approx(FM_HIGH, abs=1_000)


def test_the_stitched_spectrum_shows_the_stations_it_reported():
    """The detector and the picture must never disagree about what is there."""
    _, result = sweep(synth.Air(STATIONS), passes=1)

    for hz in STATIONS:
        near = np.abs(result.frequencies - hz) < 50_000
        far = np.abs(result.frequencies - (hz + 700_000)) < 50_000
        assert result.spectrum_db[near].max() > result.spectrum_db[far].max() + 20


def test_a_sweep_that_was_stopped_early_still_returns_a_result():
    sweeper = Sweeper(FM_LOW, FM_HIGH, RATE, passes=3)
    result = run_sweep(
        sweeper,
        synth.Air(STATIONS),
        settle_s=0.0,
        sleep=lambda _: None,
        should_stop=lambda: True,
    )

    assert result.signals == ()
    assert result.duration_s >= 0.0


# -- HD Radio in a scan ----------------------------------------------------


def test_an_hd_station_is_flagged_during_the_sweep():
    """The scan is where the user first learns a station carries HD."""
    hd_air = synth.Air(
        {
            94_500_000: lambda n, off, rate: synth.hd_radio_fm(
                n, off, amplitude=0.5, digital_dbc=-14.0, rate=rate
            )
        }
    )
    _, result = sweep(hd_air, passes=2)

    assert len(result.signals) == 1
    signal = result.signals[0]
    assert signal.hd is not None and signal.hd.present
    assert "digital sidebands" in signal.explanation


def test_an_ordinary_station_is_not_flagged_as_hd():
    _, result = sweep(synth.Air({94_500_000: synth.wfm_station(0.5)}), passes=2)

    signal = result.signals[0]
    assert signal.hd is None or not signal.hd.present
    assert "digital sidebands" not in signal.explanation


# -- the tile has to fit inside the part of the window we trust -------------


@pytest.mark.parametrize("rate", [240_000, 288_000, 960_000, 1_200_000, 2_400_000])
def test_a_tile_never_extends_past_the_guard(rate):
    """The tuner sits off-centre, so the tile's far edge is further out than
    its width suggests. Where that overshoots the guard, the bottom of every
    tile is measured and then silently thrown away."""
    span = usable_span(rate)
    furthest = span / 2.0 + TUNE_OFFSET_HZ
    assert furthest <= rate * EDGE_GUARD + 1e-6, rate


def test_the_full_rate_plan_is_unchanged():
    """The FM band's 12 steps are a measured figure; this must not move it."""
    assert usable_span(2_400_000) == 2_400_000 * (1.0 - DEFAULT_OVERLAP)
    assert len(plan_steps(88_000_000, 108_000_000, 2_400_000)) == 12


def test_no_frequency_in_a_range_falls_between_two_tiles():
    """KIRO on 710 kHz vanished from a scan of its own band this way."""
    rate = 240_000
    steps = plan_steps(530_000, 1_700_000, rate)
    span = usable_span(rate)
    for hz in range(530_000, 1_700_001, 1_000):
        owned = [
            c for c in steps if c - span / 2.0 <= hz < c + span / 2.0
        ]
        assert owned, f"{hz} Hz is in no tile"
        # And whichever step owns it can actually see it.
        tuned = owned[0] + TUNE_OFFSET_HZ
        assert abs(hz - tuned) <= rate * EDGE_GUARD, f"{hz} Hz is owned but discarded"
