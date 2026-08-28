"""Tests for signal detection.

Two failure modes matter here and they pull in opposite directions. A detector
that invents signals makes the app untrustworthy in a way the user cannot
argue with - there is nothing there and it says there is. A detector that
misses real ones makes it useless. So these tests check both ends: quiet air
must produce nothing at all, and planted signals must come back at the
frequency and width they were planted with.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp.psd import Spectrum
from bettersdr.scan.detector import (
    Detection,
    Persistence,
    detect,
    merge_nearby,
    noise_floor_curve,
)

from . import synth

RATE = 2_400_000
FRAMES = 4096 * 24


def spectrum_of(iq: np.ndarray, fft_size: int = 4096) -> tuple[np.ndarray, float]:
    spectrum = Spectrum(fft_size=fft_size, sample_rate=RATE)
    return spectrum.process(iq), spectrum.bin_width_hz


# -- Noise floor -----------------------------------------------------------


def test_floor_of_flat_noise_is_flat():
    db, bin_width = spectrum_of(synth.noise(FRAMES, rms=0.01))
    floor = noise_floor_curve(db, bin_width)

    assert float(np.ptp(floor)) < 3.0
    assert -100.0 < float(np.median(floor)) < -50.0


def test_floor_follows_a_sloping_band():
    """A flat threshold would go blind at one end of a tilted band."""
    db, bin_width = spectrum_of(synth.noise(FRAMES, rms=0.01))
    tilt = np.linspace(-20.0, 20.0, db.size).astype(np.float32)
    floor = noise_floor_curve(db + tilt, bin_width)

    assert floor[-1] - floor[0] > 30.0


def test_a_strong_signal_does_not_lift_its_own_floor():
    """If it did, a loud station would hide underneath itself."""
    iq = synth.scene(
        FRAMES, [synth.fm(FRAMES, 300_000, amplitude=0.6)], noise_rms=0.01
    )
    db, bin_width = spectrum_of(iq)
    floor = noise_floor_curve(db, bin_width)

    peak = int(np.argmax(db))
    assert db[peak] - floor[peak] > 30.0


def test_floor_of_an_empty_spectrum_is_empty():
    assert noise_floor_curve(np.zeros(0), 100.0).size == 0


def test_a_kernel_wider_than_the_span_falls_back_to_one_number():
    db, bin_width = spectrum_of(synth.noise(FRAMES, rms=0.01))
    floor = noise_floor_curve(db, bin_width, kernel_hz=RATE * 10)

    assert float(np.ptp(floor)) == 0.0


# -- Detection -------------------------------------------------------------


def test_quiet_air_produces_no_detections():
    """The single most damaging failure: signals that are not there."""
    db, bin_width = spectrum_of(synth.noise(FRAMES, rms=0.01))
    assert detect(db, bin_width) == []


def test_planted_signals_come_back_at_the_right_frequencies():
    planted = [-700_000.0, 100_000.0, 800_000.0]
    iq = synth.scene(
        FRAMES,
        [synth.fm(FRAMES, offset, amplitude=0.4) for offset in planted],
        noise_rms=0.01,
    )
    db, bin_width = spectrum_of(iq)
    found = detect(db, bin_width, center_hz=98_500_000)

    assert len(found) == len(planted)
    for detection, offset in zip(found, planted, strict=True):
        assert detection.center_hz == pytest.approx(98_500_000 + offset, abs=2_000)


def test_measured_width_matches_the_modulation():
    """Broadcast FM is wide, a two-way radio is narrow, and it must show."""
    wide, bin_width = spectrum_of(
        synth.scene(FRAMES, [synth.fm(FRAMES, 400_000, deviation_hz=75_000)], 0.01)
    )
    narrow, _ = spectrum_of(
        synth.scene(FRAMES, [synth.fm(FRAMES, 400_000, deviation_hz=2_500)], 0.01)
    )

    assert detect(wide, bin_width)[0].bandwidth_hz > 120_000
    assert detect(narrow, bin_width)[0].bandwidth_hz < 40_000


def test_snr_is_measured_against_the_local_floor():
    iq = synth.scene(FRAMES, [synth.fm(FRAMES, 500_000, amplitude=0.5)], 0.01)
    db, bin_width = spectrum_of(iq)
    detection = detect(db, bin_width)[0]

    assert detection.snr_db > 20.0
    assert detection.peak_dbfs > detection.floor_dbfs


def test_raising_the_threshold_drops_the_weaker_signal():
    """This is the Sensitivity control, so the direction of it has to hold."""
    iq = synth.scene(
        FRAMES,
        [
            synth.fm(FRAMES, -500_000, amplitude=0.5),
            synth.fm(FRAMES, 500_000, amplitude=0.05),
        ],
        noise_rms=0.01,
    )
    db, bin_width = spectrum_of(iq)

    sensitive = detect(db, bin_width, threshold_db=10.0)
    picky = detect(db, bin_width, threshold_db=40.0)

    assert len(sensitive) == 2
    assert len(picky) == 1
    assert picky[0].center_hz == pytest.approx(-500_000, abs=2_000)


def test_a_signal_at_the_window_edge_is_flagged_truncated():
    """The sweeper uses this to prefer the step that saw the whole thing."""
    iq = synth.scene(FRAMES, [synth.fm(FRAMES, 1_150_000, amplitude=0.5)], 0.01)
    db, bin_width = spectrum_of(iq)
    found = detect(db, bin_width)

    assert found
    assert any(detection.truncated for detection in found)


def test_a_centred_signal_is_not_flagged_truncated():
    iq = synth.scene(FRAMES, [synth.fm(FRAMES, 200_000, amplitude=0.5)], 0.01)
    db, bin_width = spectrum_of(iq)

    assert not detect(db, bin_width)[0].truncated


def test_an_empty_spectrum_detects_nothing():
    assert detect(np.zeros(0), 100.0) == []


# -- Persistence -----------------------------------------------------------


def one(hz: float) -> Detection:
    return Detection(
        center_hz=hz,
        bandwidth_hz=200_000,
        peak_hz=hz,
        peak_dbfs=-30.0,
        floor_dbfs=-80.0,
    )


def test_a_signal_seen_once_is_not_reported():
    """A noise peak clears the threshold now and then; a station does not stop."""
    gate = Persistence(needed=2, window=3)
    assert gate.update([one(94.9e6)]) == []


def test_a_signal_seen_twice_is_reported():
    gate = Persistence(needed=2, window=3)
    gate.update([one(94.9e6)])
    confirmed = gate.update([one(94.9e6)])

    assert [d.center_hz for d in confirmed] == [94.9e6]


def test_a_signal_that_drifts_within_tolerance_still_counts():
    gate = Persistence(needed=2, window=3, tolerance_hz=25_000)
    gate.update([one(94_900_000)])

    assert gate.update([one(94_910_000)])


def test_a_signal_that_moves_too_far_is_a_different_signal():
    gate = Persistence(needed=2, window=3, tolerance_hz=25_000)
    gate.update([one(94_900_000)])

    assert gate.update([one(95_400_000)]) == []


def test_the_reported_copy_is_the_newest_sighting():
    """So strength on the card is current, not whatever it was on sweep one."""
    gate = Persistence(needed=2, window=3)
    gate.update([one(94.9e6)])
    faded = Detection(94.9e6, 200_000, 94.9e6, -55.0, -80.0)
    confirmed = gate.update([faded])

    assert confirmed[0].peak_dbfs == -55.0


def test_history_never_grows_past_the_window():
    gate = Persistence(needed=2, window=3)
    for _ in range(10):
        gate.update([one(94.9e6)])

    assert gate.sweeps == 3


def test_reset_forgets_everything():
    gate = Persistence(needed=2, window=3)
    gate.update([one(94.9e6)])
    gate.reset()

    assert gate.update([one(94.9e6)]) == []


def test_an_impossible_gate_is_rejected():
    with pytest.raises(ValueError, match="needed"):
        Persistence(needed=4, window=3)


# -- Merging overlapping steps ---------------------------------------------


def test_two_sightings_of_one_station_collapse_to_one():
    merged = merge_nearby([one(94_900_000), one(94_905_000)], tolerance_hz=25_000)
    assert len(merged) == 1


def test_distinct_stations_are_kept_apart():
    merged = merge_nearby([one(94_900_000), one(95_100_000)], tolerance_hz=25_000)
    assert len(merged) == 2


def test_the_untruncated_sighting_wins():
    """The step that saw the whole signal measured its width correctly."""
    half = Detection(94.9e6, 90_000, 94.9e6, -20.0, -80.0, truncated=True)
    whole = Detection(94.9e6, 180_000, 94.9e6, -25.0, -80.0, truncated=False)

    assert merge_nearby([half, whole])[0].bandwidth_hz == 180_000


def test_the_stronger_sighting_wins_when_both_are_whole():
    weak = Detection(94.9e6, 180_000, 94.9e6, -40.0, -80.0)
    strong = Detection(94.9e6, 180_000, 94.9e6, -20.0, -80.0)

    assert merge_nearby([weak, strong])[0].peak_dbfs == -20.0


def test_merging_returns_them_in_frequency_order():
    merged = merge_nearby([one(99e6), one(94e6), one(96e6)])
    assert [d.center_hz for d in merged] == [94e6, 96e6, 99e6]
