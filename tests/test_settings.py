"""Tests for persisted settings, bookmarks and PPM calibration.

The settings tests are mostly about robustness rather than features: the
requirement is that no state of the settings file can stop the radio opening,
because a beginner has no way to recover from that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bettersdr.core.bookmarks import Bookmark, BookmarkStore, from_signal
from bettersdr.core.calibrate import calibrate, measure_offset_hz, ppm_from_offset
from bettersdr.core.settings import DEFAULTS, Settings

# -- settings --------------------------------------------------------------


def test_missing_file_gives_defaults(tmp_path):
    settings = Settings(tmp_path / "nothing.json")
    assert settings["level"] == DEFAULTS["level"]
    assert settings["fft_size"] == 4096


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    Settings(path).update(level="expert", frequency_hz=162_550_000).save()
    assert Settings(path)["level"] == "expert"
    assert Settings(path)["frequency_hz"] == 162_550_000


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert Settings(path)["level"] == DEFAULTS["level"]


def test_unknown_keys_are_ignored(tmp_path):
    """A file written by a later version must not smuggle keys back in."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"level": "simple", "warp_drive": True}), encoding="utf-8")
    settings = Settings(path)
    assert settings["level"] == "simple"
    assert "warp_drive" not in settings.as_dict()


def test_bias_tee_never_persists_as_on(tmp_path):
    """4.5 V on the antenna port is not a preference to restore silently."""
    assert DEFAULTS["bias_tee"] is False
    path = tmp_path / "settings.json"
    Settings(path).update(bias_tee=True).save()
    # It round-trips if explicitly saved, but it is never the default, so a
    # fresh profile or a lost file can never surprise anyone.
    assert Settings(tmp_path / "fresh.json")["bias_tee"] is False


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "settings.json"
    Settings(path).update(volume=0.9).save()
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]


# -- bookmarks -------------------------------------------------------------


def _entry(name="KUOW", hz=94_900_000, group="FM") -> Bookmark:
    return Bookmark(name=name, frequency_hz=hz, mode="wfm", bandwidth_hz=200_000.0,
                    group=group)


def test_bookmarks_round_trip(tmp_path):
    store = BookmarkStore.open(tmp_path / "b.json")
    store.add(_entry())
    store.add(_entry("NOAA", 162_550_000, "Weather"))
    store.save()

    reloaded = BookmarkStore.open(tmp_path / "b.json")
    assert len(reloaded) == 2
    assert reloaded.groups == ["FM", "Weather"]
    assert reloaded.entries[0].mode == "wfm"


def test_adding_the_same_frequency_twice_replaces_rather_than_duplicates():
    store = BookmarkStore()
    store.add(_entry("First"))
    store.add(_entry("Second", 94_901_000))
    assert len(store) == 1
    assert store.entries[0].name == "Second"


def test_find_locates_a_nearby_entry():
    store = BookmarkStore()
    store.add(_entry())
    assert store.find(94_902_000) is not None
    assert store.find(95_500_000) is None


def test_entries_are_sorted_by_group_then_frequency():
    store = BookmarkStore()
    store.add(_entry("High", 108_000_000, "FM"))
    store.add(_entry("Low", 88_100_000, "FM"))
    store.add(_entry("Air", 121_500_000, "Airband"))
    assert [e.name for e in store] == ["Air", "Low", "High"]


def test_csv_round_trip():
    store = BookmarkStore()
    store.add(_entry())
    store.add(_entry("NOAA", 162_550_000, "Weather"))

    restored = BookmarkStore()
    assert restored.from_csv(store.to_csv(), merge=False) == 2
    assert [e.name for e in restored] == [e.name for e in store]
    assert restored.entries[0].bandwidth_hz == 200_000.0


def test_csv_import_keeps_usable_rows_and_skips_broken_ones():
    text = (
        "name,frequency_hz,mode,bandwidth_hz,group,notes\n"
        "Good,94900000,wfm,200000,FM,\n"
        "No frequency,,nfm,12500,FM,\n"
        "Sparse,162550000,,,,\n"
    )
    store = BookmarkStore()
    assert store.from_csv(text) == 2
    sparse = store.find(162_550_000)
    assert sparse is not None
    assert sparse.mode == "wfm"
    assert sparse.group == "General"


def test_label_reads_naturally():
    assert _entry().label == "KUOW - 94.9 MHz"
    assert Bookmark("KIRO", 710_000).label == "KIRO - 710.0 kHz"


def test_from_signal_keeps_the_classifier_decision():
    class FakeSignal:
        frequency_hz = 162_550_000.0
        label = "Weather radio"
        mode = "nfm"
        demod_bandwidth_hz = 12_500.0
        description = "Continuous forecasts"

    entry = from_signal(FakeSignal())
    assert entry.mode == "nfm"
    assert entry.bandwidth_hz == 12_500.0
    assert entry.frequency_hz == 162_550_000


# -- calibration -----------------------------------------------------------

RATE = 240_000.0


def _capture(offset_hz: float, n: int = 131_072, snr: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n) / RATE
    amplitude = 0.3
    noise_rms = amplitude / (10 ** (snr / 20))
    noise = rng.normal(0, noise_rms, n) + 1j * rng.normal(0, noise_rms, n)
    return (amplitude * np.exp(2j * np.pi * offset_hz * t) + noise).astype(np.complex64)


@pytest.mark.parametrize("planted", [-3_000.0, -420.0, 0.0, 137.0, 5_000.0])
def test_offset_is_recovered_to_a_fraction_of_a_bin(planted):
    """Interpolation is the point: bins are 14.6 Hz wide here."""
    measured, peak_db = measure_offset_hz(_capture(planted), RATE)
    assert measured == pytest.approx(planted, abs=2.0)
    assert peak_db > 20.0


def test_a_carrier_far_below_the_noise_is_still_measured_exactly():
    """A carrier 20 dB under the noise in the time domain still calibrates.

    The transform has around 39 dB of processing gain, so the peak stands
    well clear even though nothing about the capture sounds like a signal.
    That is why the trust threshold is set against bare noise rather than
    against a signal-to-noise ratio measured on the samples.
    """
    measured, peak_db = measure_offset_hz(_capture(500.0, snr=-20.0), RATE)
    assert measured == pytest.approx(500.0, abs=2.0)
    assert peak_db > 15.0


def test_ppm_sign_removes_the_error_rather_than_doubling_it():
    """A carrier above centre means the crystal is slow, so ppm goes down."""
    assert ppm_from_offset(+500.0, 100e6) < 0
    assert ppm_from_offset(-500.0, 100e6) > 0
    assert ppm_from_offset(+100.0, 100e6) == pytest.approx(-1.0)


def test_ppm_accumulates_onto_the_correction_already_in_force():
    assert ppm_from_offset(100.0, 100e6, current_ppm=5) == pytest.approx(4.0)


def test_calibration_refuses_a_reference_whose_frequency_wanders():
    """The finding that made this feature honest rather than misleading.

    Measured on air: six readings against a wideband FM station spread by
    1814 Hz - 19 ppm of random number - while the same six against a steady
    weather-radio carrier spread by 11.5 Hz. A confident ppm figure derived
    from the first is worse than no figure at all.
    """
    rng = np.random.default_rng(3)
    n = 131_072
    t = np.arange(n) / RATE
    # A carrier swinging back and forth the way a modulated one does.
    swing = 4_000.0 * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(500.0 + swing) / RATE
    wobbly = (0.3 * np.exp(1j * phase) + rng.normal(0, 0.01, n)).astype(np.complex64)

    result = calibrate(wobbly, RATE, 100e6)
    assert not result.steady
    assert result.spread_hz > 100.0
    assert "steady carrier" in result.summary


def test_a_steady_carrier_passes_the_agreement_check():
    result = calibrate(_capture(500.0), RATE, 100e6)
    assert result.steady
    assert result.spread_hz < 5.0


def test_calibration_refuses_to_measure_bare_noise():
    """Pure noise peaks 4-5 dB above its own median, so the tallest bin is
    not a carrier and must not be reported as a ppm figure."""
    rng = np.random.default_rng(7)
    noise = (rng.normal(0, 0.05, 131_072) + 1j * rng.normal(0, 0.05, 131_072)).astype(
        np.complex64
    )
    result = calibrate(noise, RATE, 100e6)
    assert not result.trustworthy
    # Bare noise fails both checks; the weakness message is the one shown.
    assert not result.steady
    assert "too weak" in result.summary


def test_calibration_of_a_strong_carrier_is_trustworthy():
    result = calibrate(_capture(940.0), RATE, 94.9e6)
    assert result.trustworthy
    assert result.ppm == pytest.approx(-10, abs=1)
    assert "ppm" in result.summary


def test_short_capture_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="at least"):
        measure_offset_hz(np.zeros(1_024, dtype=np.complex64), RATE)
