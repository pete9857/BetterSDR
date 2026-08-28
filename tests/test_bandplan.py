"""Tests for the band plan.

The band plan drives what the app tells the user a signal is, and - since
tuning into a new band switches the demodulator - what it sounds like. A wrong
entry here is a wrong answer everywhere, so these check the data itself as
well as the lookup logic.
"""

from __future__ import annotations

import pytest

from bettersdr.dsp import demod
from bettersdr.scan import bandplan


def test_every_band_is_well_formed():
    for band in bandplan.load():
        assert band.start_hz < band.end_hz, band.name
        assert band.mode in demod.MODES, f"{band.name} has unknown mode {band.mode}"
        assert band.bandwidth_hz > 0, band.name
        assert band.description, band.name
        assert band.colour.startswith("#"), band.name


def test_bands_are_within_what_the_dongle_can_tune():
    from bettersdr.core.device import MAX_TUNE_HZ, MIN_TUNE_HZ

    for band in bandplan.load():
        assert band.start_hz >= MIN_TUNE_HZ, band.name
        assert band.end_hz <= MAX_TUNE_HZ, band.name


def test_bands_come_back_in_frequency_order():
    bands = bandplan.load()
    assert [b.start_hz for b in bands] == sorted(b.start_hz for b in bands)


@pytest.mark.parametrize(
    ("hz", "expected"),
    [
        (1_000_000, "AM Radio"),
        (98_500_000, "FM Radio"),
        (118_300_000, "Aircraft"),
        (146_520_000, "2 m Amateur"),
        (1_090_000_000, "Aircraft Tracking"),
    ],
)
def test_find_identifies_the_obvious_bands(hz: int, expected: str):
    band = bandplan.find(hz)
    assert band is not None
    assert band.name == expected


def test_the_narrower_allocation_wins_when_they_overlap():
    """Weather Radio sits inside the wider public service range.

    "Weather Radio" is a useful answer; "Public Service" is technically true
    and useless, so the more specific band has to win.
    """
    band = bandplan.find(162_475_000)
    assert band is not None
    assert band.name == "Weather Radio"


def test_unallocated_frequency_returns_nothing():
    # Between the top of the 23 cm amateur band and the tuning ceiling.
    assert bandplan.find(1_400_000_000) is None


def test_snap_lands_on_the_us_fm_raster():
    """US FM sits on odd tenths, so a rough click must land on a station."""
    fm = bandplan.find(94_900_000)
    assert fm is not None
    assert fm.snap(94_873_000) == pytest.approx(94_900_000)
    assert fm.snap(94_960_000) == pytest.approx(94_900_000)
    assert fm.snap(95_020_000) == pytest.approx(95_100_000)


@pytest.mark.parametrize(
    ("band_name", "channel_hz"),
    [
        ("AM Radio", 1_010_000),  # a US AM channel; the band starts at 530
        ("FM Radio", 94_900_000),  # odd tenths, starting at 88.1
        ("CB Radio", 27_185_000),  # channel 19, at the band edge
        ("Aircraft", 118_300_000),  # 25 kHz from 118.000
        ("Marine VHF", 156_800_000),  # channel 16, counted from 156.025
        ("Weather Radio", 162_550_000),  # the channels start at the band edge
        ("Walkie-Talkies", 462_562_500),  # FRS channel 1
    ],
)
def test_a_real_channel_snaps_to_itself(band_name: str, channel_hz: int):
    """Where a band's channels start is per-band data, not a formula.

    Deriving it as "half a raster in from the band edge" is right for FM
    broadcast and wrong for most of the rest: it put the NOAA weather station
    on 162.550 MHz on screen as 162.537 MHz. Each raster is checked here
    against a frequency that genuinely exists.
    """
    band = next(b for b in bandplan.load() if b.name == band_name)
    assert band.raster_hz, f"{band_name} has no raster to check"
    assert band.snap(channel_hz) == pytest.approx(channel_hz)
    # And a measurement a little off it lands back on the channel.
    assert band.snap(channel_hz + band.raster_hz * 0.2) == pytest.approx(channel_hz)
    assert band.snap(channel_hz - band.raster_hz * 0.2) == pytest.approx(channel_hz)


def test_every_snapped_channel_stays_inside_its_band():
    for band in bandplan.load():
        if not band.raster_hz:
            continue
        for hz in (band.start_hz, band.center_hz, band.end_hz):
            snapped = band.snap(hz)
            assert band.start_hz - band.raster_hz <= snapped <= (
                band.end_hz + band.raster_hz
            ), band.name


def test_snap_is_a_no_op_without_a_raster():
    shortwave = bandplan.find(9_500_000)
    assert shortwave is not None
    assert shortwave.raster_hz is None
    assert shortwave.snap(9_512_345) == 9_512_345


def test_overlapping_covers_a_span_for_the_ribbon():
    names = [band.name for band in bandplan.overlapping(156_000_000, 163_000_000)]
    assert "Marine VHF" in names
    assert "Weather Radio" in names
    assert "FM Radio" not in names


def test_unknown_region_says_so():
    with pytest.raises(FileNotFoundError, match="no band plan"):
        bandplan.load("atlantis")
