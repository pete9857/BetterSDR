"""Tests for the band plan.

The band plan drives what the app tells the user a signal is, and - since
tuning into a new band switches the demodulator - what it sounds like. A wrong
entry here is a wrong answer everywhere, so these check the data itself as
well as the lookup logic.
"""

from __future__ import annotations

import pytest

from bettersdr.core.device import DEFAULT_SAMPLE_RATE
from bettersdr.core.frontend import (
    SUPPORTED_SAMPLE_RATES,
    safe_sample_rate,
)
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


# -- window width ----------------------------------------------------------


def test_am_broadcast_asks_for_a_narrow_window():
    """The band that forced `sample_rate_hz` to exist in the first place."""
    band = bandplan.find(710_000)
    assert band is not None and band.name == "AM Radio"
    assert band.sample_rate_hz is not None
    assert band.sample_rate_hz < 2_400_000


def test_declared_rates_are_ones_the_hardware_and_the_demodulators_accept():
    for band in bandplan.load():
        if band.sample_rate_hz is None:
            continue
        assert band.sample_rate_hz in SUPPORTED_SAMPLE_RATES, band.name


def test_no_band_would_be_swept_through_zero_hz():
    """Every band, at its lowest edge, gets a window clear of the dial's end.

    A sweep that reaches 0 Hz reports the upconverter's own oscillator as the
    strongest signal in the band, which is the most confidently wrong thing
    the scanner could say.
    """
    for band in bandplan.load():
        preferred = band.sample_rate_hz or DEFAULT_SAMPLE_RATE
        rate = safe_sample_rate(band.start_hz, preferred_hz=preferred)
        assert band.start_hz - rate / 2 > 0, band.name


# -- named channels --------------------------------------------------------


def test_every_channel_is_well_formed():
    for band in bandplan.load():
        for channel in band.channels:
            where = f"{band.name} {channel.name}"
            assert channel.name, where
            assert channel.use, where
            assert channel.official, where
            assert band.contains(channel.frequency_hz), where


def test_every_channel_sits_on_its_own_band_raster():
    """The two ways of saying where a channel is must agree.

    `snap` moves a click onto a legal channel centre and `channels` names the
    one it landed on. If a channel were half a raster step off, clicking it
    would tune somewhere the app then refused to name - the same class of
    fault as the NOAA station that came out as 162.537 MHz.
    """
    for band in bandplan.load():
        if not band.raster_hz:
            continue
        for channel in band.channels:
            assert band.snap(channel.frequency_hz) == pytest.approx(
                channel.frequency_hz
            ), f"{band.name} {channel.name}"


def test_no_band_lists_one_frequency_twice():
    for band in bandplan.load():
        seen = [channel.frequency_hz for channel in band.channels]
        assert len(seen) == len(set(seen)), band.name


def test_channels_come_back_in_frequency_order():
    for band in bandplan.load():
        listed = [channel.frequency_hz for channel in band.channels]
        assert listed == sorted(listed), band.name


@pytest.mark.parametrize(
    ("hz", "expected"),
    [
        (156_800_000, "Channel 16"),  # the one everybody knows
        (156_650_000, "Channel 13"),  # bridge to bridge
        (157_100_000, "Channel 22A"),  # the Coast Guard's working channel
        (161_975_000, "AIS 1"),  # ship positions, as data
        (162_550_000, "WX1"),  # weather radio numbering is not in order
        (162_400_000, "WX2"),
        (27_185_000, "Channel 19"),  # the truckers' channel
        (462_675_000, "Channel 20"),  # GMRS travel and emergency
        (467_712_500, "Channel 14"),  # the low-power family radio channels
        (121_500_000, "Guard"),  # not everything is called a channel
        (146_520_000, "National calling"),
    ],
)
def test_the_channel_at_a_known_frequency(hz: int, expected: str):
    band = bandplan.find(hz)
    assert band is not None
    channel = band.channel(hz)
    assert channel is not None
    assert channel.name == expected


def test_a_channel_is_named_from_anywhere_inside_it():
    """A dial a few kilohertz off is still that channel."""
    marine = bandplan.find(156_800_000)
    assert marine is not None
    for offset in (-12_000, -3_000, 0, 3_000, 12_000):
        channel = marine.channel(156_800_000 + offset)
        assert channel is not None and channel.name == "Channel 16"


def test_a_channel_claims_half_a_raster_and_no_more():
    """Marine channels are shoulder to shoulder, so every dial position is on
    one of them - but it must be the nearer one, and the moment the dial
    crosses the halfway point the name has to change with it."""
    marine = bandplan.find(156_800_000)
    assert marine is not None
    assert marine.channel(156_812_000).name == "Channel 16"
    assert marine.channel(156_813_000).name == "Channel 76"


def test_a_frequency_with_no_named_channel_near_it_is_not_named():
    """Most of the airband is ordinary tower and approach frequencies.

    Only the handful everybody knows are named, so the rest must come back
    with nothing rather than borrowing the name of the nearest one.
    """
    airband = bandplan.find(118_300_000)
    assert airband is not None
    assert airband.channels
    assert airband.channel(118_300_000) is None
    assert airband.channel(121_500_000).name == "Guard"


def test_the_cb_channels_that_are_out_of_order_stay_out_of_order():
    """Channel 23 sits above 24 and 25, and has since 1977.

    It is exactly the kind of thing a formula gets wrong, which is why the
    forty channels are listed rather than counted off the raster.
    """
    cb = bandplan.find(27_185_000)
    assert cb is not None
    assert cb.channel(27_235_000).name == "Channel 24"
    assert cb.channel(27_245_000).name == "Channel 25"
    assert cb.channel(27_255_000).name == "Channel 23"


def test_a_band_whose_frequencies_are_their_own_names_has_no_channels():
    """"94.9" is what anybody would say about an FM station."""
    fm = bandplan.find(94_900_000)
    assert fm is not None
    assert fm.channels == ()
    assert fm.channel(94_900_000) is None


# -- what the rest of the dial is licensed for -----------------------------


def test_every_allocation_is_well_formed():
    for allocation in bandplan.allocations():
        assert allocation.start_hz < allocation.end_hz, allocation.name
        assert allocation.name, allocation.start_hz
        assert allocation.use, allocation.name


def test_allocations_do_not_overlap_each_other():
    listed = bandplan.allocations()
    for earlier, later in zip(listed, listed[1:], strict=False):
        assert earlier.end_hz <= later.start_hz, f"{earlier.name} / {later.name}"


def test_no_allocation_covers_ground_a_band_already_covers():
    """The two lists answer the same question and must not both answer it."""
    for allocation in bandplan.allocations():
        for band in bandplan.load():
            assert not (
                allocation.start_hz < band.end_hz
                and band.start_hz < allocation.end_hz
            ), f"{allocation.name} overlaps {band.name}"


def test_every_gap_between_the_bands_says_who_owns_it():
    """The whole dial answers "what is this?", not only the listenable part.

    Anywhere the receiver can tune and the band plan has nothing to offer,
    Standard level is meant to say what the space is licensed for. A gap
    nobody wrote an entry for is the one place that silently falls back to
    "nothing is normally broadcast here", which for the 700 MHz mobile phone
    band would simply be untrue.
    """
    from bettersdr.core.device import MAX_TUNE_HZ, MIN_TUNE_HZ

    covered: list[tuple[int, int]] = []
    for band in sorted(bandplan.load(), key=lambda b: b.start_hz):
        if covered and band.start_hz <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], band.end_hz))
        else:
            covered.append((band.start_hz, band.end_hz))

    edge = MIN_TUNE_HZ
    gaps: list[tuple[int, int]] = []
    for start, end in covered:
        if start > edge:
            gaps.append((edge, start))
        edge = max(edge, end)
    if edge < MAX_TUNE_HZ:
        gaps.append((edge, MAX_TUNE_HZ))

    for start, end in gaps:
        for hz in (start + 1, (start + end) // 2, end - 1):
            assert bandplan.official(hz) is not None, (
                f"{hz / 1e6:.3f} MHz is in no band and no allocation"
            )


def test_a_frequency_inside_a_band_is_not_answered_twice():
    """A band describes itself; the allocation list speaks only for the gaps."""
    assert bandplan.official(94_900_000) is None
    assert bandplan.official(162_550_000) is None


def test_the_allocation_list_reaches_the_ends_of_the_dial():
    from bettersdr.core.device import MAX_TUNE_HZ, MIN_TUNE_HZ

    assert bandplan.official(MIN_TUNE_HZ) is not None
    assert bandplan.official(MAX_TUNE_HZ) is not None
