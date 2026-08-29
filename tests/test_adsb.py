"""ADS-B decoding, against synthetic Mode S bursts.

The generator in `tests/synth_adsb.py` builds frames from field definitions
and signs them with a real CRC-24, so these tests check the decoder's
arithmetic rather than agreeing with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.decode import adsb
from tests import synth_adsb as gen

RATE = 2_400_000.0


def feed(receiver: adsb.AdsbReceiver, iq: np.ndarray, block: int = 65_536) -> None:
    """In blocks, always: a message straddling a boundary must still decode."""
    for start in range(0, iq.size, block):
        receiver.process(iq[start : start + block])


# -- the checkword ---------------------------------------------------------


def test_generated_frame_passes_its_own_checkword():
    message = gen.squitter(0xABCDEF, gen.identity_me("UAL123"))
    assert len(message) == 14
    assert adsb.checksum(message) == 0


def test_one_flipped_bit_fails():
    message = bytearray(gen.squitter(0x484175, gen.identity_me("KLM1234")))
    message[6] ^= 0x08
    assert adsb.checksum(bytes(message)) != 0


# -- the fields ------------------------------------------------------------


def test_altitude_round_trips_in_25_foot_steps():
    for feet in (0, 1000, 5500, 35_000, 43_975):
        assert adsb.altitude_ft(gen.altitude_code(feet)) == feet


def test_altitude_without_the_q_bit_is_not_guessed_at():
    # Gillham coding, used above 50,000 ft. Reporting nothing is the honest
    # answer; a wrong altitude is worse than a missing one.
    assert adsb.altitude_ft(0x0EA0) is None
    assert adsb.altitude_ft(0) is None


def test_callsign_is_read_out_of_the_six_bit_alphabet():
    decoder = adsb.AdsbDecoder()
    decoder.feed(gen.squitter(0x4CA2D6, gen.identity_me("RYR85TA")), now=0.0)
    (aircraft,) = decoder.snapshot(now=0.0, seconds=1.0, bad=0).aircraft
    assert aircraft.callsign == "RYR85TA"
    assert aircraft.address == "4CA2D6"


def test_velocity_gives_speed_track_and_climb():
    decoder = adsb.AdsbDecoder()
    # 300 kt east, 400 kt north: a 500 kt ground speed on a 036.9 degree track.
    decoder.feed(
        gen.squitter(0x400000, gen.velocity_me(300, 400, vertical_fpm=1024)), now=0.0
    )
    (aircraft,) = decoder.snapshot(now=0.0, seconds=1.0, bad=0).aircraft
    assert aircraft.ground_speed_kt == pytest.approx(500.0, abs=2.0)
    assert aircraft.track_deg == pytest.approx(36.87, abs=0.5)
    assert aircraft.vertical_rate_fpm == pytest.approx(1024, abs=64)


def test_a_descent_reads_negative():
    decoder = adsb.AdsbDecoder()
    decoder.feed(
        gen.squitter(0x400001, gen.velocity_me(-100, -50, vertical_fpm=-1600)), now=0.0
    )
    (aircraft,) = decoder.snapshot(now=0.0, seconds=1.0, bad=0).aircraft
    assert aircraft.vertical_rate_fpm < 0
    assert 180.0 < aircraft.track_deg < 270.0


def test_airspeed_subtypes_are_left_undecoded():
    # Subtypes 3 and 4 report airspeed, not velocity over the ground. Writing
    # one into a field labelled "ground speed" would be a quiet lie.
    decoder = adsb.AdsbDecoder()
    decoder.feed(gen.squitter(0x400002, gen.velocity_me(300, 400, subtype=3)), now=0.0)
    (aircraft,) = decoder.snapshot(now=0.0, seconds=1.0, bad=0).aircraft
    assert aircraft.ground_speed_kt is None


# -- position --------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (47.6062, -122.3321),  # Seattle
        (51.4700, -0.4543),  # Heathrow
        (-33.9399, 151.1753),  # Sydney, southern and eastern
        (1.3644, 103.9915),  # Singapore, near the equator
        (64.1300, -21.9406),  # Reykjavik, a high latitude zone
    ],
)
def test_an_even_odd_pair_resolves_to_the_right_place(lat, lon):
    decoder = adsb.AdsbDecoder()
    icao = 0x3C6444
    for odd in (False, True):
        cpr = gen.cpr_encode(lat, lon, odd)
        me = gen.position_me(11, 35_000, odd, *cpr)
        decoder.feed(gen.squitter(icao, me), now=0.0 if not odd else 1.0)
    (aircraft,) = decoder.snapshot(now=1.0, seconds=2.0, bad=0).aircraft
    assert aircraft.latitude == pytest.approx(lat, abs=0.01)
    assert aircraft.longitude == pytest.approx(lon, abs=0.01)
    assert aircraft.altitude_ft == 35_000


def test_one_frame_alone_gives_no_position():
    decoder = adsb.AdsbDecoder()
    cpr = gen.cpr_encode(47.6, -122.3, odd=False)
    decoder.feed(gen.squitter(0xA0A0A0, gen.position_me(11, 10_000, False, *cpr)), 0.0)
    (aircraft,) = decoder.snapshot(now=0.0, seconds=1.0, bad=0).aircraft
    assert not aircraft.has_position
    assert aircraft.altitude_ft == 10_000


def test_an_airborne_frame_and_a_surface_frame_are_not_a_pair():
    """A surface frame divides the globe into 90 degrees of latitude where an
    airborne one uses 360, so pairing one of each applies the wrong span to
    half the arithmetic. What comes out is not obviously broken - it is an
    ordinary-looking position somewhere else entirely, which nothing later can
    catch. Seen off air on 2026-08-28: an aircraft on approach to Boeing Field
    appeared at 57 degrees east with its latitude still right. Reproduced
    here, the pair resolved to 11.9 N, 59.6 E.
    """
    icao = 0xA63879
    lat, lon = 47.7311, -122.2955
    even = gen.cpr_encode(lat, lon, odd=False)
    odd = gen.cpr_encode(lat, lon, odd=True)
    # Either way round: whichever arrives second is the one whose kind the
    # decode would have been run with.
    orders = (
        (gen.position_me(11, 1_500, False, *even), gen.position_me(7, 0, True, *odd)),
        (gen.position_me(7, 0, False, *even), gen.position_me(11, 1_500, True, *odd)),
    )
    for first, second in orders:
        decoder = adsb.AdsbDecoder()
        for when, me in enumerate((first, second)):
            decoder.feed(gen.squitter(icao, me), now=float(when))
        (aircraft,) = decoder.snapshot(now=1.0, seconds=2.0, bad=0).aircraft
        assert not aircraft.has_position


def test_a_landing_aircraft_pairs_again_once_both_halves_are_on_the_ground():
    """The guard above must not leave an aircraft positionless for good: two
    surface frames are a pair like any other."""
    decoder = adsb.AdsbDecoder()
    icao = 0xA63879
    lat, lon = 47.5300, -122.3000
    airborne = gen.cpr_encode(lat, lon, odd=False)
    decoder.feed(
        gen.squitter(icao, gen.position_me(11, 500, False, *airborne)), now=0.0
    )
    for when, odd in enumerate((False, True), start=1):
        cpr = gen.cpr_encode(lat, lon, odd)
        decoder.feed(
            gen.squitter(icao, gen.position_me(7, 0, odd, *cpr)), now=float(when)
        )
    (aircraft,) = decoder.snapshot(now=2.0, seconds=3.0, bad=0).aircraft
    assert aircraft.on_ground
    assert aircraft.has_position


def test_a_stale_partner_is_not_a_pair():
    # A jet covers 4 km in ten seconds, further than CPR's ambiguity spacing,
    # so an old frame is not a partner however well it happens to fit.
    decoder = adsb.AdsbDecoder()
    icao = 0xB0B0B0
    even = gen.cpr_encode(47.6, -122.3, odd=False)
    odd = gen.cpr_encode(47.6, -122.3, odd=True)
    decoder.feed(gen.squitter(icao, gen.position_me(11, 10_000, False, *even)), 0.0)
    decoder.feed(
        gen.squitter(icao, gen.position_me(11, 10_000, True, *odd)),
        adsb.CPR_MAX_AGE_S + 5.0,
    )
    (aircraft,) = decoder.snapshot(now=20.0, seconds=20.0, bad=0).aircraft
    assert not aircraft.has_position


def test_later_frames_update_the_track_on_their_own():
    """Once a fix exists, a single frame is enough - which is what keeps a
    track moving twice a second rather than once a pair."""
    decoder = adsb.AdsbDecoder()
    icao = 0xC0FFEE
    for odd in (False, True):
        cpr = gen.cpr_encode(47.60, -122.33, odd)
        decoder.feed(
            gen.squitter(icao, gen.position_me(11, 30_000, odd, *cpr)), 1.0 * odd
        )
    moved = (47.65, -122.28)
    cpr = gen.cpr_encode(*moved, odd=False)
    decoder.feed(gen.squitter(icao, gen.position_me(11, 30_000, False, *cpr)), 2.0)
    (aircraft,) = decoder.snapshot(now=2.0, seconds=3.0, bad=0).aircraft
    assert aircraft.latitude == pytest.approx(moved[0], abs=0.01)
    assert aircraft.longitude == pytest.approx(moved[1], abs=0.01)


def test_a_pair_that_resolves_off_the_globe_is_thrown_away():
    """The zone arithmetic runs from -90 to 270, so a mismatched pair can
    resolve to a latitude no aircraft can be at. Nothing may reach the screen
    from there."""
    decoder = adsb.AdsbDecoder()
    icao = 0xD00D00
    even = gen.cpr_encode(10.0, 20.0, odd=False)
    odd = gen.cpr_encode(45.0, 20.0, odd=True)
    decoder.feed(gen.squitter(icao, gen.position_me(11, 30_000, False, *even)), 0.0)
    decoder.feed(gen.squitter(icao, gen.position_me(11, 30_000, True, *odd)), 0.5)
    (aircraft,) = decoder.snapshot(now=0.5, seconds=1.0, bad=0).aircraft
    assert not aircraft.has_position


# -- the receiver, off synthetic air ---------------------------------------


def test_a_single_burst_decodes_end_to_end():
    receiver = adsb.AdsbReceiver(RATE)
    iq = gen.burst([gen.squitter(0x4B1234, gen.identity_me("SWR22K"))], rate=RATE)
    feed(receiver, iq)
    state = receiver.snapshot()
    assert state.messages == 1
    assert [a.callsign for a in state.aircraft] == ["SWR22K"]


@pytest.mark.parametrize(
    "start_us",
    # A full sample period at 2.4 MS/s is 0.417 us, so these walk right
    # through every arrival phase there is.
    [30.0, 30.05, 30.1, 30.153, 30.2, 30.25, 30.3, 30.35, 30.41, 30.47, 30.62],
)
def test_it_does_not_care_where_in_the_sample_grid_a_burst_lands(start_us):
    """2.4 MS/s is 2.4 samples per bit, so a burst starts at a different point
    within the sample grid every time. A decoder that only works at one phase
    would pass a single fixed test and fail on air."""
    receiver = adsb.AdsbReceiver(RATE)
    iq = gen.burst(
        [gen.squitter(0x484175, gen.identity_me("KLM56X"))],
        rate=RATE,
        start_us=start_us,
    )
    feed(receiver, iq)
    assert [a.callsign for a in receiver.snapshot().aircraft] == ["KLM56X"]


def test_several_aircraft_in_one_capture():
    frames = [
        gen.squitter(0xAAAAAA, gen.identity_me("AAL100")),
        gen.squitter(0xBBBBBB, gen.identity_me("DAL200")),
        gen.squitter(0xCCCCCC, gen.velocity_me(200, 200)),
    ]
    receiver = adsb.AdsbReceiver(RATE)
    feed(receiver, gen.burst(frames, rate=RATE, noise_rms=0.01, seed=3))
    state = receiver.snapshot()
    assert {a.address for a in state.aircraft} == {"AAAAAA", "BBBBBB", "CCCCCC"}
    assert state.messages == 3


def test_a_burst_split_across_blocks_still_decodes():
    """The interpolation phase and the framing buffer both carry across
    blocks. Without either, roughly one message in 250 would vanish at
    2.4 MS/s - a loss nothing else in the app would report."""
    message = gen.squitter(0x3C4B26, gen.identity_me("DLH400"))
    iq = gen.burst([message], rate=RATE, start_us=30.0)
    # Cut inside the burst: 40 us in is a third of the way through the data.
    cut = int(70.0 * RATE / 1e6)
    receiver = adsb.AdsbReceiver(RATE)
    receiver.process(iq[:cut])
    receiver.process(iq[cut:])
    assert [a.callsign for a in receiver.snapshot().aircraft] == ["DLH400"]


def test_pure_noise_produces_no_aircraft():
    rng = np.random.default_rng(11)
    noise = (rng.normal(0, 0.02, 400_000) + 1j * rng.normal(0, 0.02, 400_000)).astype(
        np.complex64
    )
    receiver = adsb.AdsbReceiver(RATE)
    feed(receiver, noise)
    state = receiver.snapshot()
    assert state.aircraft == ()
    assert state.messages == 0


def test_a_corrupted_burst_is_counted_bad_rather_than_believed():
    """A well-formed burst carrying a frame whose checkword does not match.
    The decoder could repair a single bit error and deliberately does not: a
    mis-corrected frame is a plausible aircraft in the wrong place."""
    message = bytearray(gen.squitter(0x111111, gen.identity_me("BAW99")))
    message[6] ^= 0x08
    receiver = adsb.AdsbReceiver(RATE)
    feed(receiver, gen.burst([bytes(message)], rate=RATE))
    state = receiver.snapshot()
    assert state.messages == 0
    assert state.aircraft == ()
    assert state.bad >= 1


def test_a_frequency_offset_does_not_matter():
    """Mode S is on-off keying, so the decoder reads magnitude and the phase
    carries nothing. A tuning error large enough to ruin any coherent scheme
    is invisible here."""
    receiver = adsb.AdsbReceiver(RATE)
    iq = gen.burst(
        [gen.squitter(0x222222, gen.identity_me("AFR11"))],
        rate=RATE,
        offset_hz=150_000.0,
    )
    feed(receiver, iq)
    assert [a.callsign for a in receiver.snapshot().aircraft] == ["AFR11"]


def test_a_full_track_builds_over_several_messages():
    icao = 0x71BE45
    lat, lon = 47.4502, -122.3088
    frames = [gen.squitter(icao, gen.identity_me("ASA455"))]
    for odd in (False, True):
        cpr = gen.cpr_encode(lat, lon, odd)
        frames.append(gen.squitter(icao, gen.position_me(11, 12_500, odd, *cpr)))
    frames.append(gen.squitter(icao, gen.velocity_me(-120, 90, vertical_fpm=-1088)))

    receiver = adsb.AdsbReceiver(RATE)
    feed(receiver, gen.burst(frames, rate=RATE, noise_rms=0.008, seed=7))
    (aircraft,) = receiver.snapshot().aircraft
    assert aircraft.callsign == "ASA455"
    assert aircraft.altitude_ft == 12_500
    assert aircraft.latitude == pytest.approx(lat, abs=0.01)
    assert aircraft.longitude == pytest.approx(lon, abs=0.01)
    assert aircraft.ground_speed_kt == pytest.approx(150.0, abs=3.0)
    assert aircraft.vertical_rate_fpm < 0
    assert aircraft.label == "ASA455"


def test_weak_bursts_still_decode():
    """A message 20 dB down on the strong case is still well clear of the
    detector's threshold - which is set against the noise floor, not against
    any absolute level."""
    receiver = adsb.AdsbReceiver(RATE)
    iq = gen.burst(
        [gen.squitter(0x333333, gen.identity_me("QFA12"))],
        rate=RATE,
        amplitude=0.05,
        noise_rms=0.004,
        seed=5,
    )
    feed(receiver, iq)
    assert [a.callsign for a in receiver.snapshot().aircraft] == ["QFA12"]


# -- housekeeping ----------------------------------------------------------


def test_aircraft_age_out_of_the_list():
    decoder = adsb.AdsbDecoder()
    decoder.feed(gen.squitter(0x999999, gen.identity_me("OLD1")), now=0.0)
    assert decoder.snapshot(now=10.0, seconds=10.0, bad=0).aircraft
    assert not decoder.snapshot(
        now=adsb.AIRCRAFT_TIMEOUT_S + 1.0, seconds=100.0, bad=0
    ).aircraft
    # Still tracked, so a returning aircraft keeps its callsign.
    assert 0x999999 in decoder.tracks
    decoder.snapshot(now=adsb.AIRCRAFT_FORGET_S + 1.0, seconds=400.0, bad=0)
    assert 0x999999 not in decoder.tracks


def test_the_clock_comes_from_the_sample_count():
    receiver = adsb.AdsbReceiver(RATE)
    receiver.process(np.zeros(int(RATE), dtype=np.complex64))
    assert receiver.snapshot().seconds == pytest.approx(1.0, abs=1e-6)


def test_reset_forgets_everything():
    receiver = adsb.AdsbReceiver(RATE)
    message = gen.squitter(0x123456, gen.identity_me("RST1"))
    feed(receiver, gen.burst([message], rate=RATE))
    assert receiver.snapshot().aircraft
    receiver.reset()
    state = receiver.snapshot()
    assert state.aircraft == () and state.messages == 0 and state.seconds == 0.0


def test_too_slow_a_sample_rate_is_refused():
    with pytest.raises(ValueError, match="at least"):
        adsb.AdsbReceiver(1_024_000)


def test_message_rate_is_reported_per_minute():
    state = adsb.AdsbState(messages=30, seconds=60.0)
    assert state.rate_per_minute == pytest.approx(30.0)
    assert adsb.AdsbState().rate_per_minute == 0.0


def test_the_decoder_costs_a_small_fraction_of_a_core():
    """One second of 2.4 MS/s air, timed. The budget matters because this runs
    on the DSP thread alongside everything else."""
    import time

    receiver = adsb.AdsbReceiver(RATE)
    rng = np.random.default_rng(2)
    n = int(RATE)
    iq = (rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)).astype(np.complex64)
    start = time.perf_counter()
    feed(receiver, iq)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"{elapsed:.3f} s per second of radio"


def test_surface_position_reports_ground_speed_not_altitude():
    decoder = adsb.AdsbDecoder()
    me = bytearray(7)
    me[0] = (7 << 3) | 0x02  # type code 7, movement high bits
    me[1] = 0x60  # movement low bits, ground track invalid
    decoder.feed(gen.squitter(0x555555, bytes(me)), now=0.0)
    (aircraft,) = decoder.snapshot(now=0.0, seconds=1.0, bad=0).aircraft
    assert aircraft.on_ground
    assert aircraft.altitude_ft is None
    assert aircraft.ground_speed_kt is not None


def test_nl_matches_the_published_boundaries():
    assert adsb._nl(0.0) == 59
    assert adsb._nl(87.5) == 1
    assert adsb._nl(-87.5) == 1
    # The published boundaries are the first latitude of each new zone count.
    assert adsb._nl(10.46) == 59
    assert adsb._nl(10.47047130) == 58
    assert adsb._nl(86.50) == 3
    assert adsb._nl(86.53536998) == 2
