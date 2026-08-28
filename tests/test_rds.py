"""Tests for the RDS decoder, against a synthetic station.

The transmitter below is the specification read forwards - group, checkword,
differential encoding, biphase, 57 kHz - so a round trip through it exercises
every step of the receiver reading it backwards. Where a test plants a phase
or a timing error, that is a fault the real air produces and the receiver has
to survive on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.decode.rds import (
    RdsReceiver,
    block_offset,
    callsign,
    checkword,
    encode_block,
)

IF_RATE = 240_000.0
SYMBOL_RATE = 1187.5


def _group_bits(pi: int, b: int, c: int, d: int) -> list[int]:
    """One four-block group as 104 raw bits, before differential encoding."""
    words = (
        encode_block(pi, "A"),
        encode_block(b, "B"),
        encode_block(c, "C"),
        encode_block(d, "D"),
    )
    return [(word >> i) & 1 for word in words for i in range(25, -1, -1)]


def _program_service_groups(pi: int, name: str, pty: int = 10) -> list[int]:
    """The four 0A groups that spell out an eight-character station name."""
    padded = f"{name:<8.8s}"
    bits: list[int] = []
    for segment in range(4):
        b = (0 << 12) | (0 << 11) | (1 << 10) | (pty << 5) | segment
        pair = padded[segment * 2 : segment * 2 + 2]
        bits += _group_bits(pi, b, pi, (ord(pair[0]) << 8) | ord(pair[1]))
    return bits


def _radio_text_groups(pi: int, text: str, pty: int = 10) -> list[int]:
    """2A groups carrying up to 64 characters, four at a time."""
    padded = f"{text:<64.64s}"
    bits: list[int] = []
    for segment in range(16):
        chunk = padded[segment * 4 : segment * 4 + 4]
        b = (2 << 12) | (0 << 11) | (1 << 10) | (pty << 5) | segment
        bits += _group_bits(
            pi,
            b,
            (ord(chunk[0]) << 8) | ord(chunk[1]),
            (ord(chunk[2]) << 8) | ord(chunk[3]),
        )
    return bits


def mpx(
    bits: list[int],
    repeats: int = 6,
    amplitude: float = 0.05,
    quadrature: bool = False,
    noise: float = 0.0,
    rate: float = IF_RATE,
    seed: int = 7,
) -> np.ndarray:
    """A multiplex signal carrying `bits`, the way a transmitter builds one."""
    stream = bits * repeats
    # Differential encoding: the wire carries changes, not values, which is
    # what makes the receiver immune to the 180 degree ambiguity of BPSK.
    level = 0
    symbols = []
    for bit in stream:
        level ^= bit
        symbols.append(1.0 if level else -1.0)
    symbol = np.asarray(symbols, dtype=np.float64)

    sps = rate / SYMBOL_RATE
    count = int(len(symbols) * sps)
    position = np.arange(count) / sps
    index = position.astype(int)
    # Biphase: each symbol is sent as itself then its opposite, so every
    # symbol has a transition in the middle and the waveform carries no DC.
    half = np.where(position - index < 0.5, 1.0, -1.0)
    baseband = symbol[index] * half

    t = np.arange(count) / rate
    angle = 2.0 * np.pi * 57_000.0 * t
    carrier = np.sin(angle) if quadrature else np.cos(angle)
    out = amplitude * baseband * carrier
    if noise:
        out = out + np.random.default_rng(seed).normal(0.0, noise, count)
    return out.astype(np.float32)


def _receive(signal: np.ndarray, block: int = 3_277) -> RdsReceiver:
    receiver = RdsReceiver(IF_RATE)
    for start in range(0, signal.size, block):
        receiver.process(signal[start : start + block])
    return receiver


# -- the arithmetic --------------------------------------------------------


def test_checkword_round_trips_through_every_offset():
    for offset in ("A", "B", "C", "C'", "D"):
        for info in (0x0000, 0x1234, 0xABCD, 0xFFFF):
            assert block_offset(encode_block(info, offset)) == offset


def test_a_corrupt_block_is_reported_as_corrupt():
    """A single flipped bit must not pass. Half of what RDS does is refusing."""
    good = encode_block(0x4D26, "B")
    for bit in range(26):
        assert block_offset(good ^ (1 << bit)) is None


def test_known_callsigns_come_out_of_their_pi_codes():
    assert callsign(0x1000) == "KAAA"
    assert callsign(0x54A8) == "WAAA"
    assert callsign(0x994F) == "WZZZ"
    # Outside the two arithmetic ranges there is a lookup table we do not
    # carry, and no answer beats a wrong one.
    assert callsign(0x0FFF) is None
    assert callsign(0x9950) is None


def test_checkword_matches_the_generator_polynomial():
    """Divide by hand once, so the shift register cannot drift undetected."""
    info = 0x8000
    remainder = info << 10
    for shift in range(25, 9, -1):
        if remainder & (1 << shift):
            remainder ^= 0x5B9 << (shift - 10)
    assert checkword(info) == remainder


# -- the receiver ----------------------------------------------------------


def test_station_name_and_identifier_come_back_off_the_air():
    receiver = _receive(mpx(_program_service_groups(0x54A8 + 8_845, "BetterFM")))
    state = receiver.snapshot()

    assert state.synced
    assert state.pi == 0x54A8 + 8_845
    assert state.station.strip() == "BetterFM"
    assert state.pty_name == "Country"
    assert state.traffic_program
    assert state.blocks_bad == 0
    assert state.name == "BetterFM"


def test_the_callsign_is_derived_from_the_identifier():
    receiver = _receive(mpx(_program_service_groups(0x54A8, "  WAAA  ")))
    assert receiver.snapshot().callsign == "WAAA"


def test_radio_text_arrives_whole():
    message = "Now playing: something with a long title"
    receiver = _receive(mpx(_radio_text_groups(0x1234, message), repeats=2))
    assert receiver.snapshot().text == message


def test_a_quadrature_subcarrier_decodes_just_the_same():
    """The standard permits either phase, so the receiver must find the data.

    It has no pilot to refer to, so it works the angle out of the symbols
    themselves - and a decoder that assumed in-phase would silently produce
    nothing on half the stations it met.
    """
    receiver = _receive(
        mpx(_program_service_groups(0x2000, "QUADFM"), quadrature=True)
    )
    assert receiver.snapshot().station.strip() == "QUADFM"


def test_a_station_is_still_read_under_noise():
    signal = mpx(_program_service_groups(0x2468, "NOISYFM"), repeats=10, noise=0.03)
    state = _receive(signal).snapshot()
    assert state.station.strip() == "NOISYFM"
    assert state.quality > 0.8


def test_block_size_does_not_change_what_is_decoded():
    """The stream is chopped by USB transfers, not by anything RDS cares about.

    Every stage carrying history has to prove it across the boundary; a
    subcarrier oscillator or a symbol cursor that restarts each block would
    decode the first group of every block and nothing else.
    """
    signal = mpx(_program_service_groups(0x3000, "BLOCKFM"))
    for block in (512, 3_277, 20_000):
        assert _receive(signal, block=block).snapshot().station.strip() == "BLOCKFM"


def test_sync_is_found_from_an_arbitrary_starting_point():
    """A receiver switched on mid-group has to hunt for the block boundary."""
    signal = mpx(_program_service_groups(0x4000, "MIDWAYFM"))
    state = _receive(signal[7_919:]).snapshot()
    assert state.synced
    assert state.station.strip() == "MIDWAYFM"


def test_silence_decodes_to_nothing_rather_than_to_something():
    """The expensive failure mode is a receiver that invents a station."""
    quiet = np.random.default_rng(3).normal(0.0, 0.02, 240_000).astype(np.float32)
    state = _receive(quiet).snapshot()
    assert state.station.strip() == ""
    assert state.pi is None
    assert state.groups == 0


def test_reset_forgets_the_previous_station():
    receiver = _receive(mpx(_program_service_groups(0x5000, "FIRSTFM")))
    assert receiver.snapshot().station.strip() == "FIRSTFM"
    receiver.reset()
    state = receiver.snapshot()
    assert state.pi is None
    assert state.station.strip() == ""
    assert not state.synced


def test_a_narrow_if_is_refused_rather_than_decoded_badly():
    with pytest.raises(ValueError):
        RdsReceiver(96_000.0)


# -- through the real demodulator ------------------------------------------


def test_a_station_is_read_through_the_fm_demodulator():
    """The path the app actually uses: modulated RF in, station name out.

    Everything above tests the receiver against a multiplex handed to it
    directly. This one puts that multiplex where it really lives - inside the
    frequency modulation of a broadcast carrier, under an audio tone twenty
    times its size - and takes it back out through the same demodulator that
    produces the sound.
    """
    from bettersdr.dsp.demod import WfmDemodulator

    rate = 2_400_000.0
    bits = _program_service_groups(0x54A8 + 8_845, "ONAIRFM")
    # 2 kHz of the transmitter's 75 kHz deviation goes to the subcarrier,
    # which is what the standard suggests and what the local stations use.
    subcarrier = mpx(bits, repeats=3, amplitude=2_000.0 / 75_000.0, rate=rate)
    t = np.arange(subcarrier.size) / rate
    composite = 0.4 * np.sin(2 * np.pi * 1_000.0 * t) + subcarrier
    phase = np.cumsum(2.0 * np.pi * 75_000.0 * composite / rate)
    iq = np.exp(1j * phase).astype(np.complex64)

    receiver = RdsReceiver(240_000.0)
    demodulator = WfmDemodulator(rate)
    demodulator.mpx_sink = receiver
    for start in range(0, iq.size, 32_768):
        audio = demodulator.process(iq[start : start + 32_768])
        assert np.all(np.isfinite(audio))

    state = receiver.snapshot()
    assert state.station.strip() == "ONAIRFM"
    assert state.callsign == "WNCF"
    assert state.quality > 0.95


def test_a_scrolling_name_field_falls_back_to_the_callsign():
    """Stations push song titles through the eight-character name field.

    Measured off 94.9 MHz, where the frames read " on KUOW" and "NPR's He" in
    turn. Showing whichever arrived last as the station's name is wrong twice
    a second; the callsign is right permanently.
    """
    pi = 0x54A8 + 8_845
    bits: list[int] = []
    for frame in ("Song One", "Song Two", "Song Six"):
        bits += _program_service_groups(pi, frame)
    state = _receive(mpx(bits, repeats=2)).snapshot()

    assert not state.station_steady
    assert state.station.strip().startswith("Song")
    assert state.name == "WNCF"


def test_a_station_that_keeps_its_name_still_uses_it():
    state = _receive(mpx(_program_service_groups(0x54A8, "SteadyFM"))).snapshot()
    assert state.station_steady
    assert state.name == "SteadyFM"
