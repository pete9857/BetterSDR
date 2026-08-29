"""Tests for the POCSAG decoder, against a synthetic pager transmitter.

`tests/synth_pocsag.py` is the specification read forwards - preamble, sync
codeword, batches of eight frames, BCH checkword, 7-bit ASCII - so a round
trip through it exercises every step of the receiver reading it backwards.
Where a test plants a timing offset, an inverted deviation or a tuning error,
that is a fault the real air produces and the receiver has to survive on its
own.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.decode.pocsag import (
    IDLE_WORD,
    SYNC_WORD,
    PocsagReceiver,
    check,
    syndrome,
)
from bettersdr.dsp.demod import NfmDemodulator

from . import synth_pocsag

IF_RATE = 96_000.0


def _receive(
    signal: np.ndarray, block: int = 1_310, if_rate: float = IF_RATE
) -> PocsagReceiver:
    """Feed a whole transmission through in blocks, as the engine would.

    1,310 samples is what a 64 KB DSP block comes to at a 96 kHz IF, and no
    part of the chain is allowed to care: a stage that forgets its history at
    a block boundary ticks once per block, which is invisible in a one-shot
    test and obvious on air.
    """
    receiver = PocsagReceiver(if_rate)
    for start in range(0, signal.size, block):
        receiver.process(signal[start : start + block])
    return receiver


def _pages(receiver: PocsagReceiver) -> list:
    return [page for page in receiver.snapshot().pages if page.kind != "tone"]


def _transmit(pages, baud=1200, **kwargs) -> np.ndarray:
    bits = synth_pocsag.bitstream(pages)
    return synth_pocsag.deviation(bits, baud, IF_RATE, **kwargs)


# -- the checkword -----------------------------------------------------------


def test_the_checkword_matches_an_independent_long_division():
    for payload in (0x000000, 0x1FFFFF, 0x0ABCDE, 0x123456):
        word = synth_pocsag.codeword(payload)
        assert syndrome(word) == 0
        assert check(word) == (word, 0)


def test_the_standard_constants_check_out_as_codewords():
    # Both are real codewords, which is the cheapest possible confirmation
    # that the polynomial and the bit order are the ones the standard means.
    assert check(SYNC_WORD) == (SYNC_WORD, 0)
    assert check(IDLE_WORD) == (IDLE_WORD, 0)


def test_one_bad_bit_is_repaired_and_two_are_refused():
    word = synth_pocsag.codeword(0x0ABCDE)
    for bit in range(32):
        assert check(word ^ (1 << bit)) == (word, 1)
    for first in range(0, 32, 7):
        for second in range(first + 1, 32, 5):
            assert check(word ^ (1 << first) ^ (1 << second)) is None


# -- reading a message off the air -------------------------------------------


@pytest.mark.parametrize("baud", [512, 1200, 2400])
def test_an_alphanumeric_page_comes_back_whole_at_every_rate(baud):
    message = "CALL SWITCHBOARD EXT 4471"
    signal = _transmit([(1234568, 3, synth_pocsag.alphanumeric(message))], baud=baud)
    pages = _pages(_receive(signal))
    assert len(pages) == 1
    assert pages[0].capcode == 1234568
    assert pages[0].kind == "alphanumeric"
    assert pages[0].text == message
    assert pages[0].baud == baud


def test_a_numeric_page_comes_back_as_its_digits():
    signal = _transmit([(45678, 0, synth_pocsag.numeric("206-555-0142"))])
    pages = _pages(_receive(signal))
    assert len(pages) == 1
    assert pages[0].kind == "numeric"
    assert pages[0].text.startswith("206-555-0142")


def test_the_capcode_carries_the_frame_it_arrived_in():
    # The low three bits are never transmitted; they are which of the eight
    # frames the address codeword was placed in. A decoder that ignores that
    # gets every capcode wrong by up to seven and looks almost right.
    capcodes = [1000000 + offset for offset in range(8)]
    signal = _transmit(
        [(code, 3, synth_pocsag.alphanumeric(f"N{code % 10}")) for code in capcodes]
    )
    assert [page.capcode for page in _pages(_receive(signal))] == capcodes


def test_several_pages_in_one_transmission_all_arrive():
    messages = ["FIRST ONE", "SECOND MESSAGE HERE", "THIRD"]
    signal = _transmit(
        [
            (900008 + index, 3, synth_pocsag.alphanumeric(text))
            for index, text in enumerate(messages)
        ]
    )
    assert [page.text for page in _pages(_receive(signal))] == messages


def test_a_message_spanning_several_batches_is_not_cut_in_half():
    long_message = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG " * 3
    signal = _transmit([(777776, 3, synth_pocsag.alphanumeric(long_message))])
    pages = _pages(_receive(signal))
    assert len(pages) == 1
    assert pages[0].text == long_message.rstrip()


def test_a_page_with_no_message_is_reported_as_a_beep():
    signal = _transmit([(654320, 1, [])])
    pages = _receive(signal).snapshot().pages
    assert len(pages) == 1
    assert pages[0].kind == "tone"
    assert pages[0].label == "Beep - no message"


# -- the things the air does to it -------------------------------------------


def test_inverted_deviation_reads_the_same_message():
    # Which way round the deviation runs depends on the transmitter and on
    # which side of the tuner the channel landed. The sync codeword is what
    # says which it was, and both readings have to give the same text.
    message = "POLARITY DOES NOT MATTER"
    words = synth_pocsag.alphanumeric(message)
    for invert in (False, True):
        signal = _transmit([(112232, 3, words)], invert=invert)
        assert [page.text for page in _pages(_receive(signal))] == [message]


@pytest.mark.parametrize("offset", [0.0, 0.17, 0.31, 0.5, 0.73, 0.94])
def test_an_arbitrary_arrival_phase_decodes_just_as_well(offset):
    # A transmission starts whenever it starts. A decoder that only works at
    # one alignment passes a fixed test and fails on air, which is the exact
    # shape of the ADS-B slicing bug this test exists to rule out.
    message = "ARRIVED MID SAMPLE"
    signal = _transmit([(334456, 3, synth_pocsag.alphanumeric(message))], offset=offset)
    assert [page.text for page in _pages(_receive(signal))] == [message]


def test_a_tuning_error_offsets_the_slicer_and_is_tracked_out():
    # An FM discriminator sits off zero by however far the tuning is off. At
    # a quarter of full deviation a fixed threshold at zero reads every bit
    # as a one.
    message = "OFF FREQUENCY"
    signal = synth_pocsag.channel(
        _transmit([(556678, 3, synth_pocsag.alphanumeric(message))]), dc=0.25
    )
    assert [page.text for page in _pages(_receive(signal))] == [message]


def test_a_transmitter_clock_a_hundred_ppm_out_still_reads():
    message = "SOMEBODY ELSES CLOCK"
    signal = _transmit([(778890, 3, synth_pocsag.alphanumeric(message))], ppm=100.0)
    assert [page.text for page in _pages(_receive(signal))] == [message]


def test_a_page_survives_noise():
    message = "NOISY BUT LEGIBLE"
    signal = synth_pocsag.channel(
        _transmit([(223344, 3, synth_pocsag.alphanumeric(message))]),
        noise=0.15,
        seed=7,
    )
    assert [page.text for page in _pages(_receive(signal))] == [message]


def test_the_block_size_does_not_change_what_is_decoded():
    message = "BLOCK BOUNDARIES ARE NOT AUDIBLE"
    signal = _transmit([(998874, 3, synth_pocsag.alphanumeric(message))])
    for block in (256, 1_310, 4_096, 32_768):
        assert [page.text for page in _pages(_receive(signal, block=block))] == [message]


def test_noise_alone_decodes_to_nothing_rather_than_to_something():
    noise = np.random.default_rng(3).normal(0.0, 1.0, int(IF_RATE * 3))
    state = _receive(noise).snapshot()
    assert state.pages == ()


def test_silence_decodes_to_nothing():
    state = _receive(np.zeros(int(IF_RATE * 2))).snapshot()
    assert state.pages == ()


def test_reset_forgets_the_previous_transmitter():
    signal = _transmit([(123456, 3, synth_pocsag.alphanumeric("BEFORE"))])
    receiver = _receive(signal)
    assert receiver.snapshot().pages
    receiver.reset()
    state = receiver.snapshot()
    assert state.pages == ()
    assert state.codewords_ok == 0
    assert state.baud is None


def test_a_narrow_if_is_refused_rather_than_decoded_badly():
    with pytest.raises(ValueError):
        PocsagReceiver(8_000.0)


def test_the_reported_rate_is_the_one_that_produced_the_pages():
    signal = _transmit([(556670, 3, synth_pocsag.alphanumeric("RATE"))], baud=2400)
    assert _receive(signal).snapshot().baud == 2400


# -- through the radio -------------------------------------------------------


def test_a_page_is_read_through_the_fm_demodulator():
    """End to end: FSK on a carrier, through the NFM path, out as text.

    This is the only test that exercises the tap the engine actually uses -
    the discriminator output, before the audio filter has rounded the corners
    off the bits and before any squelch could have muted them.
    """
    rate = 2_400_000.0
    message = "OFF THE AIR AND ONTO THE SCREEN"
    bits = synth_pocsag.bitstream([(1400008, 3, synth_pocsag.alphanumeric(message))])
    baseband = synth_pocsag.deviation(bits, 1200, rate, lead=0.05)
    # Phase is the integral of instantaneous frequency, so the deviation
    # accumulates rather than being written into the phase directly.
    phase = np.cumsum(2 * np.pi * 4_500.0 * baseband / rate)
    iq = (0.4 * np.exp(1j * phase)).astype(np.complex64)

    demod = NfmDemodulator(rate, bandwidth_hz=20_000.0)
    assert demod.if_rate == 96_000.0
    receiver = PocsagReceiver(demod.if_rate)
    # `data_sink`, not `mpx_sink`: the pager decoder has a tap of its own, so
    # that attaching one decoder can never quietly detach the other.
    demod.data_sink = receiver
    for start in range(0, iq.size, 32_768):
        demod.process(iq[start : start + 32_768])

    pages = _pages(receiver)
    assert [page.text for page in pages] == [message]
    assert pages[0].capcode == 1400008
