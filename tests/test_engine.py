"""Tests for the engine's thread-crossing pieces.

The engine itself needs a dongle, but the two things that carry data between
threads do not, and they are where a mistake would be least visible: a mailbox
that queued instead of dropping would build unbounded latency, and a frame
whose bin frequencies were off by half a bin would put every signal on screen
at the wrong place.
"""

from __future__ import annotations

import threading

import numpy as np

from bettersdr.core.engine import DisplayFrame, Mailbox


def frame(bins: int = 8, center_hz: float = 100e6, rate: float = 2.4e6) -> DisplayFrame:
    return DisplayFrame(
        spectrum_db=np.zeros(bins, dtype=np.float32),
        center_hz=center_hz,
        sample_rate=rate,
        bin_width_hz=rate / bins,
        channel_power_dbfs=-30.0,
        bandwidth_hz=200_000.0,
        squelch_open=None,
        audio_latency_s=0.15,
        underruns=0,
        ring_overruns=0,
    )


# -- Mailbox ---------------------------------------------------------------


def test_empty_mailbox_reads_as_nothing():
    assert Mailbox().peek() is None


def test_newest_value_replaces_the_old_one():
    """Dropping frames is correct for a display; queueing them is not."""
    box: Mailbox[int] = Mailbox()
    for value in range(100):
        box.put(value)
    assert box.peek() == 99


def test_peek_leaves_the_value_in_place():
    box: Mailbox[str] = Mailbox()
    box.put("frame")
    assert box.peek() == "frame"
    assert box.peek() == "frame"


def test_a_writer_thread_and_a_reader_thread_do_not_tear():
    """The GUI must never see a half-written slot while the DSP thread writes."""
    box: Mailbox[int] = Mailbox()
    box.put(0)
    stop = threading.Event()
    seen: list[int | None] = []

    def writer() -> None:
        for value in range(20_000):
            box.put(value)
        stop.set()

    thread = threading.Thread(target=writer)
    thread.start()
    while not stop.is_set():
        seen.append(box.peek())
    thread.join()

    assert all(value is not None for value in seen)
    assert box.peek() == 19_999


# -- DisplayFrame ----------------------------------------------------------

def test_frame_bin_frequencies_match_the_fft_convention():
    """Must agree with psd.Spectrum, or the display and detector disagree."""
    bins, rate, center = 8, 2.4e6, 100e6
    expected = center + np.fft.fftshift(np.fft.fftfreq(bins, 1.0 / rate))
    np.testing.assert_allclose(frame(bins, center, rate).frequencies(), expected)


def test_frame_is_centred_on_the_tuned_frequency():
    freqs = frame(bins=1024).frequencies()
    assert freqs[512] == 100e6
    assert freqs[0] < 100e6 < freqs[-1]


def test_frame_cannot_be_mutated_after_it_crosses_threads():
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        frame().center_hz = 1.0
