"""Tests for the engine's thread-crossing pieces.

The engine itself needs a dongle, but the two things that carry data between
threads do not, and they are where a mistake would be least visible: a mailbox
that queued instead of dropping would build unbounded latency, and a frame
whose bin frequencies were off by half a bin would put every signal on screen
at the wrong place.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from bettersdr.core.device import MAX_TUNE_HZ, MIN_TUNE_HZ
from bettersdr.core.engine import (
    DSP_BLOCK_BYTES,
    DSP_BLOCK_SECONDS,
    SPECTRUM_HZ,
    DisplayFrame,
    Engine,
    Mailbox,
    dsp_block_bytes_for,
)
from bettersdr.dsp.psd import DEFAULT_FFT_SIZE


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


def test_dsp_block_reproduces_the_default_at_full_rate():
    assert dsp_block_bytes_for(2_400_000) == DSP_BLOCK_BYTES


def test_dsp_block_covers_the_same_span_of_time_at_every_rate():
    for rate in (240_000, 960_000, 2_400_000):
        seconds = dsp_block_bytes_for(rate) / 2 / rate
        assert 0.5 * DSP_BLOCK_SECONDS <= seconds <= 1.5 * DSP_BLOCK_SECONDS, rate


@pytest.mark.parametrize("rate", [240_000, 288_000, 960_000, 1_200_000, 2_400_000])
def test_a_display_frame_can_always_be_filled_in_time(rate):
    """A block is eight FFT frames at 2.4 MS/s and less than one at 240 kS/s.

    Where it is less than one the engine carries the remainder; this checks
    the carry can never take longer than a display interval to fill, which is
    what would turn "the spectrum updates slowly" into "the spectrum stops".
    """
    samples_per_block = dsp_block_bytes_for(rate) / 2
    blocks_needed = math.ceil(DEFAULT_FFT_SIZE / samples_per_block)
    seconds = blocks_needed * samples_per_block / rate
    assert seconds <= 1.0 / SPECTRUM_HZ * 1.5, f"{rate}: {seconds*1000:.0f} ms"


# -- the tuning choke point ------------------------------------------------


def test_tuning_out_of_range_is_clamped_rather_than_raised():
    """Every tuning path goes through here, and the one that used to bypass
    the display's own clamp - click-to-tune - killed the reader thread."""
    engine = Engine()
    engine.tune(400_000)
    assert engine.center_hz == MIN_TUNE_HZ
    engine.tune(2_000_000_000)
    assert engine.center_hz == MAX_TUNE_HZ


def test_tuning_in_range_is_untouched():
    engine = Engine()
    engine.tune(94_900_000)
    assert engine.center_hz == 94_900_000


def test_asking_for_a_gain_measurement_without_a_radio_does_nothing():
    Engine().auto_gain()  # must not raise


# -- gain measurement scheduling -------------------------------------------


class FakeReader:
    """Captures submitted commands instead of touching a device."""

    def __init__(self) -> None:
        self.commands = []

    def submit(self, command) -> None:
        self.commands.append(command)


def test_two_callers_asking_at_once_produce_one_probe():
    """A band change asks, and the window change it triggers asks again. The
    probe holds the reader thread, so running it twice is a real cost - it
    emptied the audio buffer for 23 underruns on a single hop to the AM band.
    """
    engine = Engine()
    engine.reader = FakeReader()
    engine.auto_gain()
    engine.auto_gain()
    engine.auto_gain()
    assert len(engine.reader.commands) == 1


def test_a_later_ask_is_honoured_once_the_probe_has_run():
    engine = Engine()
    engine.reader = FakeReader()
    engine.auto_gain()

    class Dev:
        sample_rate = 2_400_000
        gains_db = [0.0]

        def set_manual_gain(self, on): pass
        def reset_buffer(self): pass
        def read(self, n): return np.full(n, 128, dtype=np.uint8)

    engine.reader.commands[0](Dev())
    engine.auto_gain()
    assert len(engine.reader.commands) == 2


def test_a_failing_probe_does_not_wedge_later_ones():
    """`_gain_pending` must clear even when the measurement raises, or every
    subsequent band change silently skips its gain measurement."""
    engine = Engine()
    engine.reader = FakeReader()
    engine.auto_gain()

    class Broken:
        sample_rate = 2_400_000

        @property
        def gains_db(self):
            raise RuntimeError("dongle went away")

        def set_manual_gain(self, on): pass

    with pytest.raises(RuntimeError):
        engine.reader.commands[0](Broken())

    engine.auto_gain()
    assert len(engine.reader.commands) == 2
