"""Tests for the reader thread.

No hardware involved: a fake device stands in for the dongle, which lets the
tests pin down the behaviour that actually matters - that the loop keeps
pumping, that device calls happen on the reader thread and nowhere else, and
that an unplugged dongle surfaces as an error instead of a hang.
"""

from __future__ import annotations

import ctypes
import threading
import time

import numpy as np

from bettersdr.audio.output import DEFAULT_TARGET_LATENCY_S
from bettersdr.core.native import RtlSdrError
from bettersdr.core.reader import DEFAULT_READ_BYTES, READ_SECONDS, Reader, read_bytes_for
from bettersdr.core.ringbuffer import RingBuffer


class FakeDevice:
    """Stands in for `Device`, recording which thread touched it."""

    def __init__(self, fail_from: int | None = None) -> None:
        self.reads = 0
        self.resets = 0
        self.center_freq = 0
        self.gain_db = 0.0
        self.manual_gain: bool | None = None
        self.fail_from = fail_from
        self.threads: set[str] = set()

    def _note_thread(self) -> None:
        self.threads.add(threading.current_thread().name)

    def reset_buffer(self) -> None:
        self._note_thread()
        self.resets += 1

    def read(self, n_bytes: int) -> np.ndarray:
        self._note_thread()
        self.reads += 1
        if self.fail_from is not None and self.reads >= self.fail_from:
            raise RtlSdrError(-5, "rtlsdr_read_sync")
        time.sleep(0.001)
        return np.full(n_bytes, self.reads % 256, dtype=np.uint8)

    def set_manual_gain(self, enabled: bool) -> None:
        self._note_thread()
        self.manual_gain = enabled


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_reader_fills_the_ring():
    device = FakeDevice()
    reader = Reader(device, RingBuffer(65_536), block_bytes=1024)
    reader.start()
    try:
        assert reader.wait_until_running(timeout=2.0)
        assert wait_for(lambda: len(reader.ring) >= 4096)
    finally:
        reader.stop()

    assert reader.blocks_read >= 4
    assert reader.errors == 0


def test_reader_resets_the_device_buffer_before_reading():
    device = FakeDevice()
    reader = Reader(device, RingBuffer(8192), block_bytes=512)
    reader.start()
    try:
        assert reader.wait_until_running(timeout=2.0)
    finally:
        reader.stop()
    # Stale samples from before the stream started must not reach the DSP.
    assert device.resets >= 1


def test_commands_run_on_the_reader_thread():
    """`Device` is not thread-safe, so only the reader may call into it."""
    device = FakeDevice()
    reader = Reader(device, RingBuffer(8192), block_bytes=512)
    reader.start()
    try:
        assert reader.wait_until_running(timeout=2.0)
        reader.tune(101_100_000)
        reader.set_gain(28.0)
        assert wait_for(lambda: device.center_freq == 101_100_000)
        assert wait_for(lambda: device.gain_db == 28.0)
    finally:
        reader.stop()

    assert device.manual_gain is True
    assert device.threads == {"sdr-reader"}


def test_set_gain_none_restores_hardware_auto_gain():
    device = FakeDevice()
    device.manual_gain = True
    reader = Reader(device, RingBuffer(8192), block_bytes=512)
    reader.start()
    try:
        assert reader.wait_until_running(timeout=2.0)
        reader.set_gain(None)
        assert wait_for(lambda: device.manual_gain is False)
    finally:
        reader.stop()


def test_stop_ends_the_thread():
    reader = Reader(FakeDevice(), RingBuffer(8192), block_bytes=512)
    reader.start()
    assert reader.wait_until_running(timeout=2.0)
    reader.stop(timeout=2.0)
    assert not reader.is_alive()


def test_read_failures_are_counted_and_give_up():
    """An unplugged dongle should end the loop, not spin or hang forever."""
    device = FakeDevice(fail_from=3)
    reader = Reader(device, RingBuffer(8192), block_bytes=512)
    reader.start()
    reader.join(timeout=3.0)

    assert not reader.is_alive()
    assert reader.errors > 10
    assert reader.last_error is not None
    assert "rtlsdr_read_sync" in reader.last_error


def test_ring_overruns_when_nothing_consumes():
    """A stalled consumer must cost old samples, never block the reader."""
    device = FakeDevice()
    reader = Reader(device, RingBuffer(2048), block_bytes=512)
    reader.start()
    try:
        assert wait_for(lambda: reader.ring.overruns > 0, timeout=3.0)
    finally:
        reader.stop()

    assert reader.blocks_read > 4
    assert not reader.is_alive()


# -- read size follows the rate --------------------------------------------


def test_read_size_reproduces_the_measured_default_at_full_rate():
    """128 KB at 2.4 MS/s is a measured optimum, not a round number."""
    assert read_bytes_for(2_400_000) == DEFAULT_READ_BYTES


def test_read_size_covers_the_same_span_of_time_at_every_rate():
    for rate in (240_000, 960_000, 1_200_000, 2_400_000):
        seconds = read_bytes_for(rate) / 2 / rate
        assert 0.5 * READ_SECONDS <= seconds <= 1.5 * READ_SECONDS, rate


def test_read_size_stays_legal_for_read_sync():
    """`rtlsdr_read_sync` requires a whole number of 512-byte blocks."""
    for rate in (240_000, 288_000, 960_000, 1_152_000, 2_400_000):
        assert read_bytes_for(rate) % 512 == 0


def test_a_narrow_window_never_reads_longer_than_the_audio_buffer():
    """The bug this replaced: 128 KB at 240 kS/s is 273 ms per read, against
    a 150 ms jitter buffer, so every read arrived after the sink had run dry."""
    for rate in (240_000, 960_000, 2_400_000):
        assert read_bytes_for(rate) / 2 / rate < DEFAULT_TARGET_LATENCY_S


# -- a rejected command must not take the radio with it --------------------


class PickyDevice(FakeDevice):
    """Refuses a frequency outside the dongle's range, as `Device` does."""

    armed = False

    def __setattr__(self, name: str, value: object) -> None:
        if name == "center_freq" and self.armed and value < 500_000:
            raise ValueError(f"{value} Hz is outside the dongle's range")
        object.__setattr__(self, name, value)


def test_a_rejected_tune_does_not_kill_the_reader():
    """Clicking the spectrum below 500 kHz used to stop the radio dead.

    `Device.center_freq` raises ValueError rather than RtlSdrError, which the
    command loop did not catch, so the thread unwound and the app went deaf
    with a frozen display and nothing in `errors` to say why.
    """
    device = PickyDevice()
    device.armed = True
    reader = Reader(device, block_bytes=4096)
    reader.start()
    try:
        assert reader.wait_until_running()
        before = reader.blocks_read
        reader.tune(400_000)
        assert wait_for(lambda: reader.errors > 0)

        assert reader.is_alive(), "the reader must survive a rejected command"
        assert wait_for(lambda: reader.blocks_read > before + 3), "still pumping"
        assert reader.last_error is not None
    finally:
        reader.stop()


def test_a_rejected_command_is_reported_not_swallowed():
    device = FakeDevice()
    reader = Reader(device, block_bytes=4096)
    reader.start()
    try:
        reader.wait_until_running()
        reader.submit(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
        assert wait_for(lambda: reader.errors == 1)
        assert "boom" in (reader.last_error or "")
        assert reader.is_alive()
    finally:
        reader.stop()


# -- gapless streaming -------------------------------------------------------
#
# HD Radio is the one thing in the app that cannot live with the gap between
# two `read_sync` calls - see the module docstring in `core/reader.py`. What
# is checked here is the shape of the second mode rather than the timing: that
# it is entered and left through the ordinary command queue, that the callback
# still runs on the reader thread, and above all that no device call is made
# from inside it. libusb is not reentrant, and a control transfer issued from
# within its own event handling fails silently - a radio that did not retune.


class StreamingDevice(FakeDevice):
    """A fake `read_async`: delivers transfers until cancelled.

    Deliberately calls the callback the way librtlsdr does, on the thread that
    asked for the stream, so `test_no_device_call_happens_inside_the_callback`
    is testing the real arrangement rather than a convenient one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.transfers = 0
        self.streams = 0
        self.cancelled = False
        self.calls_during_callback: list[str] = []
        self._in_callback = False

    def _note_thread(self) -> None:
        super()._note_thread()
        if self._in_callback:
            self.calls_during_callback.append(threading.current_thread().name)

    def read_async(self, callback, buffer_count: int = 16,
                   buffer_bytes: int = 131_072) -> None:
        self._note_thread()
        self.streams += 1
        self.cancelled = False
        block = (ctypes.c_uint8 * 1024)()
        while not self.cancelled:
            self.transfers += 1
            self._in_callback = True
            try:
                callback(block, 1024, None)
            finally:
                self._in_callback = False
            time.sleep(0.001)

    def cancel_async(self) -> None:
        self.cancelled = True


def test_gapless_mode_is_entered_through_the_command_queue():
    device = StreamingDevice()
    reader = Reader(device, RingBuffer(65_536), block_bytes=1024)
    reader.start()
    try:
        assert reader.wait_until_running(timeout=2.0)
        reads_before = device.reads
        reader.set_gapless(True)
        assert wait_for(lambda: device.transfers > 20)
        # Nothing is reading a block at a time any more.
        assert device.reads == reads_before
        assert device.streams == 1
    finally:
        reader.stop()


def test_the_stream_callback_runs_on_the_reader_thread():
    """`Device` has exactly one owner in both modes, which is what lets the
    gapless path exist at all rather than being a second way in."""
    device = StreamingDevice()
    reader = Reader(device, RingBuffer(65_536), block_bytes=1024)
    reader.start()
    try:
        reader.set_gapless(True)
        assert wait_for(lambda: len(reader.ring) > 4096)
    finally:
        reader.stop()
    assert device.threads == {"sdr-reader"}


def test_no_device_call_happens_inside_the_callback():
    """libusb is not reentrant. A control transfer from inside its own event
    handling comes back `LIBUSB_ERROR_BUSY`, and librtlsdr carries on - so the
    radio silently does not retune. Found off air by surveying eight
    frequencies and reading the same call letters off all of them.
    """
    device = StreamingDevice()
    reader = Reader(device, RingBuffer(65_536), block_bytes=1024)
    reader.start()
    try:
        reader.set_gapless(True)
        assert wait_for(lambda: device.transfers > 5)
        reader.tune(94_900_000)
        assert wait_for(lambda: device.center_freq == 94_900_000)
    finally:
        reader.stop()
    assert device.calls_during_callback == []
    # The retune ended the stream and it was entered again afterwards.
    assert device.streams >= 2


def test_leaving_gapless_mode_goes_back_to_reading_blocks():
    device = StreamingDevice()
    reader = Reader(device, RingBuffer(65_536), block_bytes=1024)
    reader.start()
    try:
        reader.set_gapless(True)
        assert wait_for(lambda: device.transfers > 5)
        reads_before = device.reads
        reader.set_gapless(False)
        assert wait_for(lambda: device.reads > reads_before + 5)
        transfers = device.transfers
        time.sleep(0.05)
        assert device.transfers == transfers
    finally:
        reader.stop()


def test_stopping_cancels_a_running_stream():
    """A thread inside `read_async` is not looking at the stop flag; it is
    blocked in librtlsdr until something cancels it."""
    device = StreamingDevice()
    reader = Reader(device, RingBuffer(65_536), block_bytes=1024)
    reader.start()
    reader.set_gapless(True)
    assert wait_for(lambda: device.transfers > 5)
    reader.stop(timeout=2.0)
    assert not reader.is_alive()
