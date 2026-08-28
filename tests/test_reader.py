"""Tests for the reader thread.

No hardware involved: a fake device stands in for the dongle, which lets the
tests pin down the behaviour that actually matters - that the loop keeps
pumping, that device calls happen on the reader thread and nowhere else, and
that an unplugged dongle surfaces as an error instead of a hang.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from bettersdr.core.native import RtlSdrError
from bettersdr.core.reader import Reader
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
