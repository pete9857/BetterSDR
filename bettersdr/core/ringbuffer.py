"""A single-producer, single-consumer ring buffer for raw IQ bytes.

The reader thread must never wait for the DSP thread. If it does, no USB
transfer is in flight and the dongle's samples are simply lost - there is no
back-pressure mechanism on a radio. So the buffer is preallocated, writes never
block, and when the consumer falls behind the *oldest* data is discarded rather
than the newest. Stale IQ has no value; the newest samples are the ones the
user is listening to.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Fixed-capacity byte ring. One writer thread, one reader thread."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"capacity must be positive, got {capacity_bytes}")
        self.capacity = int(capacity_bytes)
        self._buf = np.zeros(self.capacity, dtype=np.uint8)
        self._write = 0
        self._read = 0
        self._available = 0
        self._lock = threading.Lock()
        self._data_ready = threading.Condition(self._lock)
        self.overruns = 0
        self.dropped_bytes = 0

    def __len__(self) -> int:
        return self._available

    @property
    def fill_fraction(self) -> float:
        return self._available / self.capacity

    def clear(self) -> None:
        with self._lock:
            self._write = self._read = self._available = 0

    def write(self, block: np.ndarray) -> None:
        """Append a block. Never blocks; overwrites the oldest data if full."""
        block = np.asarray(block, dtype=np.uint8)
        if block.size == 0:
            return
        # A block larger than the whole ring can only leave its tail behind.
        if block.size >= self.capacity:
            block = block[-self.capacity :]

        with self._lock:
            end = self._write + block.size
            if end <= self.capacity:
                self._buf[self._write : end] = block
            else:
                split = self.capacity - self._write
                self._buf[self._write :] = block[:split]
                self._buf[: end - self.capacity] = block[split:]
            self._write = end % self.capacity
            self._available += block.size

            if self._available > self.capacity:
                lost = self._available - self.capacity
                self._read = (self._read + lost) % self.capacity
                self._available = self.capacity
                self.overruns += 1
                self.dropped_bytes += lost

            self._data_ready.notify()

    def read(self, n_bytes: int, timeout: float | None = None) -> np.ndarray | None:
        """Take exactly `n_bytes`, waiting up to `timeout` for them to arrive.

        Returns None on timeout so a stalled dongle surfaces as a stall rather
        than a hang.
        """
        if n_bytes <= 0:
            return np.zeros(0, dtype=np.uint8)
        if n_bytes > self.capacity:
            raise ValueError(f"asked for {n_bytes} bytes from a {self.capacity} ring")

        with self._lock:
            if not self._data_ready.wait_for(
                lambda: self._available >= n_bytes, timeout
            ):
                return None
            end = self._read + n_bytes
            if end <= self.capacity:
                out = self._buf[self._read : end].copy()
            else:
                out = np.concatenate(
                    (self._buf[self._read :], self._buf[: end - self.capacity])
                )
            self._read = end % self.capacity
            self._available -= n_bytes
            return out


__all__ = ["RingBuffer"]
