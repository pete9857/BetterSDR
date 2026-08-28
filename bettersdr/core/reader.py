"""The reader thread: the only thread allowed to touch the device.

Its loop does as close to nothing as possible - `rtlsdr_read_sync` straight
into the ring buffer - because any work done here is time during which no USB
transfer is in flight and samples are dropped on the floor. Measured on a V4,
demodulating on this thread instead of a separate one costs about 11% of the
sample stream, which is audible as a steady trickle of audio underruns.

Retuning and gain changes arrive as callables on a queue and run *between*
reads. That is why the app uses `read_sync` rather than `read_async`: it gives
a natural point to reconfigure the hardware synchronously, which the scanner
needs, without a second mechanism.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .device import Device
from .native import RtlSdrError
from .ringbuffer import RingBuffer

# Half a second of headroom at 2.4 MS/s. Large enough to ride out a scheduling
# stall, small enough that a backlog cannot become audible latency.
DEFAULT_CAPACITY_BYTES = 4 * 1024 * 1024

# 128 KB is ~27 ms of radio at 2.4 MS/s. Read size turns out to matter a lot
# more than it looks: every call has a fixed cost during which no transfer is
# in flight, so capture measured 99.49% at 16 KB, 99.72% at 64 KB and 100% at
# 256 KB. Anything below 100% drains the audio buffer without ever showing up
# as a ring overrun. 128 KB gets the loss into the noise while keeping retune
# latency low enough for the scanner to step quickly.
DEFAULT_READ_BYTES = 131_072

Command = Callable[[Device], None]


class Reader(threading.Thread):
    """Pumps IQ from the device into a ring buffer until stopped."""

    def __init__(
        self,
        device: Device,
        ring: RingBuffer | None = None,
        block_bytes: int = DEFAULT_READ_BYTES,
    ) -> None:
        super().__init__(name="sdr-reader", daemon=True)
        self.device = device
        self.ring = ring if ring is not None else RingBuffer(DEFAULT_CAPACITY_BYTES)
        self.block_bytes = int(block_bytes)
        self.blocks_read = 0
        self.errors = 0
        self.last_error: str | None = None

        self._commands: queue.Queue[Command] = queue.Queue()
        self._stop = threading.Event()
        self._idle = threading.Event()

    # -- control -----------------------------------------------------------

    def submit(self, command: Command) -> None:
        """Queue a device call to run between reads.

        `Device` is not thread-safe and the reader owns it, so this is the only
        supported way to change the radio from another thread.
        """
        self._commands.put(command)

    def tune(self, hz: int) -> None:
        def command(device: Device) -> None:
            device.center_freq = hz
            device.reset_buffer()

        self.submit(command)

    def set_gain(self, db: float | None) -> None:
        def command(device: Device) -> None:
            if db is None:
                device.set_manual_gain(False)
            else:
                device.set_manual_gain(True)
                device.gain_db = db

        self.submit(command)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout)

    def wait_until_running(self, timeout: float = 2.0) -> bool:
        """Block until the first read has completed."""
        return self._idle.wait(timeout)

    # -- thread body -------------------------------------------------------

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                command(self.device)
            except RtlSdrError as exc:
                self.errors += 1
                self.last_error = str(exc)
            finally:
                self._commands.task_done()

    def run(self) -> None:
        try:
            self.device.reset_buffer()
        except RtlSdrError as exc:
            self.last_error = str(exc)
        while not self._stop.is_set():
            self._drain_commands()
            try:
                block = self.device.read(self.block_bytes)
            except RtlSdrError as exc:
                self.errors += 1
                self.last_error = str(exc)
                # A failed read usually means the dongle was unplugged. Keep
                # the thread alive so the UI can report it rather than hang.
                if self.errors > 10:
                    break
                continue
            self.ring.write(block)
            self.blocks_read += 1
            self._idle.set()


__all__ = ["DEFAULT_CAPACITY_BYTES", "DEFAULT_READ_BYTES", "Command", "Reader"]
