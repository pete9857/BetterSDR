"""The reader thread: the only thread allowed to touch the device.

Its loop does as close to nothing as possible - `rtlsdr_read_sync` straight
into the ring buffer - because any work done here is time during which no USB
transfer is in flight and samples are dropped on the floor. Measured on a V4,
demodulating on this thread instead of a separate one costs about 11% of the
sample stream, which is audible as a steady trickle of audio underruns.

Retuning and gain changes arrive as callables on a queue and run *between*
reads. That is why the app reads one block at a time rather than streaming:
it gives a natural point to reconfigure the hardware synchronously, which the
scanner needs, without a second mechanism.

There is one thing that cannot live with it, and only one. Between two reads
no USB transfer is in flight, so the sample stream has a small gap at every
read boundary - inaudible on any analog mode, invisible to the scanner, and
irrelevant to ADS-B, all of which work a block at a time. An OFDM receiver
tracks a frame *across* blocks, and HD Radio simply does not decode through
it: measured off air on the same station, at the same gain and rate, one
block at a time gave -13 dB of modulation error ratio and no audio, and
several transfers in flight gave +9 to +10 dB with no loss of sync at all.
So `set_gapless` exists, it is used only while HD Radio holds the radio, and
the callback it streams into runs on this same thread - which is what keeps
the one-owner rule intact rather than adding a second way in.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable

import numpy as np

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

# What actually matters is the *duration* a read covers, not its size: 128 KB
# is 27 ms at 2.4 MS/s but 273 ms at 240 kS/s, which is longer than the whole
# audio jitter buffer and starves it on every read. So the byte count is
# derived from the rate and this constant is the real setting, chosen to
# reproduce the measured 128 KB exactly at 2.4 MS/s.
READ_SECONDS = DEFAULT_READ_BYTES / 2 / 2_400_000


# How many USB transfers to keep queued in gapless mode, and how big each
# one is. The size is *not* an audio latency setting the way `READ_SECONDS`
# is - with transfers queued behind the one being delivered, latency is one
# transfer rather than a round trip - so it is stated in bytes and this is
# the pair that was measured to decode. Gapless mode only ever runs at HD
# Radio's 1,488,375 S/s, where 128 KB is 44 ms.
GAPLESS_BUFFERS = 16
GAPLESS_BYTES = 131_072


def read_bytes_for(sample_rate_hz: float) -> int:
    """USB read size covering `READ_SECONDS` of radio at this rate.

    `rtlsdr_read_sync` requires a multiple of 512 bytes, and a floor keeps a
    very narrow window from reducing this to an inefficient dribble.
    """
    raw = int(sample_rate_hz * 2 * READ_SECONDS)
    return max(16_384, (raw // 512) * 512)

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
        # Owned by this thread once it is running. Set through `set_gapless`,
        # which is a queued command like every other change to the radio.
        self._gapless = False

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

    def set_sample_rate(self, hz: int) -> None:
        """Change how wide a window the dongle looks through.

        The ring still holds bytes captured at the old rate, and demodulating
        those at the new one is a burst of noise, so it is emptied here rather
        than left for the DSP thread to trip over.
        """

        def command(device: Device) -> None:
            device.sample_rate = int(hz)
            self.block_bytes = read_bytes_for(hz)
            device.reset_buffer()
            self.ring.clear()

        self.submit(command)

    def set_gain(self, db: float | None) -> None:
        def command(device: Device) -> None:
            if db is None:
                device.set_manual_gain(False)
            else:
                device.set_manual_gain(True)
                device.gain_db = db

        self.submit(command)

    def set_gapless(self, enabled: bool) -> None:
        """Keep several USB transfers in flight instead of reading one block.

        Only HD Radio needs this - see the module docstring for the
        measurement - and it costs something real, so it is not the default:
        while streaming, a queued command only runs when a transfer completes,
        and the stream has to be torn down and rebuilt around anything that
        reads the device directly, such as a gain probe.

        Queued like every other change to the radio, so it is ordered against
        the retunes and rate changes around it. A caller that wants a gain
        measured before streaming starts submits the probe first.
        """

        def command(_device: Device) -> None:
            self._gapless = bool(enabled)

        self.submit(command)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        # A thread inside `read_async` is not looking at `_stop`; it is
        # blocked in librtlsdr until something cancels it.
        if self._gapless:
            with contextlib.suppress(Exception):  # already on the way out
                self.device.cancel_async()
        if self.is_alive():
            self.join(timeout)

    def wait_until_running(self, timeout: float = 2.0) -> bool:
        """Block until the first read has completed."""
        return self._idle.wait(timeout)

    # -- thread body -------------------------------------------------------

    def _drain_commands(self) -> None:
        while self._run_one_command():
            pass

    def _run_one_command(self) -> bool:
        """Run the next queued device call. False when there was none."""
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return False
        try:
            command(self.device)
        except Exception as exc:
            # Deliberately every exception, not just RtlSdrError. A
            # command that raises anything else used to take the whole
            # reader thread with it, and silently: `errors` stayed at
            # zero, so nothing upstream could tell that the radio had
            # stopped. Clicking the spectrum below 500 kHz did exactly
            # that - `Device.center_freq` raises ValueError for a
            # frequency the dongle cannot reach - and the app went deaf
            # with a frozen display. A rejected command is a diagnosable
            # condition; it is never a reason to stop pumping samples.
            self.errors += 1
            self.last_error = str(exc)
        finally:
            self._commands.task_done()
        return True

    def _stream(self) -> None:
        """Pump the device with several transfers queued, until asked to stop.

        Blocks for as long as gapless mode lasts: `rtlsdr_read_async` does not
        return until it is cancelled. Everything the loop in `run` would do
        between reads happens in the callback instead - which librtlsdr calls
        on *this* thread, so `Device` still has exactly one owner and
        `tests/test_reader.py` still holds.

        **No device call may be made from inside the callback.** libusb is
        not reentrant, so a control transfer issued from within its own event
        handling comes back `LIBUSB_ERROR_BUSY` - which librtlsdr reports as
        `r82xx_set_freq: failed=-6` and then carries on, leaving a radio that
        did not retune and a decoder happily reporting the station it was
        already on. Found off air surveying eight frequencies and reading the
        same call letters off all eight. So a queued command ends the stream
        instead, `run` handles it the ordinary way, and this is entered again
        afterwards. That costs a discontinuity per command, which is
        acceptable because nothing queues commands during a settled session -
        and a retune, the one thing that does, restarts the decoder anyway.

        The callback must not raise. An exception escaping a ctypes callback
        is printed to stderr and swallowed, so a fault here would otherwise
        be a radio that quietly stopped with nothing to show for it.
        """
        cancelled = False

        def on_transfer(buffer, length, _context) -> None:
            nonlocal cancelled
            try:
                self.ring.write(np.ctypeslib.as_array(buffer, shape=(int(length),)))
                self.blocks_read += 1
                self._idle.set()
            except Exception as exc:  # noqa: BLE001 - see the docstring
                self.errors += 1
                self.last_error = str(exc)
            if not cancelled and (
                self._stop.is_set() or not self._commands.empty()
            ):
                cancelled = True
                self.device.cancel_async()

        try:
            self.device.read_async(
                on_transfer,
                buffer_count=GAPLESS_BUFFERS,
                buffer_bytes=GAPLESS_BYTES,
            )
        except Exception as exc:  # noqa: BLE001 - never kill the thread
            self.errors += 1
            self.last_error = str(exc)
        if not cancelled and self._gapless and not self._stop.is_set():
            # It returned without being asked to, which means streaming is
            # not going to work on this device. Falling back to one block at
            # a time is a degraded radio; spinning here would be no radio.
            self._gapless = False
            self.last_error = self.last_error or "the gapless stream ended early"

    def run(self) -> None:
        try:
            self.device.reset_buffer()
        except RtlSdrError as exc:
            self.last_error = str(exc)
        while not self._stop.is_set():
            self._drain_commands()
            if self._gapless:
                self._stream()
                continue
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


__all__ = [
    "DEFAULT_CAPACITY_BYTES",
    "DEFAULT_READ_BYTES",
    "GAPLESS_BUFFERS",
    "GAPLESS_BYTES",
    "Command",
    "Reader",
]
