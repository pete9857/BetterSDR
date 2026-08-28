"""The radio engine: everything between the dongle and the screen.

This owns the second of the three threads. The reader thread fills a ring
buffer; this drains it, demodulates to audio, measures a spectrum, and drops
the result into a single-slot mailbox. The GUI thread reads that mailbox on a
timer and never blocks, never queues, and never touches the device.

Keeping the engine out of `ui/` is deliberate: `listen.py` proves the audio
path with no Qt loaded at all, so an audio fault and a UI fault stay
distinguishable. The GUI is a view onto this, not the owner of it.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..audio.output import AudioSink
from ..dsp import convert, demod
from ..dsp.psd import DEFAULT_FFT_SIZE, Spectrum
from .device import DEFAULT_SAMPLE_RATE, Device
from .frontend import GainChoice, choose_gain
from .reader import Reader

# How much of the ring to take per pass. 64 KB is ~14 ms at 2.4 MS/s, which
# keeps the meter and spectrum responsive without making the loop spin.
DSP_BLOCK_BYTES = 65_536
# Spectrum frames per second. Deliberately above the 30 Hz display rate so the
# GUI never waits on a frame, and far below the block rate so we do not compute
# frames nobody will look at.
SPECTRUM_HZ = 45.0


class Mailbox[T]:
    """A single slot where the newest value wins.

    Not a queue on purpose. If the GUI stalls for a moment we want it to
    resume with the current picture, not to work through a backlog of stale
    ones - dropping frames is the correct behaviour for a display.
    """

    def __init__(self) -> None:
        self._value: T | None = None
        self._lock = threading.Lock()

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value

    def peek(self) -> T | None:
        """The latest value, left in place."""
        with self._lock:
            return self._value


@dataclass(frozen=True)
class DisplayFrame:
    """One update for the GUI. Immutable, so it can cross threads freely."""

    spectrum_db: np.ndarray
    center_hz: float
    sample_rate: float
    bin_width_hz: float
    channel_power_dbfs: float
    bandwidth_hz: float
    squelch_open: bool | None
    audio_latency_s: float
    underruns: int
    ring_overruns: int

    def frequencies(self) -> np.ndarray:
        """Absolute frequency of each bin, low to high."""
        bins = self.spectrum_db.size
        return self.center_hz + (np.arange(bins) - bins // 2) * self.bin_width_hz


class Engine:
    """Owns the device, both worker threads, and the audio sink."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        fft_size: int = DEFAULT_FFT_SIZE,
        audio_device: int | str | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.audio_device = audio_device

        self.device: Device | None = None
        self.reader: Reader | None = None
        self.sink: AudioSink | None = None
        self.gain: GainChoice | None = None
        self.last_error: str | None = None

        self.center_hz = 98_500_000
        self.mode = "wfm"
        self.volume = 0.5
        self.squelch_dbfs: float | None = None

        self._spectrum = Spectrum(fft_size=fft_size, sample_rate=float(sample_rate))
        self._demod = demod.create(self.mode, float(sample_rate), volume=self.volume)
        self._mailbox: Mailbox[DisplayFrame] = Mailbox()
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self, center_hz: int | None = None) -> Engine:
        if self._thread is not None:
            return self
        if center_hz is not None:
            self.center_hz = int(center_hz)

        self.device = Device()
        self.device.open()
        self.device.configure(center_freq=self.center_hz, sample_rate=self.sample_rate)
        # Before the reader starts, so the probe reads have the device to
        # themselves - afterwards only the reader thread may touch it.
        self.gain = choose_gain(self.device)

        self.reader = Reader(self.device)
        self.reader.start()
        self.reader.wait_until_running()
        self.sink = AudioSink(rate=demod.AUDIO_RATE, device=self.audio_device).start()

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sdr-dsp", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.sink is not None:
            self.sink.stop()
            self.sink = None
        if self.device is not None:
            self.device.close()
            self.device = None

    def __enter__(self) -> Engine:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- controls, all callable from the GUI thread ------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> DisplayFrame | None:
        return self._mailbox.peek()

    def tune(self, center_hz: int) -> None:
        self.center_hz = int(center_hz)
        if self.reader is not None:
            self.reader.tune(self.center_hz)

    def set_gain(self, gain_db: float) -> None:
        if self.reader is not None:
            self.reader.set_gain(gain_db)

    def set_volume(self, volume: float) -> None:
        # A bare float assignment, so it takes effect on the next block with no
        # need to round-trip through the command queue.
        self.volume = float(volume)
        self._demod.volume = self.volume

    def set_mode(self, mode: str, bandwidth_hz: float | None = None) -> None:
        """Swap the demodulator. Runs on the DSP thread, which owns it."""
        self.mode = mode
        self._commands.put(lambda: self._rebuild(mode, bandwidth_hz))

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        self._commands.put(lambda: self._rebuild(self.mode, bandwidth_hz))

    def set_squelch(self, threshold_dbfs: float | None) -> None:
        self.squelch_dbfs = threshold_dbfs
        self._commands.put(lambda: self._rebuild(self.mode, None))

    # -- DSP thread --------------------------------------------------------

    def _rebuild(self, mode: str, bandwidth_hz: float | None) -> None:
        self._demod = demod.create(
            mode,
            float(self.sample_rate),
            bandwidth_hz=bandwidth_hz,
            volume=self.volume,
            squelch_dbfs=self.squelch_dbfs,
        )

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command()

    def _run(self) -> None:
        reader, sink = self.reader, self.sink
        if reader is None or sink is None:
            return
        interval = 1.0 / SPECTRUM_HZ
        last_frame = 0.0

        while not self._stop.is_set():
            self._drain_commands()
            raw = reader.ring.read(DSP_BLOCK_BYTES, timeout=0.5)
            if raw is None:
                if reader.last_error:
                    self.last_error = reader.last_error
                    break
                continue

            iq = convert.to_complex(raw)
            sink.write(self._demod.process(iq))

            now = time.perf_counter()
            if now - last_frame < interval:
                continue
            last_frame = now
            spectrum_db = self._spectrum.process(iq)
            if spectrum_db.size == 0:
                continue
            squelch = self._demod.squelch
            self._mailbox.put(
                DisplayFrame(
                    spectrum_db=spectrum_db,
                    center_hz=float(self.center_hz),
                    sample_rate=float(self.sample_rate),
                    bin_width_hz=self._spectrum.bin_width_hz,
                    channel_power_dbfs=self._demod.channel_power_dbfs,
                    bandwidth_hz=self._demod.bandwidth_hz,
                    squelch_open=None if squelch is None else squelch.is_open,
                    audio_latency_s=sink.latency_s,
                    underruns=sink.underruns,
                    ring_overruns=reader.ring.overruns,
                )
            )


__all__ = ["DisplayFrame", "Engine", "Mailbox"]
