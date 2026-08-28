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
from ..scan.classifier import Signal
from ..scan.detector import DEFAULT_THRESHOLD_DB
from ..scan.sweeper import (
    DEFAULT_PASSES,
    DEFAULT_SETTLE_S,
    Sweeper,
    SweepProgress,
    SweepResult,
)
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


@dataclass(frozen=True)
class ScanUpdate:
    """One update from a scan in progress, or the finished article.

    Carries the list found so far as well as the progress, so the discovery
    screen fills in as the sweep moves rather than staying empty until the end.
    `result` is set only on the last update.
    """

    progress: SweepProgress
    signals: tuple[Signal, ...]
    result: SweepResult | None = None

    @property
    def complete(self) -> bool:
        return self.result is not None


class _ReaderSource:
    """Presents the reader thread to a sweeper as something tunable.

    Retuning has to be ordered against the sample stream, which is the whole
    reason the reader takes commands rather than exposing the device: the
    command runs between two reads, so once it has run every byte that arrives
    afterwards is on the new frequency. What is already in the ring is not, and
    is dropped - measuring the previous step's spectrum at this step's
    frequency would put signals in the wrong place, which is the one mistake a
    scanner must not make.
    """

    def __init__(self, reader: Reader, settle_s: float = DEFAULT_SETTLE_S) -> None:
        self.reader = reader
        self.settle_s = float(settle_s)

    def tune(self, hz: int) -> None:
        retuned = threading.Event()

        def command(device: Device) -> None:
            device.center_freq = int(hz)
            device.reset_buffer()
            retuned.set()

        self.reader.submit(command)
        retuned.wait(timeout=1.0)
        if self.settle_s > 0:
            # The tuner's PLL is still moving for a moment after the register
            # write, so these samples are neither one frequency nor the other.
            time.sleep(self.settle_s)
        self.reader.ring.clear()

    def read(self, samples: int) -> np.ndarray | None:
        raw = self.reader.ring.read(samples * 2, timeout=2.0)
        return None if raw is None else convert.to_complex(raw)


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
        self._scan_mailbox: Mailbox[ScanUpdate] = Mailbox()
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Owned by the DSP thread once a scan starts. Listening and scanning
        # are the same thread taking turns, not two consumers of one ring:
        # two things draining the same buffer would each get half the samples.
        self._sweeper: Sweeper | None = None
        self._scan_source: _ReaderSource | None = None
        self._resume_hz: int | None = None
        # Set the instant a scan is asked for, cleared when the sweep ends.
        # See `scanning` for why this is not just "is there a sweeper".
        self._scan_wanted = threading.Event()

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

    # -- scanning ----------------------------------------------------------

    @property
    def scanning(self) -> bool:
        """Whether a scan is wanted, not whether one has started yet.

        Deliberately not `self._sweeper is not None`. Starting a scan queues a
        command for the DSP thread, so the sweeper appears a fraction of a
        second after the caller asks for it - and a view polling at 20 Hz sees
        "not scanning" in that gap and concludes the scan has already finished.
        Off air that left the discovery screen stuck on "Listening around
        108.7 MHz" after a completed sweep, because its one chance to write the
        summary had been spent before the sweep began.
        """
        return self._scan_wanted.is_set()

    def scan_update(self) -> ScanUpdate | None:
        """The latest news from a scan, or the finished result. Never blocks."""
        return self._scan_mailbox.peek()

    def start_scan(
        self,
        low_hz: float,
        high_hz: float,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        passes: int = DEFAULT_PASSES,
    ) -> None:
        """Sweep a range and report what is in it.

        Audio stops for the duration - the radio is somewhere else entirely
        for most of it - and resumes on the frequency it was tuned to before.
        """
        if self.reader is None or self.scanning:
            return
        sweeper = Sweeper(
            low_hz,
            high_hz,
            float(self.sample_rate),
            fft_size=self._spectrum.fft_size,
            threshold_db=threshold_db,
            passes=passes,
        )
        self._scan_wanted.set()
        self._scan_mailbox.put(ScanUpdate(progress=sweeper.progress, signals=()))
        self._commands.put(lambda: self._begin_scan(sweeper))

    def stop_scan(self) -> None:
        """Abandon a scan and go back to listening."""
        self._commands.put(self._end_scan)

    # -- DSP thread --------------------------------------------------------

    def _begin_scan(self, sweeper: Sweeper) -> None:
        if self.reader is None:
            return
        self._resume_hz = self.center_hz
        self._sweeper = sweeper
        self._scan_source = _ReaderSource(self.reader)
        if self.sink is not None:
            # Rather than let it starve for the length of the sweep: an
            # underrun count is how the audio path reports a real fault, and
            # filling it with expected ones would make it useless.
            self.sink.stop()

    def _end_scan(self) -> None:
        sweeper, self._sweeper = self._sweeper, None
        self._scan_source = None
        self._scan_wanted.clear()
        if sweeper is not None:
            self._scan_mailbox.put(
                ScanUpdate(
                    progress=sweeper.progress,
                    signals=sweeper.signals(),
                    result=sweeper.result(),
                )
            )
        if self._resume_hz is not None:
            self.tune(self._resume_hz)
            self._resume_hz = None
        if self.sink is not None:
            self.sink.start()

    def _scan_step(self) -> None:
        sweeper, source = self._sweeper, self._scan_source
        if sweeper is None or source is None:
            return
        if sweeper.complete:
            self._end_scan()
            return

        source.tune(sweeper.current_hz)
        iq = source.read(sweeper.dwell_samples)
        if iq is None:
            self._end_scan()
            return
        sweeper.feed(iq)
        self._publish_scan_frame(sweeper)
        self._scan_mailbox.put(
            ScanUpdate(progress=sweeper.progress, signals=sweeper.signals())
        )
        if sweeper.complete:
            self._end_scan()

    def _publish_scan_frame(self, sweeper: Sweeper) -> None:
        """Keep the spectrum and waterfall alive while the sweep runs.

        A frozen display for the length of a scan reads as a hang. This shows
        the window actually moving across the band, which is both honest and
        the best explanation of what scanning is that the app can give.
        """
        spectrum_db = sweeper.last_spectrum_db
        if spectrum_db is None or self.reader is None:
            return
        self._mailbox.put(
            DisplayFrame(
                spectrum_db=spectrum_db,
                center_hz=sweeper.last_center_hz,
                sample_rate=float(self.sample_rate),
                bin_width_hz=self._spectrum.bin_width_hz,
                channel_power_dbfs=float(np.max(spectrum_db)),
                bandwidth_hz=self._demod.bandwidth_hz,
                squelch_open=None,
                audio_latency_s=0.0,
                underruns=0 if self.sink is None else self.sink.underruns,
                ring_overruns=self.reader.ring.overruns,
            )
        )

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
            if self._sweeper is not None:
                self._scan_step()
                continue
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


__all__ = ["DisplayFrame", "Engine", "Mailbox", "ScanUpdate"]
