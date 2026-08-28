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
from .frontend import GainChoice, choose_gain, safe_center_hz, safe_sample_rate
from .reader import Reader, read_bytes_for

# How much of the ring to take per pass. 64 KB is ~14 ms at 2.4 MS/s, which
# keeps the meter and spectrum responsive without making the loop spin.
DSP_BLOCK_BYTES = 65_536
# As with the reader's read size, what matters is the span of time a block
# covers rather than its size in bytes. 64 KB is 14 ms at 2.4 MS/s and 137 ms
# at 240 kS/s, and handing the audio sink 137 ms of audio at a time - against
# a 150 ms target buffer - underruns on every block.
DSP_BLOCK_SECONDS = DSP_BLOCK_BYTES / 2 / 2_400_000


def dsp_block_bytes_for(sample_rate_hz: float) -> int:
    raw = int(sample_rate_hz * 2 * DSP_BLOCK_SECONDS)
    return max(4_096, (raw // 512) * 512)
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

        self._block_bytes = dsp_block_bytes_for(sample_rate)
        self._carry = np.empty(0, dtype=np.complex64)
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
        self._resume_rate: int | None = None
        # Set the instant a scan is asked for, cleared when the sweep ends.
        # See `scanning` for why this is not just "is there a sweeper".
        self._scan_wanted = threading.Event()
        # Set while a gain measurement is queued on the reader thread, so two
        # callers asking at once produce one probe rather than two.
        self._gain_pending = threading.Event()
        # Whether audio is currently parked for the duration of one. See
        # `_run`; the DSP thread owns this and nothing else touches it.
        self._sink_parked = False

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

        self.reader = Reader(
            self.device, block_bytes=read_bytes_for(self.sample_rate)
        )
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
        # Clamped here rather than trusted from the caller: this is the one
        # choke point every tuning path goes through, and a frequency the
        # dongle cannot reach is rejected on the reader thread where nobody
        # can see it. See `frontend.safe_center_hz`.
        self.center_hz = safe_center_hz(center_hz)
        if self.reader is not None:
            self.reader.tune(self.center_hz)

    def auto_gain(self) -> None:
        """Re-pick the RF gain for whatever the radio is pointed at now.

        Gain used to be chosen once, in `start`, for whichever band the app
        happened to open on - and the FM band is the loudest thing the dongle
        ever sees, so that one measurement stepped the tuner down near the
        bottom of its range. Tuning from there to the AM band kept the FM
        setting and the station was barely audible; the fix from the user's
        side was to find the RF gain control and wind it up by hand, which is
        exactly the knowledge this app exists to not require.

        Runs on the reader thread because `choose_gain` reads from the device
        directly and the reader owns it. That blocks capture for the length of
        the probe, so it belongs at the moments the front end genuinely
        changed - a new band, a new window width - and not on every retune.
        """
        if self.reader is None or self._gain_pending.is_set():
            # Already queued and not yet run. A band change asks for this and
            # so does the window change it triggers, and the probe is the one
            # thing on the reader thread long enough to matter: measured at
            # ~85 ms on the AM band, where running it twice emptied the audio
            # buffer and cost 23 underruns for a single hop.
            return
        self._gain_pending.set()

        def command(device: Device) -> None:
            try:
                self.gain = choose_gain(device)
            finally:
                self._gain_pending.clear()

        self.reader.submit(command)

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

    def set_sample_rate(self, sample_rate_hz: int) -> None:
        """Change how wide a window the radio looks through.

        Narrowing it is how the HF bands become listenable at all - see
        `frontend.safe_sample_rate` - so this is a correctness control, not
        just a display preference. The request is clamped the same way
        everywhere else is, because a caller asking for a window that reaches
        0 Hz is asking for the upconverter's oscillator instead of a station.
        """
        rate = safe_sample_rate(self.center_hz, preferred_hz=int(sample_rate_hz))
        if rate == self.sample_rate:
            return
        self.sample_rate = rate
        if self.reader is not None:
            self.reader.set_sample_rate(rate)
            # After the rate command, so the probe measures through the new
            # window. Narrowing it is what moves the upconverter's oscillator
            # out of view, and the whole point of doing so is the headroom it
            # frees up - which is only collected by measuring again.
            self.auto_gain()
        self._commands.put(lambda: self._apply_sample_rate(rate))

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
        sample_rate_hz: int | None = None,
    ) -> None:
        """Sweep a range and report what is in it.

        Audio stops for the duration - the radio is somewhere else entirely
        for most of it - and resumes on the frequency it was tuned to before.
        """
        if self.reader is None or self.scanning:
            return
        # The bottom of the range is where the window is most likely to reach
        # below 0 Hz, so that is what the guard is asked about. Sweeping the
        # AM band at 2.4 MHz would otherwise report the upconverter's own
        # oscillator as the strongest station in Seattle.
        rate = safe_sample_rate(
            low_hz, preferred_hz=int(sample_rate_hz or self.sample_rate)
        )
        sweeper = Sweeper(
            low_hz,
            high_hz,
            float(rate),
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
        self._resume_rate = self.sample_rate
        self._sweeper = sweeper
        if int(sweeper.sample_rate) != self.sample_rate:
            self.sample_rate = int(sweeper.sample_rate)
            if self.reader is not None:
                self.reader.set_sample_rate(self.sample_rate)
            self._apply_sample_rate(self.sample_rate)
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
        if self._resume_rate is not None:
            # Before the retune, so the reader's queue puts the window back
            # the way it was and only then moves the tuner.
            if self._resume_rate != self.sample_rate:
                self.sample_rate = self._resume_rate
                if self.reader is not None:
                    self.reader.set_sample_rate(self.sample_rate)
                self._apply_sample_rate(self.sample_rate)
            self._resume_rate = None
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

    def _apply_sample_rate(self, rate: int) -> None:
        """Rebuild everything that was built around the old rate.

        The spectrum and the demodulator both bake the rate into their filter
        design, and `_publish_scan_frame` reports this spectrum's bin width
        alongside the sweeper's data - so letting the two disagree would put
        every frequency on the display wrong.
        """
        # The ring was emptied when the rate changed, so the sink is about to
        # run dry for as long as it takes to refill. That starvation is
        # expected, and letting it land in the underrun count is how the one
        # number that reports a real audio fault stops meaning anything -
        # measured at 25 underruns for a single hop from FM down to the AM
        # band. Same reasoning as `_begin_scan`; `start` re-primes with
        # silence and keeps the running total.
        if self.sink is not None:
            self.sink.stop()
        self._block_bytes = dsp_block_bytes_for(rate)
        # Samples captured at the old rate must not end up in a frame measured
        # at the new one.
        self._carry = np.empty(0, dtype=np.complex64)
        self._spectrum = Spectrum(
            fft_size=self._spectrum.fft_size, sample_rate=float(rate)
        )
        self._rebuild(self.mode, self._demod.bandwidth_hz)
        if self.sink is not None:
            self.sink.start()

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

            # A gain probe runs on the reader thread and reads the device
            # directly, so no samples are captured while it does. That is
            # brief on a quiet band - it accepts a high gain almost at once -
            # but on the FM band it walks most of the tuner's table two reads
            # at a time, measured at about 340 ms, which is twice the jitter
            # buffer. So audio is parked for the duration rather than left to
            # starve: an underrun count full of expected underruns cannot
            # report a real fault. Same reasoning as `_begin_scan`.
            probing = self._gain_pending.is_set()
            if probing != self._sink_parked:
                self._sink_parked = probing
                if probing:
                    sink.stop()
                else:
                    sink.start()

            raw = reader.ring.read(self._block_bytes, timeout=0.5)
            if raw is None:
                if reader.last_error:
                    self.last_error = reader.last_error
                    break
                continue

            iq = convert.to_complex(raw)
            sink.write(self._demod.process(iq))

            # The display FFT needs a whole frame's worth of samples. A block
            # is eight frames at 2.4 MS/s but less than one at the 240 kS/s
            # the AM band uses, so the remainder is carried rather than
            # dropped - without this the spectrum silently stops updating the
            # moment the window narrows, while the audio carries on fine.
            self._carry = (
                np.concatenate((self._carry, iq)) if self._carry.size else iq
            )
            if self._carry.size < self._spectrum.fft_size:
                continue

            now = time.perf_counter()
            if now - last_frame < interval:
                self._carry = self._carry[-self._spectrum.fft_size :]
                continue
            last_frame = now
            spectrum_db = self._spectrum.process(self._carry)
            self._carry = np.empty(0, dtype=np.complex64)
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
