"""Stepping the tuner across a range and collecting what is there.

The dongle sees 2.4 MHz at a time, so covering a band means retuning, dwelling,
measuring, and moving on. Three details make the difference between a sweep
that finds real stations and one that invents them:

**Steps overlap by a quarter.** Only the middle of each window is trusted for
deciding *where* a signal is, but detection runs across the whole window, so a
signal sitting on a tile boundary is still measured whole by the step that has
it in the middle.

**The tuner is parked slightly off the tile centre.** Every step has one blind
spot: its own DC bin, which the per-frame DC removal in `dsp/psd.py` empties
along with anything transmitting exactly there. Removing DC is not optional -
the RTL2832U's offset would otherwise raise a phantom signal at the centre of
every single step - and the overlap does not rescue it either, because a signal
at one step's centre is 1.8 MHz from its neighbours and outside their windows
entirely. So the tuner is offset by `TUNE_OFFSET_HZ` from the tile it is
measuring: the tile boundaries stay where they were, and DC lands on a
frequency that is not a channel on any raster in the band plan. A transmitter
on a non-standard frequency can still land in the notch, which is why the
offset is a named constant rather than an accident.

**Every step is measured the same way as the picture on screen.** The sweep
uses `dsp/psd.py`, the same module and the same calibration the spectrum
display uses, so "the app found a signal I cannot see" cannot happen.

**Nothing is reported until it has been seen more than once.** A sweep is
repeated a few times and only signals that show up consistently survive. One
pass over the FM band takes about six hundred milliseconds, so three of them
still feels instant, and the list that appears does not then reshuffle itself.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..dsp.features import HD_REFERENCE_LO_HZ, HdRadio, detect_hd_radio
from ..dsp.psd import DEFAULT_FFT_SIZE, Spectrum
from .bandplan import DEFAULT_REGION
from .classifier import Shape, Signal, classify, measure_shape
from .detector import (
    DEFAULT_THRESHOLD_DB,
    DEFAULT_TOLERANCE_HZ,
    PERSISTENCE_TOLERANCE_HZ,
    Detection,
    Persistence,
    detect,
    merge_nearby,
)

# Fraction of each window thrown away at the edges when deciding which step a
# signal belongs to. Also the margin that lets a neighbouring step measure a
# signal this one has cut in half at its window edge.
DEFAULT_OVERLAP = 0.25
# How long to sit on each step. Long enough for a steady PSD average, short
# enough that the FM band finishes in well under a second.
DEFAULT_DWELL_S = 0.05
# The tuner's PLL needs a moment, and the first samples after a retune are
# still the old frequency. Discarded rather than measured.
DEFAULT_SETTLE_S = 0.005
# Passes per scan. Three, to match the persistence gate's window.
DEFAULT_PASSES = 3

# Beyond this fraction of the sample rate the analog front end is rolling off,
# so bins out there under-report and are not worth searching.
EDGE_GUARD = 0.45

# How far the tuner sits from the centre of the tile it is measuring, so that
# the dead DC bin never lands on a real channel. 37 kHz is a whole number of
# kilohertz for the log, and divides none of the rasters in the band plan -
# 5, 10, 12.5, 25 or 200 kHz - so no standard channel can land on it. Small
# next to the 1.2 MHz half-window, so the tile stays comfortably in view.
TUNE_OFFSET_HZ = 37_000


def usable_span(
    sample_rate: float,
    overlap: float = DEFAULT_OVERLAP,
    tune_offset_hz: float = TUNE_OFFSET_HZ,
) -> float:
    """How much spectrum one step can actually be held responsible for.

    Two limits, and which one binds depends on the rate. The first is the
    overlap, which is there to keep a signal off the filter roll-off. The
    second is easy to miss: the tuner sits `tune_offset_hz` away from the tile
    it is measuring, so the tile's lower edge is that much *further* out in
    the window than its width suggests, and past `EDGE_GUARD` a bin is in the
    analog front end's roll-off and under-reports.

    At 2.4 MS/s the offset is 1.5% of the window and this never binds. At the
    240 kS/s the AM band needs it is 15%, and without it the bottom 19 kHz of
    every tile was measured and then silently discarded for being too far from
    centre. KIRO on 710 kHz sat in exactly that sliver and did not appear in a
    scan of its own band.
    """
    by_overlap = sample_rate * (1.0 - overlap)
    by_guard = 2.0 * (sample_rate * EDGE_GUARD - abs(tune_offset_hz))
    return max(1.0, min(by_overlap, by_guard))


def plan_steps(
    low_hz: float,
    high_hz: float,
    sample_rate: float,
    overlap: float = DEFAULT_OVERLAP,
    tune_offset_hz: float = TUNE_OFFSET_HZ,
) -> tuple[int, ...]:
    """Tile centres covering `low_hz` to `high_hz`, with overlap between them.

    These are the stretches of spectrum each step is responsible for, not what
    the dongle is tuned to - the tuner sits `TUNE_OFFSET_HZ` away from each.

    The last step may reach past `high_hz`; detections are clipped to the range
    afterwards. Stopping short instead would leave a hole at the top of every
    band, which is exactly where the user would notice it.
    """
    if high_hz <= low_hz:
        raise ValueError(f"range must ascend, got {low_hz} to {high_hz}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    usable = usable_span(sample_rate, overlap, tune_offset_hz)
    count = max(1, math.ceil((high_hz - low_hz) / usable))
    if count == 1:
        # A band narrower than one window gets that window centred on it,
        # rather than parked with the band up against one edge. Weather Radio
        # is 150 kHz wide and would otherwise sit 900 kHz off centre, out where
        # the analog front end is already rolling off.
        return (int(round((low_hz + high_hz) / 2.0)),)
    first = low_hz + usable / 2.0
    return tuple(int(round(first + index * usable)) for index in range(count))


class SweepSource(Protocol):
    """Whatever the sweeper is stepping: a real reader, or a synthetic scene."""

    def tune(self, hz: int) -> None: ...

    def read(self, samples: int) -> np.ndarray | None: ...


@dataclass(frozen=True)
class SweepResult:
    """One completed scan: the picture, and the list of what was in it."""

    low_hz: float
    high_hz: float
    sample_rate: float
    frequencies: np.ndarray
    spectrum_db: np.ndarray
    signals: tuple[Signal, ...]
    passes: int
    duration_s: float

    @property
    def strongest(self) -> tuple[Signal, ...]:
        return tuple(sorted(self.signals, key=lambda s: s.snr_db, reverse=True))


@dataclass
class _Sighting:
    """A detection plus everything measured about it while the step was live."""

    detection: Detection
    shape: Shape
    hd: HdRadio | None = None


@dataclass
class SweepProgress:
    """What to show while a scan is running."""

    pass_index: int = 0
    passes: int = DEFAULT_PASSES
    step_index: int = 0
    steps: int = 1
    center_hz: int = 0
    found: int = 0

    @property
    def fraction(self) -> float:
        total = max(1, self.passes * self.steps)
        return min(1.0, (self.pass_index * self.steps + self.step_index) / total)

    @property
    def pass_number(self) -> int:
        """Which pass to show the user, counting from one and never past the end.

        `pass_index` reaches `passes` on completion, which is correct as an
        index and reads as "pass 4 of 3" the moment anyone displays it.
        """
        return min(self.pass_index + 1, self.passes)


class Sweeper:
    """A scan in progress: a state machine fed one dwell of IQ at a time.

    Deliberately has no threads and no device in it. The caller tunes to
    `current_hz`, hands over a block of IQ, and repeats until `complete`. That
    keeps the whole of the scan's logic testable against synthetic air, and
    lets the engine run it on the DSP thread it already owns.
    """

    def __init__(
        self,
        low_hz: float,
        high_hz: float,
        sample_rate: float,
        fft_size: int = DEFAULT_FFT_SIZE,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        overlap: float = DEFAULT_OVERLAP,
        dwell_s: float = DEFAULT_DWELL_S,
        passes: int = DEFAULT_PASSES,
        region: str = DEFAULT_REGION,
        detect_hd: bool = True,
        tune_offset_hz: float = TUNE_OFFSET_HZ,
    ) -> None:
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.sample_rate = float(sample_rate)
        self.threshold_db = float(threshold_db)
        self.overlap = float(overlap)
        self.dwell_s = float(dwell_s)
        self.tune_offset_hz = int(tune_offset_hz)
        self.passes = max(1, int(passes))
        self.region = region
        self.detect_hd = bool(detect_hd)

        self.steps = plan_steps(
            low_hz, high_hz, sample_rate, overlap, self.tune_offset_hz
        )
        self.tile_hz = usable_span(sample_rate, overlap, self.tune_offset_hz)
        self._spectrum = Spectrum(fft_size=fft_size, sample_rate=self.sample_rate)
        # One pass is one sweep as far as the persistence gate is concerned.
        self._persistence = Persistence(
            needed=min(2, self.passes), window=self.passes
        )

        self._pass_index = 0
        self._step_index = 0
        self._pass_sightings: list[_Sighting] = []
        self._confirmed: list[_Sighting] = []
        # The best view of each channel so far, keyed by what it was called and
        # where it snapped to. Survives across passes; see `_remember`.
        self._best: dict[int, Signal] = {}
        self._picture: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._started = time.perf_counter()
        self._finished: float | None = None

        # The most recent step's own spectrum, so a caller with a display can
        # show the sweep moving across the band instead of a frozen picture.
        # Kept rather than recomputed: the FFT has already been paid for.
        self.last_spectrum_db: np.ndarray | None = None
        self.last_center_hz: float = 0.0

    # -- what the caller needs to drive it ---------------------------------

    @property
    def dwell_samples(self) -> int:
        """Whole PSD frames, so the dwell divides evenly into what is measured."""
        wanted = int(self.sample_rate * self.dwell_s)
        frames = max(1, round(wanted / self._spectrum.fft_size))
        return frames * self._spectrum.fft_size

    @property
    def current_tile_hz(self) -> int:
        """Centre of the stretch of spectrum this step is responsible for."""
        return self.steps[min(self._step_index, len(self.steps) - 1)]

    @property
    def current_hz(self) -> int:
        """What to tune the dongle to: the tile centre, offset off-channel."""
        return self.current_tile_hz + self.tune_offset_hz

    @property
    def complete(self) -> bool:
        return self._pass_index >= self.passes

    @property
    def progress(self) -> SweepProgress:
        return SweepProgress(
            pass_index=self._pass_index,
            passes=self.passes,
            step_index=self._step_index,
            steps=len(self.steps),
            center_hz=self.current_hz,
            found=len(self._confirmed),
        )

    # -- the state machine -------------------------------------------------

    def feed(self, iq: np.ndarray) -> None:
        """Measure one dwell at `current_hz` and move to the next step."""
        if self.complete:
            return
        spectrum_db = self._spectrum.process(iq)
        if spectrum_db.size:
            self._measure(spectrum_db, float(self.current_hz), self.current_tile_hz)

        self._step_index += 1
        if self._step_index >= len(self.steps):
            self._end_pass()

    def _measure(
        self, spectrum_db: np.ndarray, center: float, tile_center: float
    ) -> None:
        """Find and measure everything in one step's dwell.

        `center` is where the dongle actually sat, which is what turns a bin
        index into a frequency. `tile_center` is the stretch this step owns,
        which is what stops two overlapping steps reporting the same station.
        """
        bin_width = self._spectrum.bin_width_hz
        self.last_spectrum_db = spectrum_db
        self.last_center_hz = center
        self._picture[int(tile_center)] = self._trusted_slice(
            spectrum_db, center, tile_center, bin_width
        )

        guard = self.sample_rate * EDGE_GUARD
        tile_low = tile_center - self.tile_hz / 2.0
        tile_high = tile_center + self.tile_hz / 2.0
        for detection in detect(
            spectrum_db, bin_width, center_hz=center, threshold_db=self.threshold_db
        ):
            offset = detection.center_hz - center
            # Owned by whichever step has it nearest the middle, so overlapping
            # steps do not each report the same station.
            if not tile_low <= detection.center_hz < tile_high:
                continue
            if abs(offset) > guard:
                continue
            if not self.low_hz <= detection.center_hz <= self.high_hz:
                continue
            self._pass_sightings.append(
                _Sighting(
                    detection=detection,
                    shape=measure_shape(spectrum_db, bin_width, center, detection),
                    hd=self._hd(spectrum_db, bin_width, offset, detection),
                )
            )

    def _hd(
        self,
        spectrum_db: np.ndarray,
        bin_width: float,
        offset: float,
        detection: Detection,
    ) -> HdRadio | None:
        """HD Radio verdict, when this looks like an FM broadcast station.

        Cheap enough to run on every candidate, but only meaningful for a wide
        signal in the broadcast band, and only answerable when the step window
        reaches far enough either side of it to see the shoulders.
        """
        if not self.detect_hd or detection.bandwidth_hz < 100_000:
            return None
        if not 87_900_000 <= detection.center_hz <= 108_100_000:
            return None
        try:
            return detect_hd_radio(spectrum_db, bin_width, carrier_offset_hz=offset)
        except ValueError:
            # Too close to the edge of this window to answer honestly.
            return None

    def _trusted_slice(
        self,
        spectrum_db: np.ndarray,
        center: float,
        tile_center: float,
        bin_width: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The middle of a step's window, which is the part the picture keeps.

        Clipped to the requested range as well, so the wide spectrum the user
        ends up looking at covers the band they asked for and not the extra
        megahertz the last step happened to reach into.
        """
        bins = spectrum_db.size
        offsets = (np.arange(bins) - bins // 2) * bin_width
        absolute = center + offsets
        keep = (
            (np.abs(absolute - tile_center) <= self.tile_hz / 2.0)
            & (absolute >= self.low_hz)
            & (absolute <= self.high_hz)
        )
        return (absolute[keep], spectrum_db[keep])

    def _absorb_hd_sidebands(self, sightings: list[_Sighting]) -> list[_Sighting]:
        """Drop an HD station's digital sidebands as entries of their own.

        Hybrid IBOC puts flat energy 129-198 kHz either side of the analog
        carrier, well clear of it, so the detector quite correctly finds three
        separate runs above the noise floor. They are one radio station, and
        listing them as three - the middle one FM, the outer two "digital
        signal" - would be both wrong and the sort of wrong that makes a
        beginner distrust the whole list.
        """
        carriers = [s for s in sightings if s.hd is not None and s.hd.present]
        if not carriers:
            return sightings
        keep = {id(s) for s in carriers}
        return [
            sighting
            for sighting in sightings
            if id(sighting) in keep
            or not any(
                abs(sighting.detection.center_hz - carrier.detection.center_hz)
                <= HD_REFERENCE_LO_HZ
                for carrier in carriers
            )
        ]

    def _end_pass(self) -> None:
        merged = merge_nearby(
            (sighting.detection for sighting in self._pass_sightings),
            tolerance_hz=DEFAULT_TOLERANCE_HZ,
        )
        by_id = {id(sighting.detection): sighting for sighting in self._pass_sightings}
        survivors = self._absorb_hd_sidebands([by_id[id(d)] for d in merged])

        by_id = {id(sighting.detection): sighting for sighting in survivors}
        confirmed = self._persistence.update(
            [sighting.detection for sighting in survivors]
        )
        self._confirmed = [by_id[id(detection)] for detection in confirmed]
        self._remember(self._confirmed)

        self._pass_sightings = []
        self._step_index = 0
        self._pass_index += 1
        if self.complete:
            self._finished = time.perf_counter()

    def _remember(self, sightings: list[_Sighting]) -> None:
        """Keep the most informative view of each channel across the whole scan.

        A single 50 ms dwell measures a signal's *instantaneous* width, and for
        anything carrying speech or music that is not its bandwidth. An FM
        station caught during a quiet moment collapses to a bare carrier a few
        kilohertz wide - 98.1 MHz measured 11 kHz at 51 dB off air - and gets
        reported as an anomaly rather than as the local station it is.

        Occupied bandwidth is bounded from below by modulation but not from
        above, so the widest view of a channel is the truest one. Keeping it
        also separates two things that look identical in any single sweep: a
        real station widens when somebody talks, and a spur never does.
        """
        for sighting in sightings:
            signal = classify(
                sighting.detection,
                shape=sighting.shape,
                hd=sighting.hd,
                region=self.region,
            )
            # Keyed on the channel alone, never on the channel *and* the label.
            # One transmitter classifies differently from pass to pass - an FM
            # station is "FM Radio" while somebody is talking and a bare
            # carrier in the gap - and keying on both put 90.3 MHz on screen
            # twice, once correctly and once as interference. Keyed on the
            # channel they compete instead, and the confident reading wins.
            key = int(round(signal.frequency_hz / PERSISTENCE_TOLERANCE_HZ))
            previous = self._best.get(key)
            if previous is None or self._rank(signal) > self._rank(previous):
                self._best[key] = signal

    @staticmethod
    def _rank(signal: Signal) -> tuple[float, float, float]:
        """Which of two views of the same channel to keep. Widest, then loudest."""
        return (signal.confidence, signal.bandwidth_hz, signal.snr_db)

    # -- the answer --------------------------------------------------------

    def signals(self) -> tuple[Signal, ...]:
        """Everything confirmed so far, classified, in frequency order.

        One channel is one entry. Two detections either side of a station's
        dip - or its skirts caught separately when it is weak - both snap to
        the same broadcast frequency, and 100.1 MHz appearing twice makes the
        list look broken however correct each half of it was. Keying on the
        snapped frequency is what makes them one thing: 100.0723 and 100.0323
        MHz are 40 kHz apart, well outside the detector's merge tolerance, and
        both are 100.1.
        """
        return tuple(sorted(self._best.values(), key=lambda s: s.frequency_hz))

    def result(self) -> SweepResult:
        frequencies, spectrum = self._stitch()
        finished = self._finished if self._finished is not None else time.perf_counter()
        return SweepResult(
            low_hz=self.low_hz,
            high_hz=self.high_hz,
            sample_rate=self.sample_rate,
            frequencies=frequencies,
            spectrum_db=spectrum,
            signals=self.signals(),
            passes=self._pass_index,
            duration_s=finished - self._started,
        )

    def _stitch(self) -> tuple[np.ndarray, np.ndarray]:
        """Join the trusted middles of every step into one wide spectrum."""
        if not self._picture:
            return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float32))
        centres = sorted(self._picture)
        frequencies = np.concatenate([self._picture[c][0] for c in centres])
        spectrum = np.concatenate([self._picture[c][1] for c in centres])
        order = np.argsort(frequencies, kind="stable")
        return frequencies[order], spectrum[order]


def run_sweep(
    sweeper: Sweeper,
    source: SweepSource,
    settle_s: float = DEFAULT_SETTLE_S,
    should_stop: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SweepResult:
    """Drive a sweeper to completion against a source. Synchronous.

    The engine runs its own loop so it can interleave a stop check with the
    rest of the DSP thread's work; this is the straight-line version, used by
    the tests and by anything scanning outside the GUI.
    """
    while not sweeper.complete:
        if should_stop is not None and should_stop():
            break
        source.tune(sweeper.current_hz)
        if settle_s > 0:
            sleep(settle_s)
        iq = source.read(sweeper.dwell_samples)
        if iq is None:
            break
        sweeper.feed(iq)
    return sweeper.result()


__all__ = [
    "usable_span",
    "DEFAULT_DWELL_S",
    "DEFAULT_OVERLAP",
    "DEFAULT_PASSES",
    "DEFAULT_SETTLE_S",
    "TUNE_OFFSET_HZ",
    "SweepProgress",
    "SweepResult",
    "SweepSource",
    "Sweeper",
    "plan_steps",
    "run_sweep",
]
