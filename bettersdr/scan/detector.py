"""Finding signals in a spectrum, and deciding which ones are real.

Two jobs live here, and the second is the one that decides whether the app
feels trustworthy. Finding bumps above a threshold is easy; the hard part is
not showing the user a list that reshuffles itself every second. A noise peak
that clears the threshold once is not a signal, so a detection has to survive
several sweeps before it is worth a line on screen.

The threshold is measured against a noise floor that follows the *shape* of
the band rather than a single number for the whole span. Real spectrum is not
flat: the front end rolls off, some ranges are full of hash and others are
dead quiet, and a flat threshold either floods one end with phantoms or goes
blind at the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# How wide a slice the floor is estimated over. Wider than any signal we expect
# to find, so a strong station cannot lift its own noise floor and hide itself,
# but narrow enough to follow the band's shape.
DEFAULT_FLOOR_KERNEL_HZ = 500_000.0
# A low percentile rather than the mean: in a busy band the mean sits inside
# the stations, and everything weaker disappears underneath it.
DEFAULT_FLOOR_PERCENTILE = 20.0

# How far above the floor a bin has to be. Exposed in the UI as a three-position
# Sensitivity control; this is the middle one.
DEFAULT_THRESHOLD_DB = 8.0
SENSITIVITY_DB = {"low": 12.0, "normal": DEFAULT_THRESHOLD_DB, "high": 5.0}

# A single bin over threshold is noise. Two adjacent bins is the narrowest
# thing a real transmitter can be.
DEFAULT_MIN_WIDTH_BINS = 2
# Gaps this small inside one signal get bridged. FM stations dip in the middle
# when the programme is quiet, and splitting one station into two entries is
# far more confusing than merging two neighbours into one.
DEFAULT_GAP_BINS = 2

# Two detections closer than this are treated as one signal seen twice, which
# is what the sweeper's overlapping steps produce at every tile boundary. It
# is generous because the same signal genuinely does measure differently in
# two windows, with different amounts of it above the local floor in each.
DEFAULT_TOLERANCE_HZ = 25_000.0

# How close two sightings must be, on separate passes, to count as the same
# signal. Far tighter than the merge tolerance, and for a different reason.
#
# The persistence gate only rejects noise if a noise peak is unlikely to find
# a partner in the previous pass. At 586 Hz per bin, 25 kHz is a window 85
# bins wide: in a quiet band with a few dozen noise peaks per sweep, most of
# them land near where some other one landed last time, and the gate passes
# nearly everything. Off air that turned an empty airband into eighty stable
# "signals". A real transmitter's centroid repeats within a bin or two, so a
# few kilohertz is all the room it needs.
PERSISTENCE_TOLERANCE_HZ = 4_000.0


@dataclass(frozen=True)
class Detection:
    """A bump above the noise floor, before anything has decided what it is."""

    center_hz: float
    bandwidth_hz: float
    peak_hz: float
    peak_dbfs: float
    floor_dbfs: float
    truncated: bool = False

    @property
    def snr_db(self) -> float:
        return self.peak_dbfs - self.floor_dbfs

    @property
    def start_hz(self) -> float:
        return self.center_hz - self.bandwidth_hz / 2.0

    @property
    def end_hz(self) -> float:
        return self.center_hz + self.bandwidth_hz / 2.0


def noise_floor_curve(
    spectrum_db: np.ndarray,
    bin_width_hz: float,
    kernel_hz: float = DEFAULT_FLOOR_KERNEL_HZ,
    percentile: float = DEFAULT_FLOOR_PERCENTILE,
) -> np.ndarray:
    """A per-bin noise floor that follows the band's shape.

    Estimated on overlapping slices rather than with a true sliding-window
    percentile, which would cost a comparison per bin per kernel tap and turn
    a full-spectrum sweep from seconds into a minute. The slices step by half a
    kernel, so no signal can sit at a slice boundary in every slice covering it.
    """
    bins = spectrum_db.size
    if bins == 0:
        return np.zeros(0, dtype=np.float32)

    kernel = max(8, int(round(kernel_hz / max(bin_width_hz, 1e-9))))
    if kernel >= bins:
        return np.full(bins, np.percentile(spectrum_db, percentile), dtype=np.float32)

    stride = max(1, kernel // 2)
    starts = list(range(0, bins - kernel + 1, stride))
    if starts[-1] + kernel < bins:
        starts.append(bins - kernel)

    # A loop over slices, not over samples: a handful of iterations whatever
    # the FFT size, so the "no Python loops over samples" rule still holds.
    values = np.array(
        [np.percentile(spectrum_db[s : s + kernel], percentile) for s in starts]
    )
    centres = np.array([s + kernel / 2.0 - 0.5 for s in starts])
    return np.interp(np.arange(bins), centres, values).astype(np.float32)


def _runs(mask: np.ndarray, gap_bins: int) -> list[tuple[int, int]]:
    """Contiguous True spans of `mask`, short gaps bridged. Ends are exclusive."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    steps = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(steps == 1)
    ends = np.flatnonzero(steps == -1)

    merged: list[tuple[int, int]] = []
    for start, end in zip(starts, ends, strict=True):
        if merged and start - merged[-1][1] <= gap_bins:
            merged[-1] = (merged[-1][0], int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def detect(
    spectrum_db: np.ndarray,
    bin_width_hz: float,
    center_hz: float = 0.0,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_width_bins: int = DEFAULT_MIN_WIDTH_BINS,
    gap_bins: int = DEFAULT_GAP_BINS,
    floor_kernel_hz: float = DEFAULT_FLOOR_KERNEL_HZ,
) -> list[Detection]:
    """Every signal standing `threshold_db` above the local noise floor.

    `spectrum_db` is an fftshifted dBFS spectrum as `psd.Spectrum.process`
    returns it, and `center_hz` is what the dongle was tuned to, so detections
    come back in absolute frequency ready to hand to the classifier.
    """
    bins = spectrum_db.size
    if bins == 0:
        return []

    floor = noise_floor_curve(spectrum_db, bin_width_hz, kernel_hz=floor_kernel_hz)
    mask = spectrum_db > floor + threshold_db

    offsets = (np.arange(bins) - bins // 2) * bin_width_hz
    power = 10.0 ** (np.asarray(spectrum_db, dtype=np.float64) / 10.0)
    floor_power = 10.0 ** (np.asarray(floor, dtype=np.float64) / 10.0)
    # Weight the centroid by power *above* the floor. Including the floor drags
    # every centroid towards the middle of its run, which for an asymmetric
    # signal is not where the transmitter is.
    excess = np.maximum(power - floor_power, 0.0)

    found: list[Detection] = []
    for start, end in _runs(mask, gap_bins):
        if end - start < min_width_bins:
            continue
        span = slice(start, end)
        weights = excess[span]
        total = float(weights.sum())
        if total <= 0.0:
            continue
        centroid = float(np.dot(offsets[span], weights) / total)
        peak = start + int(np.argmax(spectrum_db[span]))
        found.append(
            Detection(
                center_hz=center_hz + centroid,
                bandwidth_hz=(end - start) * bin_width_hz,
                peak_hz=center_hz + float(offsets[peak]),
                peak_dbfs=float(spectrum_db[peak]),
                floor_dbfs=float(np.median(floor[span])),
                # A run touching either edge is a signal the window cut in
                # half. The sweeper overlaps its steps so a neighbouring step
                # has the whole thing, and uses this flag to prefer it.
                truncated=start == 0 or end == bins,
            )
        )
    return found


class Persistence:
    """Requires a signal to appear in several consecutive sweeps.

    Without this the discovery list flickers: noise peaks arrive and leave on
    every pass, entries reorder themselves, and the app looks like it is
    guessing. Two sightings out of three kills that while still picking up a
    station on the second pass, about a second after the user clicks Scan.
    """

    def __init__(
        self,
        needed: int = 2,
        window: int = 3,
        tolerance_hz: float = PERSISTENCE_TOLERANCE_HZ,
    ) -> None:
        if needed < 1 or window < needed:
            raise ValueError(f"need 1 <= needed <= window, got {needed}/{window}")
        self.needed = int(needed)
        self.window = int(window)
        self.tolerance_hz = float(tolerance_hz)
        self._history: list[list[Detection]] = []

    @property
    def sweeps(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Record one sweep and return the detections that have earned a place.

        What comes back is the newest sighting of each, so a signal that has
        drifted or changed strength reports its current state rather than the
        one it had when it first qualified.
        """
        self._history.append(list(detections))
        del self._history[: -self.window]

        confirmed = []
        for candidate in detections:
            sightings = sum(
                any(
                    abs(other.center_hz - candidate.center_hz) <= self.tolerance_hz
                    for other in sweep
                )
                for sweep in self._history
            )
            if sightings >= self.needed:
                confirmed.append(candidate)
        return confirmed


def merge_nearby(
    detections: Iterable[Detection], tolerance_hz: float = DEFAULT_TOLERANCE_HZ
) -> list[Detection]:
    """Collapse duplicates of one signal seen by two overlapping sweep steps.

    Keeps the untruncated sighting where there is one: the step that caught the
    signal whole measured its width correctly, and the step that caught half of
    it did not.
    """
    out: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.center_hz):
        if not out or abs(detection.center_hz - out[-1].center_hz) > tolerance_hz:
            out.append(detection)
            continue
        previous = out[-1]
        if previous.truncated and not detection.truncated:
            out[-1] = detection
        elif previous.truncated == detection.truncated:
            out[-1] = max(previous, detection, key=lambda d: d.peak_dbfs)
    return out


__all__ = [
    "DEFAULT_THRESHOLD_DB",
    "DEFAULT_TOLERANCE_HZ",
    "PERSISTENCE_TOLERANCE_HZ",
    "SENSITIVITY_DB",
    "Detection",
    "Persistence",
    "detect",
    "merge_nearby",
    "noise_floor_curve",
]
