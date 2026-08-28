"""Audio AGC: hold the loudness steady without winding the noise up with it.

Automatic gain control is the difference between an SDR that is pleasant to
listen to and one that has you reaching for the volume knob every time the
band changes. Two behaviours separate a good one from a bad one, and both are
parameters here rather than assumptions:

* **Threshold.** Below it the gain stops rising. Without a threshold, an AGC
  answers a silent channel by amplifying the noise floor up to full loudness,
  which is the commonest complaint about naive implementations - the squelch
  tail becomes a roar.
* **Hang.** A speaker pausing between words is not a reason to wind the gain
  up and then slam it back down on the next syllable. Holding the gain across
  the gap is what stops the audio breathing.

`slope_db` is the third SDR# control and the least obvious: it is how many dB
the *output* is allowed to rise for every 10 dB the *input* rises above the
threshold. Zero is a perfectly flat output, ten is no action at all, and
something small in between keeps a hint of the original dynamics, which is
what most people prefer on music.

The gain is recomputed on a coarse grid rather than per sample. 750 Hz is
1.3 ms - far faster than any attack a listener can resolve - and it means the
recursive part of the loop runs 750 times a second instead of 48,000, so the
rule that nothing in `dsp/` loops over samples still holds. The gain is then
interpolated in dB across the block, so there is no step at a control boundary
and no zipper noise.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CONTROL_RATE_HZ = 750.0
# Quieter than this and the input is silence as far as the detector cares.
FLOOR_DBFS = -120.0


def _coefficient(time_ms: float, rate_hz: float) -> float:
    """One-pole coefficient reaching 63% of the way in `time_ms`."""
    seconds = max(1e-4, float(time_ms) / 1000.0)
    return float(1.0 - np.exp(-1.0 / (rate_hz * seconds)))


def _linear(gain_db):
    return 10.0 ** (np.asarray(gain_db, dtype=np.float32) / 20.0)


class Agc:
    """Level-riding gain with threshold, slope, hang and separate ramps."""

    def __init__(
        self,
        sample_rate: float,
        target_dbfs: float = -12.0,
        threshold_dbfs: float = -55.0,
        slope_db: float = 0.0,
        attack_ms: float = 10.0,
        decay_ms: float = 500.0,
        use_hang: bool = True,
        hang_ms: float = 250.0,
        max_gain_db: float = 45.0,
        control_rate_hz: float = DEFAULT_CONTROL_RATE_HZ,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.target_dbfs = float(target_dbfs)
        self.threshold_dbfs = float(threshold_dbfs)
        self.slope_db = float(slope_db)
        self.use_hang = bool(use_hang)
        self.max_gain_db = float(max_gain_db)
        # Kept as given as well as as coefficients, so a caller changing one
        # setting can rebuild without having to remember the other five.
        self.attack_ms = float(attack_ms)
        self.decay_ms = float(decay_ms)
        self.hang_ms = float(hang_ms)
        self.control_rate_hz = float(control_rate_hz)

        self.hop = max(1, int(round(self.sample_rate / float(control_rate_hz))))
        control_rate = self.sample_rate / self.hop
        self._attack = _coefficient(attack_ms, control_rate)
        self._decay = _coefficient(decay_ms, control_rate)
        self._hang_steps = max(0, int(round(hang_ms * control_rate / 1000.0)))

        self._gain_db = 0.0
        self._hang = 0
        # Samples of an incomplete control step, held back until the step
        # finishes and its gain is known. Emitting them early would mean
        # holding the gain flat over the tail of every block, which makes the
        # output depend on how the caller chopped the stream up - and a stage
        # whose result moves with the block size cannot be checked against a
        # one-shot run of the same audio. The cost is at most one control step
        # of latency, 1.3 ms.
        self._pending = np.zeros(0, dtype=np.float32)

    # -- state -------------------------------------------------------------

    @property
    def gain_db(self) -> float:
        """The gain currently being applied, for display."""
        return self._gain_db

    @property
    def ceiling_db(self) -> float:
        """The most gain the threshold allows, which is where silence sits."""
        return min(self.max_gain_db, self.target_dbfs - self.threshold_dbfs)

    def reset(self) -> None:
        self._gain_db = 0.0
        self._hang = 0
        self._pending = np.zeros(0, dtype=np.float32)

    # -- the control law ---------------------------------------------------

    def target_gain_db(self, level_dbfs: float) -> float:
        """The gain that would put `level_dbfs` where we want it.

        One straight line in dB rather than a branch per region: at and below
        the threshold it is flat at the ceiling, and above it the output rises
        at `slope_db` per 10 dB of input.
        """
        above = max(0.0, level_dbfs - self.threshold_dbfs)
        slope = float(np.clip(self.slope_db / 10.0, 0.0, 1.0))
        gain = (self.target_dbfs - self.threshold_dbfs) - (1.0 - slope) * above
        return float(np.clip(gain, -self.max_gain_db, self.ceiling_db))

    def _advance(self, peak: float) -> None:
        """One control step: move the gain towards its target."""
        level = 20.0 * float(np.log10(max(peak, 1e-12)))
        target = self.target_gain_db(max(level, FLOOR_DBFS))
        if target < self._gain_db:
            # Something got louder. Come down quickly - this is the ramp that
            # stops a strong signal arriving as a blast - and arm the hang so
            # a gap in the speech cannot undo it.
            self._gain_db += (target - self._gain_db) * self._attack
            self._hang = self._hang_steps if self.use_hang else 0
        elif self._hang > 0:
            self._hang -= 1
        else:
            self._gain_db += (target - self._gain_db) * self._decay

    # -- streaming ---------------------------------------------------------

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply the gain to a block, returning whole control steps of audio.

        The output is shorter than the input by however much of a control step
        was left over, and that remainder comes out at the front of the next
        call. Callers downstream of this already tolerate a varying block
        length - the demodulators do the same thing for the same reason.

        Stereo blocks arrive as `(frames, 2)` and get **one** gain, taken from
        the louder channel and applied to both. Two independent gain riders
        would each chase their own channel, and the stereo image would drift
        about between them - which is a far stranger fault than either channel
        simply being a decibel off its ideal level.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if self._pending.shape[1:] != audio.shape[1:]:
            self._pending = np.zeros((0, *audio.shape[1:]), dtype=np.float32)
        buffered = (
            np.concatenate((self._pending, audio)) if self._pending.size else audio
        )
        steps = buffered.shape[0] // self.hop
        if steps == 0:
            self._pending = buffered.copy()
            return np.zeros((0, *audio.shape[1:]), dtype=np.float32)

        usable = steps * self.hop
        self._pending = buffered[usable:].copy()
        block = buffered[:usable]

        # The peaks are one reshaped `max`, so only the recursion below is
        # scalar work - 750 iterations per second of audio, not 48,000.
        peaks = np.abs(block).reshape(steps, -1).max(axis=1)
        entry_db = self._gain_db
        gains: list[float] = []
        for peak in peaks.tolist():
            self._advance(peak)
            gains.append(self._gain_db)

        # Interpolated in dB rather than in amplitude, because a dB ramp is
        # what the ear hears as a smooth change. Each step ends on its last
        # sample, and the ramp into the first step starts from where the
        # previous call left off - index -1, immediately before this block.
        ends = np.arange(steps, dtype=np.float64) * self.hop + (self.hop - 1)
        curve = np.interp(
            np.arange(usable, dtype=np.float64),
            np.concatenate(([-1.0], ends)),
            np.asarray([entry_db, *gains], dtype=np.float64),
        )
        gain = _linear(curve)
        if block.ndim > 1:
            gain = gain.reshape(-1, *([1] * (block.ndim - 1)))
        return (block * gain).astype(np.float32)


__all__ = ["DEFAULT_CONTROL_RATE_HZ", "Agc"]
