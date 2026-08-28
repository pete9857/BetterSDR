"""FM stereo: recovering L and R from the 38 kHz difference channel.

A stereo broadcast is not two signals. It is the *sum* L+R where a mono
receiver expects the audio, plus the *difference* L-R amplitude-modulated on a
suppressed 38 kHz subcarrier, plus a 19 kHz pilot tone that is the only clue
to where 38 kHz was when the transmitter put it there. Adding and subtracting
the two recovers the channels, and a mono receiver hears the sum and is none
the wiser - which is why stereo was adopted in 1961 without obsoleting a
single radio.

Three things about this are easy to get wrong:

* **The subcarrier is suppressed, so its phase has to be inferred.** Getting
  it wrong by 90 degrees does not distort the difference channel, it deletes
  it: what comes back is scaled by the cosine of the phase error. There is no
  gradual degradation to notice on the way.
* **A filter's group delay on the pilot is a phase error, not a delay.** The
  bandpass that isolates the pilot needs a couple of kHz of transition either
  side, which at a 240 kHz multiplex rate is a few hundred taps and a
  fraction of a millisecond. A reference 0.6 ms late at 19 kHz is eleven
  cycles wrong. `delay` is what fixes it - the multiplex is held back to meet
  its own pilot - and the sum has to be held back by exactly the same amount
  or the two channels arrive at different times.
* **A mono station is not silent at 19 kHz**, it is noisy there, so the pilot
  cannot be found by asking whether anything is present. It is found by
  comparing the pilot band against a guard band immediately below it, where
  nothing is ever allocated: the audio stops at 15 kHz and the difference
  channel starts at 23.
"""

from __future__ import annotations

import numpy as np

from .filters import FirDecimator

PILOT_HZ = 19_000.0
# Half-width of the filter around the pilot. Wide enough for a pilot from a
# transmitter whose clock is not ours - measured at 98 ppm on this machine
# while building the RDS decoder, which is 1.9 Hz here - and narrow enough to
# keep out the 15 kHz top of the audio and the 23 kHz bottom of the
# difference channel.
PILOT_HALF_WIDTH_HZ = 1_200.0
# Immediately below the pilot and above the audio: allocated to nothing on
# any station, which is what makes it a fair measure of the noise the pilot
# has to be found against.
GUARD_CENTRE_HZ = 16_800.0
# Taps per phase in both bandpasses. At 240 kHz that is 289 taps and a group
# delay of 144 samples - 0.6 ms, which the jitter buffer never notices.
PILOT_TAPS = 288
# Pilot-to-guard power ratio at which a station is called stereo, and the
# lower one at which it stops being. The hysteresis stops a station at the
# edge of reception flickering the indicator.
LOCK_DB = 9.0
UNLOCK_DB = 6.0
# How fast the pilot amplitude estimate follows, as a time constant. The pilot
# is a constant-amplitude tone, so this only has to track fading; making it
# fast would let noise on the pilot modulate the separation.
LEVEL_TAU_S = 0.2
# The lowest multiplex rate that still carries the difference channel, which
# occupies 23-53 kHz. Nyquist, with room for an anti-alias filter to roll off.
MIN_MPX_RATE_HZ = 120_000.0


class StereoDecoder:
    """Split a multiplex into its sum and its difference channel.

    `process` is handed the multiplex exactly as the discriminator produced
    it, and returns the *delayed* sum alongside the difference. Returning both
    is the whole reason this is a transforming stage rather than a passive tap
    like the RDS receiver: the delay that aligns the multiplex with its own
    pilot has to reach the sum as well, and the only way to guarantee that is
    to make it impossible to take one without the other.
    """

    def __init__(self, sample_rate: float) -> None:
        if sample_rate < MIN_MPX_RATE_HZ:
            raise ValueError(
                f"multiplex rate {sample_rate:.0f} Hz is below the "
                f"{MIN_MPX_RATE_HZ:.0f} Hz the difference channel needs"
            )
        self.sample_rate = float(sample_rate)
        self.enabled = True
        # What the last block measured, for the indicator and for the log.
        self.pilot_db = -60.0
        self.locked = False

        self._pilot = FirDecimator.bandpass(
            1,
            PILOT_HZ - PILOT_HALF_WIDTH_HZ,
            PILOT_HZ + PILOT_HALF_WIDTH_HZ,
            self.sample_rate,
            taps_per_phase=PILOT_TAPS,
        )
        self._guard = FirDecimator.bandpass(
            1,
            GUARD_CENTRE_HZ - PILOT_HALF_WIDTH_HZ,
            GUARD_CENTRE_HZ + PILOT_HALF_WIDTH_HZ,
            self.sample_rate,
            taps_per_phase=PILOT_TAPS,
        )
        # An odd tap count, so the delay is a whole number of samples and
        # there is no half-sample skew to explain away later.
        self.delay = (self._pilot.taps.size - 1) // 2
        self._level = 0.0
        self._history = np.zeros(self.delay, dtype=np.float32)

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self._pilot.reset()
        self._guard.reset()
        self._history = np.zeros(self.delay, dtype=np.float32)
        self._level = 0.0
        self.locked = False
        self.pilot_db = -60.0

    # -- streaming ---------------------------------------------------------

    def _delayed(self, mpx: np.ndarray) -> np.ndarray:
        """`mpx` held back by the pilot filter's own group delay."""
        joined = np.concatenate((self._history, mpx))
        self._history = joined[joined.size - self.delay :].copy()
        return joined[: mpx.size]

    def process(self, mpx: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the delayed sum, and the difference when a pilot is locked.

        The difference is `None` rather than zeros when there is no pilot, so
        a caller can tell "this station is mono" apart from "this station is
        stereo and both channels happen to be playing the same thing".
        """
        mpx = np.asarray(mpx, dtype=np.float32)
        if mpx.size == 0:
            return mpx, None

        pilot = self._pilot.process(mpx)
        delayed = self._delayed(mpx)
        self._track(pilot, self._guard.process(mpx))
        if not (self.locked and self.enabled):
            return delayed, None

        # `pilot` is analytic, so squaring doubles its frequency: the 19 kHz
        # tone becomes the 38 kHz subcarrier the transmitter suppressed, with
        # the phase it had when it was suppressed. Normalising by a smoothed
        # amplitude rather than the instantaneous one keeps noise on the pilot
        # out of the audio - the reference has to carry phase and nothing else.
        #
        # The minus sign is the analytic signal's, not a fudge. The pilot is
        # transmitted as a sine, and the analytic form of a sine is -j.e^{jwt},
        # so squaring it gives -e^{2jwt} and the imaginary part comes back as
        # -sin(2wt). Left in, the difference channel arrives inverted and the
        # receiver plays the whole broadcast with its channels swapped - which
        # measures as perfect separation and sounds subtly wrong.
        reference = -2.0 * pilot.real * pilot.imag / max(self._level, 1e-12)
        # Twice, because a product detector recovers half the amplitude of a
        # suppressed-carrier signal - the same factor single sideband needs.
        return delayed, (2.0 * delayed * reference).astype(np.float32)

    def _track(self, pilot: np.ndarray, guard: np.ndarray) -> None:
        """Follow the pilot's amplitude, and decide whether it is really there."""
        pilot_power = float(np.mean(np.abs(pilot) ** 2))
        guard_power = float(np.mean(np.abs(guard) ** 2))
        self.pilot_db = 10.0 * np.log10(
            max(pilot_power, 1e-20) / max(guard_power, 1e-20)
        )
        self.locked = self.pilot_db > (UNLOCK_DB if self.locked else LOCK_DB)

        # A time constant in seconds, not in blocks: a block is 27 ms at
        # 2.4 MS/s and 273 ms at 240 kS/s, and every rate-dependent constant
        # in this app that was written per block has turned out to be a bug.
        alpha = 1.0 - np.exp(-pilot.size / (self.sample_rate * LEVEL_TAU_S))
        if self._level == 0.0:
            self._level = pilot_power
        else:
            self._level += alpha * (pilot_power - self._level)


__all__ = [
    "GUARD_CENTRE_HZ",
    "LOCK_DB",
    "MIN_MPX_RATE_HZ",
    "PILOT_HZ",
    "StereoDecoder",
]
