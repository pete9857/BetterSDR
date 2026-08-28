"""IQ correction: DC offset, quadrature imbalance, and offset tuning.

Everything here fixes a defect of the *receiver* rather than of the signal.
An RTL-SDR has three of them, and all three are visible on the spectrum
display before they are audible:

* A **DC offset** puts a permanent spike in the middle of the window. The
  scanner would happily report it as the strongest signal in the band, and a
  demodulator tuned to it hears a hum.
* **Quadrature imbalance** - the I and Q channels differing slightly in gain
  or not being exactly 90 degrees apart - makes every signal appear a second
  time, mirrored about the centre. Left uncorrected, an image of a strong
  station lands on top of a weak one and the app reports a station that is not
  there.
* **Offset tuning** does not fix a defect so much as step around one: park the
  tuner a little to the side and mix the wanted signal back to the middle in
  software, and the DC spike ends up somewhere harmless. It costs a complex
  multiply per sample and is the only complete cure for a spike sitting
  exactly on top of a weak carrier.

`dsp/psd.py` already removes DC per FFT frame for the *display*, which is a
different job: this removes it from the samples the demodulator sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def swap_iq(iq: np.ndarray) -> np.ndarray:
    """Exchange the I and Q channels, mirroring the spectrum about centre.

    Worth having as a control because some hardware and some recordings have
    the convention the other way round, and the symptom - every signal on the
    wrong side of centre, SSB decoding as the wrong sideband - is one that
    nothing else explains.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    return (iq.imag + 1j * iq.real).astype(np.complex64)


class DcRemover:
    """Subtract a slowly-tracked estimate of the receiver's own DC offset.

    Updated once per block rather than once per sample. DC is by definition
    the part of the signal that does not change, so a control loop running at
    the block rate is entirely sufficient, and it keeps this off the list of
    things that cost per-sample arithmetic on the hot path.

    The time constant is in seconds rather than in blocks for the usual
    reason: a block is 27 ms at 2.4 MS/s and 273 ms at 240 kS/s, and a filter
    specified in blocks would be ten times slower on the AM band than on FM.
    """

    def __init__(self, sample_rate: float, tau_s: float = 0.25) -> None:
        self.sample_rate = float(sample_rate)
        self.tau_s = float(tau_s)
        self._offset = complex(0.0, 0.0)
        self._seeded = False

    @property
    def offset(self) -> complex:
        return self._offset

    def reset(self) -> None:
        self._offset = complex(0.0, 0.0)
        self._seeded = False

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if iq.size == 0:
            return iq
        block_mean = complex(np.mean(iq))
        if not self._seeded:
            # Starting from zero would let a large offset through for the
            # first few blocks, which is exactly when the display is being
            # auto-ranged around it.
            self._offset = block_mean
            self._seeded = True
        else:
            span = iq.size / self.sample_rate
            alpha = 1.0 - float(np.exp(-span / max(1e-6, self.tau_s)))
            self._offset += alpha * (block_mean - self._offset)
        return (iq - np.complex64(self._offset)).astype(np.complex64)


@dataclass(frozen=True)
class ImbalanceEstimate:
    """The error the corrector has measured, in the terms it was made in.

    Reported as the fault rather than as the correction, so the number on
    screen is the one an Expert user would compare against a datasheet: how
    far from 90 degrees the two channels are, and how much louder Q is
    than I.
    """

    phase_deg: float
    gain_db: float

    @property
    def summary(self) -> str:
        return f"{self.phase_deg:+.2f} deg, {self.gain_db:+.2f} dB"


class IqBalance:
    """Estimate and undo quadrature gain and phase error.

    The estimator is the classic blind one and needs no test signal. If I and
    Q were a true quadrature pair they would be uncorrelated and equal in
    power, so any correlation between them is phase error and any difference
    in power is gain error:

        alpha = <I.Q> / <I.I>        Q is leaning on I by this much
        Q' = Q - alpha.I             remove the lean
        g  = sqrt(<I.I> / <Q'.Q'>)   then match the levels

    Both estimates are averaged across blocks with a slow leak, because a
    single block of a strong asymmetric signal is not evidence about the
    receiver. Convergence takes a second or so and then stays put.
    """

    def __init__(self, sample_rate: float, tau_s: float = 0.5) -> None:
        self.sample_rate = float(sample_rate)
        self.tau_s = float(tau_s)
        self._alpha = 0.0
        self._gain = 1.0
        self._seeded = False

    @property
    def estimate(self) -> ImbalanceEstimate:
        """Undo the two corrections to recover the error that produced them.

        `alpha` on its own is not the phase error: it is the lean of Q on I,
        which the channel's own gain error scales. Multiplying it by the gain
        correction cancels that scaling exactly and leaves the tangent of the
        angle, which is why this is an `arctan` of a product rather than the
        `arcsin` it looks like it should be.
        """
        phase = float(np.arctan(self._alpha * self._gain))
        gain = 1.0 / max(self._gain * np.cos(phase), 1e-9)
        return ImbalanceEstimate(
            phase_deg=float(np.degrees(phase)),
            gain_db=float(20.0 * np.log10(max(gain, 1e-9))),
        )

    def reset(self) -> None:
        self._alpha = 0.0
        self._gain = 1.0
        self._seeded = False

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if iq.size < 2:
            return iq
        i = iq.real.astype(np.float64)
        q = iq.imag.astype(np.float64)

        power_i = float(np.mean(i * i))
        if power_i <= 1e-20:
            return iq
        alpha = float(np.mean(i * q)) / power_i
        deskewed = q - alpha * i
        power_q = float(np.mean(deskewed * deskewed))
        gain = float(np.sqrt(power_i / power_q)) if power_q > 1e-20 else 1.0
        # A wildly wrong estimate means the block held something pathological
        # rather than that the hardware changed; clamp rather than chase it.
        alpha = float(np.clip(alpha, -0.5, 0.5))
        gain = float(np.clip(gain, 0.5, 2.0))

        if not self._seeded:
            self._alpha, self._gain = alpha, gain
            self._seeded = True
        else:
            span = iq.size / self.sample_rate
            leak = 1.0 - float(np.exp(-span / max(1e-6, self.tau_s)))
            self._alpha += leak * (alpha - self._alpha)
            self._gain += leak * (gain - self._gain)

        corrected = (q - self._alpha * i) * self._gain
        return (i + 1j * corrected).astype(np.complex64)


class FrequencyShifter:
    """Mix a stream by a fixed offset, keeping the phase continuous.

    Continuity across blocks is the whole of it: restarting the oscillator's
    phase at every block would put a step in the middle of the signal at the
    block rate, which is a buzz at 37 Hz rather than the silence it should be.
    """

    def __init__(self, sample_rate: float, offset_hz: float = 0.0) -> None:
        self.sample_rate = float(sample_rate)
        self.offset_hz = float(offset_hz)
        self._phase = 0.0

    def reset(self) -> None:
        self._phase = 0.0

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if self.offset_hz == 0.0 or iq.size == 0:
            return iq
        step = 2.0 * np.pi * self.offset_hz / self.sample_rate
        angle = self._phase + step * np.arange(iq.size, dtype=np.float64)
        # Kept inside one turn so the accumulated phase cannot lose precision
        # over a long listening session.
        self._phase = float((self._phase + step * iq.size) % (2.0 * np.pi))
        return (iq * np.exp(1j * angle)).astype(np.complex64)


__all__ = [
    "DcRemover",
    "FrequencyShifter",
    "ImbalanceEstimate",
    "IqBalance",
    "swap_iq",
]
