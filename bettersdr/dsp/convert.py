"""Conversion from the dongle's raw byte stream to complex baseband.

The RTL2832U delivers interleaved unsigned 8-bit I/Q. Every downstream stage
wants complex64, and this conversion runs on every sample the app ever sees,
so it is worth doing well: a 256-entry lookup table turns the scale-and-offset
into a single gather, and viewing the result as complex64 pairs the values up
for free rather than copying two slices.
"""

from __future__ import annotations

import numpy as np

# Unsigned 8-bit centred on 127.5 and scaled to [-1, 1]. Using 127.5 rather
# than 128 keeps the mapping symmetric, so a silent input sits at exactly zero
# instead of a half-LSB DC offset we would have to remove later.
_SCALE = 127.5
_LUT = ((np.arange(256, dtype=np.float32) - _SCALE) / _SCALE).astype(np.float32)


def to_complex(raw: np.ndarray) -> np.ndarray:
    """Interleaved uint8 I/Q to complex64.

    `raw` must have an even length: I, Q, I, Q, ...
    """
    if raw.dtype != np.uint8:
        raise TypeError(f"expected uint8 samples, got {raw.dtype}")
    if raw.size % 2:
        raise ValueError(f"expected an even number of bytes, got {raw.size}")
    return _LUT[raw].view(np.complex64)


def to_bytes(samples: np.ndarray) -> np.ndarray:
    """complex64 back to interleaved uint8, for synthesising test captures."""
    interleaved = samples.astype(np.complex64).view(np.float32)
    scaled = np.rint(interleaved * _SCALE + _SCALE)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square magnitude, the level meters' basic unit."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(samples) ** 2)))


def dbfs(value: float) -> float:
    """Linear amplitude to dB relative to full scale, floored for silence."""
    return 20.0 * np.log10(max(float(value), 1e-12))
