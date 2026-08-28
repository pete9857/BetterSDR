"""Tests for the raw-byte to complex-baseband conversion."""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp import convert

from . import synth


def test_extremes_map_to_full_scale():
    raw = np.array([0, 0, 255, 255], dtype=np.uint8)
    out = convert.to_complex(raw)
    assert out[0] == pytest.approx(-1 - 1j)
    assert out[1] == pytest.approx(1 + 1j)


def test_midpoint_is_symmetric_about_zero():
    """127 and 128 should straddle zero evenly, leaving no DC bias."""
    raw = np.array([127, 128], dtype=np.uint8)
    out = convert.to_complex(raw)
    assert out[0].real == pytest.approx(-out[0].imag, abs=1e-9)


def test_silence_has_no_dc_offset():
    raw = np.full(4096, 127, dtype=np.uint8)
    quiet = convert.to_complex(raw)
    assert abs(np.mean(quiet)) < 0.01


def test_output_length_is_half_the_input():
    raw = np.zeros(16384, dtype=np.uint8)
    assert convert.to_complex(raw).size == 8192


def test_rejects_odd_length():
    with pytest.raises(ValueError):
        convert.to_complex(np.zeros(3, dtype=np.uint8))


def test_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        convert.to_complex(np.zeros(4, dtype=np.float32))


def test_roundtrip_survives_quantisation():
    """to_bytes then to_complex should return the original within 1 LSB."""
    original = synth.carrier(4096, offset_hz=50_000, amplitude=0.8)
    recovered = convert.to_complex(convert.to_bytes(original))
    assert np.max(np.abs(recovered - original)) < 2.0 / 255


def test_rms_and_dbfs_agree():
    signal = synth.carrier(4096, offset_hz=0, amplitude=0.5)
    assert convert.rms(signal) == pytest.approx(0.5, rel=1e-3)
    assert convert.dbfs(convert.rms(signal)) == pytest.approx(-6.02, abs=0.05)


def test_dbfs_floors_on_silence():
    assert convert.dbfs(0.0) < -200
