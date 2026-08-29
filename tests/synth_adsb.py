"""Synthetic Mode S bursts, for testing the ADS-B receiver without a sky.

The generator is deliberately independent of the decoder: it builds frames
from field definitions and appends a real CRC-24, so a test that decodes one
is checking the decoder's arithmetic rather than agreeing with itself about a
constant. The only thing shared is the polynomial, which is the standard's.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RATE = 2_400_000
# A Mode S pulse is 0.5 us long and the standard allows 50-100 ns of rise
# time. Modelling it matters: an ideal square edge lands on a sample or it
# does not, which makes a decoder look phase-sensitive when it is not.
RISE_US = 0.06

_POLY = 0xFFF409

_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


def parity(payload: bytes) -> bytes:
    """The three CRC bytes that make `payload` a valid frame.

    The register form below behaves as though 24 zeros already followed the
    message, so the remainder it leaves *is* the parity. Appending the zeros
    by hand would append them twice and produce a frame that fails its own
    checkword - which is exactly the shape of bug a generator sharing code
    with its decoder would hide.
    """
    rem = 0
    for byte in payload:
        rem ^= byte << 16
        for _ in range(8):
            rem = ((rem << 1) ^ _POLY) if rem & 0x800000 else (rem << 1)
            rem &= 0xFFFFFF
    return rem.to_bytes(3, "big")


def frame(payload: bytes) -> bytes:
    return payload + parity(payload)


def squitter(icao: int, me: bytes, capability: int = 5) -> bytes:
    """A DF17 extended squitter carrying a seven-byte ME field."""
    if len(me) != 7:
        raise ValueError("ME field is seven bytes")
    head = bytes([(17 << 3) | (capability & 0x07)]) + icao.to_bytes(3, "big")
    return frame(head + me)


def identity_me(callsign: str, type_code: int = 4, category: int = 0) -> bytes:
    """ME field for an aircraft identification message."""
    text = callsign.ljust(8)[:8]
    bits = 0
    for char in text:
        bits = (bits << 6) | _CHARSET.index(char)
    return bytes([(type_code << 3) | (category & 0x07)]) + bits.to_bytes(6, "big")


def altitude_code(feet: int) -> int:
    """The 12-bit field, Q bit set, for an altitude in 25-foot steps."""
    n = (int(feet) + 1000) // 25
    return ((n & 0x7F0) << 1) | 0x10 | (n & 0x0F)


def position_me(
    type_code: int, altitude: int, odd: bool, lat_cpr: int, lon_cpr: int
) -> bytes:
    """ME field for an airborne position message."""
    ac = altitude_code(altitude)
    me = bytearray(7)
    me[0] = (type_code << 3) | 0  # surveillance status and single-antenna flag
    me[1] = (ac >> 4) & 0xFF
    me[2] = ((ac & 0x0F) << 4) | (0x04 if odd else 0x00) | ((lat_cpr >> 15) & 0x03)
    me[3] = (lat_cpr >> 7) & 0xFF
    me[4] = ((lat_cpr & 0x7F) << 1) | ((lon_cpr >> 16) & 0x01)
    me[5] = (lon_cpr >> 8) & 0xFF
    me[6] = lon_cpr & 0xFF
    return bytes(me)


def velocity_me(
    east_kt: int, north_kt: int, vertical_fpm: int = 0, subtype: int = 1
) -> bytes:
    """ME field for a ground-referenced velocity message."""
    me = bytearray(7)
    me[0] = (19 << 3) | (subtype & 0x07)
    east = abs(int(east_kt)) + 1
    north = abs(int(north_kt)) + 1
    me[1] = (0x04 if east_kt < 0 else 0x00) | ((east >> 8) & 0x03)
    me[2] = east & 0xFF
    me[3] = (0x80 if north_kt < 0 else 0x00) | ((north >> 3) & 0x7F)
    me[4] = (north & 0x07) << 5
    if vertical_fpm:
        code = abs(int(vertical_fpm)) // 64 + 1
        me[4] |= (0x08 if vertical_fpm < 0 else 0x00) | ((code >> 6) & 0x07)
        me[5] = (code & 0x3F) << 2
    return bytes(me)


def cpr_encode(lat: float, lon: float, odd: bool) -> tuple[int, int]:
    """Position to a CPR pair, the transmitter's half of the algorithm."""
    from bettersdr.decode.adsb import _nl

    i = 1 if odd else 0
    dlat = 360.0 / (60.0 - i)
    yz = int(np.floor(131072.0 * ((lat % dlat) / dlat) + 0.5)) % 131072
    rlat = dlat * (yz / 131072.0 + np.floor(lat / dlat))
    ni = max(_nl(rlat) - i, 1)
    dlon = 360.0 / ni
    xz = int(np.floor(131072.0 * ((lon % dlon) / dlon) + 0.5)) % 131072
    return yz, xz


def _envelope(edges: list[tuple[float, float]], micros: np.ndarray) -> np.ndarray:
    """Trapezoidal pulses, evaluated at the given times.

    Each pulse only touches its own few samples, so it is written into a slice
    rather than over the whole block - otherwise a one-second capture costs a
    hundred thousand full-length array passes.
    """
    out = np.zeros(micros.size, dtype=np.float64)
    if micros.size < 2:
        return out
    per_us = 1.0 / (micros[1] - micros[0])
    for start, stop in edges:
        lo = max(0, int((start - RISE_US) * per_us) - 1)
        hi = min(micros.size, int((stop + RISE_US) * per_us) + 2)
        if hi <= lo:
            continue
        window = micros[lo:hi]
        rise = np.clip((window - start) / RISE_US, 0.0, 1.0)
        fall = np.clip((stop - window) / RISE_US, 0.0, 1.0)
        np.maximum(out[lo:hi], np.minimum(rise, fall), out=out[lo:hi])
    return out


def burst(
    frames: list[bytes],
    rate: float = DEFAULT_RATE,
    amplitude: float = 0.5,
    gap_us: float = 200.0,
    start_us: float = 30.0,
    offset_hz: float = 0.0,
    noise_rms: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """One IQ block containing the given frames, laid out end to end."""
    edges: list[tuple[float, float]] = []
    cursor = float(start_us)
    for data in frames:
        for pulse in (0.0, 1.0, 3.5, 4.5):
            edges.append((cursor + pulse, cursor + pulse + 0.5))
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        for index, bit in enumerate(bits):
            half = cursor + 8.0 + index + (0.0 if bit else 0.5)
            edges.append((half, half + 0.5))
        cursor += 8.0 + len(bits) + gap_us

    total = int((cursor + start_us) * rate / 1e6)
    micros = np.arange(total, dtype=np.float64) * 1e6 / rate
    signal = amplitude * _envelope(edges, micros)
    if offset_hz:
        signal = signal * np.exp(2j * np.pi * offset_hz * micros * 1e-6)
    block = np.asarray(signal, dtype=np.complex128)
    if noise_rms:
        rng = np.random.default_rng(seed)
        scale = noise_rms / np.sqrt(2.0)
        block = block + rng.normal(0, scale, total) + 1j * rng.normal(0, scale, total)
    return block.astype(np.complex64)


__all__ = [
    "altitude_code",
    "burst",
    "cpr_encode",
    "frame",
    "identity_me",
    "parity",
    "position_me",
    "squitter",
    "velocity_me",
]
