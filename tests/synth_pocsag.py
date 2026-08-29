"""Synthetic POCSAG transmissions, for testing the decoder without a pager.

The generator is deliberately independent of the decoder. It builds the
checkword by long division against the generator polynomial rather than by
calling the receiver's own routine, so a test that decodes one of these is
checking arithmetic rather than agreeing with itself about a constant. The
polynomial, the sync codeword and the character tables are the standard's;
everything else here is written out longhand.
"""

from __future__ import annotations

import numpy as np

SYNC_WORD = 0x7CD215D8
IDLE_WORD = 0x7A89C197
PREAMBLE_BITS = 576
CODEWORDS_PER_BATCH = 16

_POLY = 0x769
# Standard numeric characters, in the order their nibbles count up once the
# four bits have been put back into reading order.
NUMERIC_DIGITS = "0123456789*U -)("


def checkword(payload: int) -> int:
    """The ten BCH bits for a 21-bit payload, by long division."""
    remainder = (payload & 0x1FFFFF) << 10
    for bit in range(30, 9, -1):
        if remainder & (1 << bit):
            remainder ^= _POLY << (bit - 10)
    return remainder & 0x3FF


def codeword(payload: int) -> int:
    """A whole 32-bit codeword: payload, checkword, even parity."""
    word = ((payload & 0x1FFFFF) << 11) | (checkword(payload) << 1)
    return word | (bin(word).count("1") & 1)


def address_codeword(capcode: int, function: int) -> int:
    """The codeword that says whose pager this is.

    The bottom three bits of the capcode are not sent: they are which of the
    eight frames the codeword is placed in, which is how a pager can sleep
    through seven eighths of the traffic.
    """
    return codeword(((capcode >> 3) << 2) | (function & 3))


def _message_codewords(bits: list[int]) -> list[int]:
    words = []
    for start in range(0, len(bits), 20):
        chunk = bits[start : start + 20]
        chunk = chunk + [0] * (20 - len(chunk))
        data = 0
        for bit in chunk:
            data = (data << 1) | bit
        words.append(codeword((1 << 20) | data))
    return words


def alphanumeric(text: str) -> list[int]:
    """Message codewords carrying 7-bit ASCII, least significant bit first."""
    bits: list[int] = []
    for char in text:
        value = ord(char)
        bits += [(value >> index) & 1 for index in range(7)]
    return _message_codewords(bits)


def numeric(digits: str) -> list[int]:
    """Message codewords carrying four-bit digits, most significant bit last."""
    bits: list[int] = []
    for char in digits:
        value = NUMERIC_DIGITS.index(char)
        bits += [(value >> index) & 1 for index in range(4)]
    return _message_codewords(bits)


def transmission(pages: list[tuple[int, int, list[int]]]) -> list[int]:
    """Lay a list of (capcode, function, message codewords) out into batches.

    Every address codeword has to sit in the frame its capcode names, so the
    gaps between messages are filled with idle codewords until the next one
    comes round - which is the same thing a real transmitter does, and the
    reason a batch is nearly always mostly idle.
    """
    slots: list[int] = []
    for capcode, function, words in pages:
        frame = capcode & 7
        while len(slots) % CODEWORDS_PER_BATCH != 2 * frame:
            slots.append(IDLE_WORD)
        slots.append(address_codeword(capcode, function))
        slots.extend(words)
    while len(slots) % CODEWORDS_PER_BATCH:
        slots.append(IDLE_WORD)
    return slots


def bitstream(pages: list[tuple[int, int, list[int]]], preamble: int = PREAMBLE_BITS):
    """The whole transmission as bits: preamble, then batch after batch."""
    bits = [(index + 1) % 2 for index in range(preamble)]
    slots = transmission(pages)
    for start in range(0, len(slots), CODEWORDS_PER_BATCH):
        for word in (SYNC_WORD, *slots[start : start + CODEWORDS_PER_BATCH]):
            bits += [(word >> index) & 1 for index in range(31, -1, -1)]
    return np.array(bits, dtype=np.uint8)


def deviation(
    bits: np.ndarray,
    baud: int,
    sample_rate: float,
    *,
    offset: float = 0.0,
    invert: bool = False,
    ppm: float = 0.0,
    lead: float = 0.02,
    tail: float = 0.6,
) -> np.ndarray:
    """The transmitter's frequency deviation, as a discriminator would see it.

    `offset` shifts the bit grid by a fraction of a bit, which is the thing a
    receiver is not allowed to be sensitive to - a real transmission starts
    wherever it starts. `ppm` stretches the transmitter's clock away from
    ours, and `lead` and `tail` put silence either side so the decoder has to
    find the signal rather than being handed it. The tail is the longer of
    the two because a message that ends flush with its last batch is only
    known to be over once a batch's worth of bits has gone by without one.
    """
    rate = baud * (1.0 + ppm * 1e-6)
    quiet_lead = int(round(lead * sample_rate))
    quiet_tail = int(round(tail * sample_rate))
    span = int(np.floor(bits.size * sample_rate / rate))
    index = np.floor(np.arange(span) * rate / sample_rate + offset).astype(int)
    index = np.clip(index, 0, bits.size - 1)
    level = (2.0 * bits[index] - 1.0).astype(np.float64)
    if invert:
        level = -level
    return np.concatenate((np.zeros(quiet_lead), level, np.zeros(quiet_tail)))


def channel(
    signal: np.ndarray,
    *,
    noise: float = 0.0,
    dc: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """What the air and the tuning error do to it on the way in."""
    out = signal + dc
    if noise > 0.0:
        out = out + np.random.default_rng(seed).normal(0.0, noise, out.size)
    return out.astype(np.float64)
