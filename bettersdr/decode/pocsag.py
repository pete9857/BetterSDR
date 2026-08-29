"""POCSAG: the text messages pagers still carry, read off a two-way FM channel.

This is the last of the Phase 4 decoders and the one that best makes the app's
argument. RDS tells you what a station calls itself; ADS-B tells you where an
aeroplane is. POCSAG puts somebody's actual words on the screen - a hospital
calling a doctor, a plant calling an engineer - out of a band a beginner would
otherwise have heard as two seconds of buzzing and moved past.

The chain, from the top:

    FM discriminator -> DC-tracked slicer -> bits at 512/1200/2400 bps
    -> 32-bit codewords -> BCH(31,21) -> address and message codewords
    -> capcode, numeric digits or 7-bit ASCII

Four decisions are worth knowing about before reading the code.

**The bit clock is recovered the way RDS recovers its symbol clock, because
the same thing goes wrong otherwise.** There is no whole number of samples
per bit at any rate the radio produces - 512 bps at a 96 kHz IF is 187.5 - so
the bit instants are a floating-point ramp read out of a running sum with
`np.interp`, exactly as in `decode/rds.py` and `decode/adsb.py`. The running
sum across one bit period *is* the matched filter for the rectangular
waveform, which is why no filter precedes it.

**All three baud rates run at once, rather than one being detected first.**
They cost a vectorised pass each, a transmitter can change rate between
transmissions, and the alternative - deciding from a preamble - throws away
every message whose preamble was missed. Which rate is actually producing
codewords is reported rather than assumed.

**Frame sync is stateless.** A batch is 544 bits and every one of them opens
with the same 32-bit sync codeword, so rather than hold a lock the receiver
looks for that codeword everywhere and decodes the 544 bits behind each hit.
Allowing two bit errors in the match, a false hit turns up once in eight
million bit positions - an hour of a 2400 bps channel - and the sixteen
codewords behind it would still have to pass their own checkwords to reach
the screen. A lock would buy nothing and would need a recovery path of its
own.

**One bit of error correction, not two.** BCH(31,21) has a minimum distance
of six and can correct two errors, which is what most pager decoders do. It
also means three errors can land within two of the *wrong* codeword, and a
mis-corrected address codeword is somebody's message shown against somebody
else's pager number. Correcting one error needs five to go wrong the same
way, so that is where the line is drawn - the same reasoning as RDS refusing
to repair a block at all, at the one setting where refusing entirely would
cost most of the traffic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# The three rates in the standard. 512 is the original, 1200 carries most of
# what is still on the air, 2400 the busiest commercial systems.
BAUD_RATES = (512, 1200, 2400)

# The frame synchronisation codeword that opens every batch, and the idle
# codeword that fills the frames nobody is using. Both are fixed constants.
SYNC_WORD = 0x7CD215D8
IDLE_WORD = 0x7A89C197

CODEWORD_BITS = 32
FRAMES_PER_BATCH = 8
CODEWORDS_PER_BATCH = 2 * FRAMES_PER_BATCH
BATCH_BITS = CODEWORD_BITS * (CODEWORDS_PER_BATCH + 1)

# g(x) = x^10 + x^9 + x^8 + x^6 + x^5 + x^3 + 1.
_BCH_POLY = 0x769
_BCH_N = 31
_BCH_K = 21

# How far from the sync codeword a 32-bit window may sit and still be taken
# for it. Two errors in thirty-two is 529 of the 2^32 possible patterns.
_SYNC_TOLERANCE = 2

# The IF has to carry the deviation - +/-4.5 kHz nominal - and the modulation
# on top of it. Below this the channel filter has already taken the signal
# apart and there is nothing left to slice.
MIN_IF_RATE_HZ = 20_000.0

# Positions across one bit the timing estimator tries. Sixteen puts the worst
# residual error at a thirty-second of a bit, well below what the slicer can
# notice.
_TIMING_CANDIDATES = 16
# How much of the measured error is acted on each block, and how long the
# per-position energies are remembered for. A DSP block holds only a handful
# of bits at 512 bps, so one block's estimate on its own is too noisy to
# steer by.
_TIMING_GAIN = 0.35
_ENERGY_DECAY = 0.6

# Time constant of the DC tracker. The discriminator sits off zero by however
# far the tuning is off, and slicing about the wrong level turns every bit
# into the same bit.
_DC_SECONDS = 0.05

# How many pages are kept. Enough to scroll back through a quiet afternoon,
# few enough that the state stays cheap to copy across threads.
MAX_PAGES = 200

# Numeric messages carry four bits per character and the bits arrive in the
# opposite order to the one they are read in, so this table is indexed by the
# nibble as transmitted and the reversal is already baked into it. 10 is
# unassigned, 11 marks an urgent call, and the last three are the brackets
# and the hyphen a telephone number needs.
_NUMERIC_DIGITS = "0123456789*U -)("


def _reverse4(value: int) -> int:
    return int(f"{value:04b}"[::-1], 2)


_NUMERIC_TABLE = "".join(_NUMERIC_DIGITS[_reverse4(n)] for n in range(16))


def syndrome(word: int) -> int:
    """The BCH remainder of a 32-bit codeword. Zero means no detected error."""
    shreg = (word >> 1) & 0x7FFFFFFF
    mask = 1 << (_BCH_N - 1)
    coeff = _BCH_POLY << (_BCH_K - 1)
    for _ in range(_BCH_K):
        if shreg & mask:
            shreg ^= coeff
        mask >>= 1
        coeff >>= 1
    return shreg


def _parity(word: int) -> int:
    """1 when the codeword carries an odd number of ones, which is a fault."""
    return bin(word).count("1") & 1


# Which single bit each syndrome accuses. Bit 0 is the parity bit, which the
# polynomial never sees - an error there shows up as odd parity with a clean
# checkword, and is the one case handled separately below.
_SINGLE_BIT = {syndrome(1 << bit): bit for bit in range(1, CODEWORD_BITS)}


def check(word: int) -> tuple[int, int] | None:
    """Validate a codeword, correcting at most one bit.

    Returns the corrected word and how many bits it changed, or None when the
    codeword cannot be trusted. See the module docstring for why the limit is
    one bit rather than the two the code could manage.
    """
    odd = _parity(word)
    remainder = syndrome(word)
    if remainder == 0:
        return (word, 0) if not odd else (word ^ 1, 1)
    if odd:
        # A single error anywhere in the polynomial's reach leaves the parity
        # odd as well. Two errors leave it even, and are refused rather than
        # guessed at.
        bit = _SINGLE_BIT.get(remainder)
        if bit is not None:
            return word ^ (1 << bit), 1
    return None


@dataclass(frozen=True)
class Page:
    """One message, as it should be read out to somebody.

    `capcode` is the pager's own number - the thing a switchboard dials.
    `text` is the best reading of the message; `numeric` and `alpha` are the
    two ways the same bits can be read, both kept so a doubtful one can be
    shown either way rather than guessed at silently.
    """

    capcode: int
    function: int
    kind: str
    text: str
    numeric: str = ""
    alpha: str = ""
    baud: int = 0
    received: float = field(default_factory=time.time)
    errors: int = 0

    @property
    def label(self) -> str:
        """What to show when the message carries no text at all."""
        return "Beep - no message" if self.kind == "tone" else self.text


@dataclass(frozen=True)
class PocsagState:
    """Everything heard so far. Immutable, so it can cross threads."""

    pages: tuple[Page, ...] = ()
    baud: int | None = None
    codewords_ok: int = 0
    codewords_bad: int = 0
    batches: int = 0

    @property
    def quality(self) -> float:
        """Share of codewords that arrived intact, 0 to 1."""
        total = self.codewords_ok + self.codewords_bad
        return self.codewords_ok / total if total else 0.0


class _BitTrack:
    """One baud rate's clock: discriminator samples in, bits out.

    A bit's value is the mean of the signal across it, taken as the difference
    of a running sum at two fractional positions. That integral is the matched
    filter for a rectangular symbol, so nothing filters ahead of it - and
    because both ends are fractional, a bit period 187.5 samples long is no
    harder than one exactly 40 samples long.
    """

    def __init__(self, sample_rate: float, baud: int) -> None:
        self.baud = baud
        self.samples_per_bit = sample_rate / baud
        step = self.samples_per_bit / _TIMING_CANDIDATES
        # Offsets either side of where the clock currently believes a bit
        # starts, so a converged loop sits on the middle one and stops moving.
        self._offsets = (np.arange(_TIMING_CANDIDATES) - _TIMING_CANDIDATES // 2) * step
        self._centre = _TIMING_CANDIDATES // 2
        self.reset()

    def reset(self) -> None:
        self.position = float(self.samples_per_bit)
        self._energy = np.zeros(_TIMING_CANDIDATES)

    @property
    def lookbehind(self) -> float:
        """Samples that have to stay in the buffer for the next call."""
        return self.samples_per_bit

    def feed(self, total: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Read whole bits out of a running sum, and re-steer the clock.

        `total[i]` is the sum of every sample before i, so the integral up to
        a continuous instant t is read at t + 0.5: a sample stands for the
        interval around its own instant, not the one before it. Half a sample
        is a fifth of a bit at 2400 bps through a 96 kHz IF, so the half is
        not a rounding detail.
        """
        samples = total.size - 1
        half = 0.5 * self.samples_per_bit
        room = samples - 0.5 - half - self.position
        count = int(room // self.samples_per_bit) if room > 0 else 0
        if count <= 0:
            return np.zeros(0, dtype=np.uint8)

        edges = (
            self.position
            + self._offsets[:, None]
            + self.samples_per_bit * np.arange(count + 1)[None, :]
        )
        run = np.interp(edges.ravel() + 0.5, grid, total).reshape(edges.shape)
        values = np.diff(run, axis=1) / self.samples_per_bit

        # The right instant is the one where each bit is read whole. A clock
        # half a bit out averages every transition away, so the mean size of
        # the readings peaks exactly where the clock belongs - and the reading
        # is taken over several blocks because one block holds seven bits at
        # 512 bps, which is not enough to trust on its own.
        energy = np.abs(values).mean(axis=1)
        self._energy = self._energy * _ENERGY_DECAY + energy * (1.0 - _ENERGY_DECAY)
        error = float(self._offsets[int(np.argmax(self._energy))])

        self.position += count * self.samples_per_bit + _TIMING_GAIN * error
        return (values[self._centre] > 0.0).astype(np.uint8)

    def consume(self, samples: int) -> None:
        self.position -= samples


class _Frames:
    """Bits in, whole batches of checked codewords out.

    The bit index each batch was found at travels with it, because whether a
    message continues into the next batch is a question of whether the two
    are adjacent - and 544 bits apart with nothing in between is the only
    thing adjacent can mean.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._bits = np.zeros(0, dtype=np.uint8)
        self._origin = 0
        self._next = 0
        self.codewords_ok = 0
        self.codewords_bad = 0
        self.batches = 0

    @property
    def received(self) -> int:
        """How many bits have arrived in total, examined or not."""
        return self._origin + self._bits.size

    def feed(self, bits: np.ndarray) -> list[tuple[int, list[int | None]]]:
        if bits.size:
            self._bits = np.concatenate((self._bits, bits))
        limit = self._bits.size - BATCH_BITS
        if limit < 0:
            return []

        window = self._bits[: limit + CODEWORD_BITS]
        words = np.zeros(limit + 1, dtype=np.uint32)
        for shift in range(CODEWORD_BITS):
            words = (words << np.uint32(1)) | window[shift : shift + limit + 1]
        upright = np.bitwise_count(words ^ np.uint32(SYNC_WORD))
        flipped = np.bitwise_count(words ^ np.uint32(SYNC_WORD ^ 0xFFFFFFFF))
        found = np.flatnonzero(np.minimum(upright, flipped) <= _SYNC_TOLERANCE)

        batches: list[tuple[int, list[int | None]]] = []
        for position in found.tolist():
            index = self._origin + position
            if index < self._next:
                continue
            self._next = index + BATCH_BITS
            inverted = bool(upright[position] > _SYNC_TOLERANCE)
            batches.append((index, self._decode(position, inverted)))

        # Everything up to `limit` has now been looked at; anything past it
        # still needs samples that have not arrived, so it stays.
        self._bits = self._bits[limit + 1 :]
        self._origin += limit + 1
        return batches

    def _decode(self, position: int, inverted: bool) -> list[int | None]:
        """The sixteen codewords behind one sync word, each checked or None."""
        start = position + CODEWORD_BITS
        raw = self._bits[start : start + CODEWORDS_PER_BATCH * CODEWORD_BITS]
        if inverted:
            # Which way round the deviation runs depends on the transmitter
            # and on which side of the tuner the channel landed, and the sync
            # codeword is what says which it was.
            raw = raw ^ np.uint8(1)
        packed = raw.reshape(CODEWORDS_PER_BATCH, CODEWORD_BITS)
        weights = (1 << np.arange(CODEWORD_BITS - 1, -1, -1)).astype(np.uint32)
        self.batches += 1
        out: list[int | None] = []
        for word in (packed * weights).sum(axis=1, dtype=np.uint32).tolist():
            checked = check(int(word))
            if checked is None:
                self.codewords_bad += 1
                out.append(None)
            else:
                self.codewords_ok += 1
                out.append(checked[0])
        return out


def _numeric(words: list[int]) -> str:
    digits = []
    for word in words:
        data = (word >> 11) & 0xFFFFF
        for shift in range(16, -1, -4):
            digits.append(_NUMERIC_TABLE[(data >> shift) & 0xF])
    return "".join(digits).rstrip()


def _alphanumeric(words: list[int]) -> str:
    """Seven-bit ASCII, each character sent least significant bit first."""
    acc = 0
    held = 0
    out: list[str] = []
    for word in words:
        data = (word >> 11) & 0xFFFFF
        for shift in range(19, -1, -1):
            acc |= ((data >> shift) & 1) << held
            held += 1
            if held == 7:
                out.append(chr(acc))
                acc = 0
                held = 0
    text = "".join(out)
    # Pagers pad the last codeword out with nulls and mark the end of the
    # message with an ETX or an EOT. None of that belongs on screen, and nor
    # does a control character that arrived because a bit was wrong.
    text = text.split("\x03")[0].split("\x04")[0]
    return "".join(c if c.isprintable() else " " for c in text).rstrip()


def _printable_share(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isprintable()) / len(text)


class _Message:
    """One page being assembled, across as many batches as it takes."""

    def __init__(self, capcode: int, function: int, baud: int) -> None:
        self.capcode = capcode
        self.function = function
        self.baud = baud
        self.words: list[int] = []
        self.errors = 0
        self.started = time.time()

    def finish(self) -> Page:
        if not self.words:
            return Page(
                capcode=self.capcode,
                function=self.function,
                kind="tone",
                text="",
                baud=self.baud,
                received=self.started,
                errors=self.errors,
            )
        numeric = _numeric(self.words)
        alpha = _alphanumeric(self.words)
        # Function 3 means alphanumeric on every network yet seen and 0 means
        # numeric, but those two bits are a hint rather than a promise. A
        # numeric pager reading out a wall of control characters is a worse
        # answer than a digit string of the wrong shape, so the hint chooses
        # and the reading still has to survive being looked at.
        readable = bool(alpha) and _printable_share(alpha) > 0.8
        kind = "alphanumeric" if (self.function == 3 and readable) else "numeric"
        if kind == "numeric" and not numeric.strip() and readable:
            kind = "alphanumeric"
        return Page(
            capcode=self.capcode,
            function=self.function,
            kind=kind,
            text=alpha if kind == "alphanumeric" else numeric,
            numeric=numeric,
            alpha=alpha,
            baud=self.baud,
            received=self.started,
            errors=self.errors,
        )


class _Assembler:
    """Codewords in, finished pages out."""

    def __init__(self, baud: int) -> None:
        self.baud = baud
        self.reset()

    def reset(self) -> None:
        self._current: _Message | None = None
        self._end = -1

    def _flush(self, out: list[Page]) -> None:
        if self._current is not None:
            out.append(self._current.finish())
            self._current = None

    def expire(self, received: int) -> list[Page]:
        """Finish a message that the bit stream has simply run away from.

        Nearly every message ends with an idle codeword or with the address
        of the next one. The exception is the last message of a transmission
        that filled its final batch exactly, which has room for neither - and
        without this it would sit half-finished for as long as the app stayed
        tuned there.

        The wait is a whole batch of bits and cannot be shorter, because that
        is how long it takes to know that a batch which *would* have been
        adjacent is not there. On air that is half a second at 1200 bps,
        which is about as fast as a message could honestly be declared over.
        """
        out: list[Page] = []
        if self._current is not None and received >= self._end + BATCH_BITS:
            self._flush(out)
        return out

    def batch(self, index: int, words: list[int | None]) -> list[Page]:
        out: list[Page] = []
        if index != self._end:
            # A gap between batches is a gap in the message: two adjacent ones
            # are 544 bits apart with nothing at all in between, so anything
            # else means the transmitter stopped or we lost the thread.
            self._flush(out)
        self._end = index + BATCH_BITS

        for position, word in enumerate(words):
            if word is None:
                if self._current is not None:
                    self._current.errors += 1
                continue
            if word == IDLE_WORD:
                self._flush(out)
                continue
            if word & 0x80000000:
                if self._current is not None:
                    self._current.words.append(word)
                continue
            # An address codeword. Its lowest three bits are not transmitted -
            # they are which frame of the batch it arrived in, which is how a
            # pager can listen to one eighth of the traffic and sleep through
            # the rest.
            self._flush(out)
            address = (word >> 13) & 0x3FFFF
            function = (word >> 11) & 0x3
            self._current = _Message(
                (address << 3) | (position // 2), function, self.baud
            )
        return out


class PocsagReceiver:
    """The whole chain: discriminator output in, `PocsagState` out.

    Lives on the demodulator for the same reason the RDS receiver does: what
    it needs is the frequency deviation itself, before the audio filter has
    rounded the corners off the bits and before a squelch has muted them.
    """

    def __init__(self, if_rate: float) -> None:
        if if_rate < MIN_IF_RATE_HZ:
            raise ValueError(f"IF rate {if_rate:.0f} Hz is too narrow for POCSAG")
        self._if_rate = float(if_rate)
        self._tracks = tuple(_BitTrack(self._if_rate, baud) for baud in BAUD_RATES)
        self._frames = tuple(_Frames() for _ in BAUD_RATES)
        self._assemblers = tuple(_Assembler(baud) for baud in BAUD_RATES)
        self.reset()

    @property
    def if_rate(self) -> float:
        return self._if_rate

    def reset(self) -> None:
        """Start again - after a retune there is a different transmitter."""
        self._carry = np.zeros(0, dtype=np.float64)
        self._dc: float | None = None
        self._pages: list[Page] = []
        self._baud: int | None = None
        for track, frames, assembler in zip(
            self._tracks, self._frames, self._assemblers, strict=True
        ):
            track.reset()
            frames.reset()
            assembler.reset()

    def process(self, discriminator: np.ndarray) -> None:
        block = np.asarray(discriminator, dtype=np.float64).ravel()
        if block.size == 0:
            return
        buffer = np.concatenate((self._carry, block)) if self._carry.size else block

        # Slicing about the wrong level turns every bit into the same bit, and
        # where that level sits is however far the tuning is off. So it is
        # tracked rather than assumed to be zero - and tracked slowly, or a
        # run of like bits inside a message pulls the threshold onto itself.
        # The rate is expressed in seconds because a DSP block is 1310 samples
        # on one window and 32768 on another.
        alpha = min(1.0, block.size / max(1.0, _DC_SECONDS * self._if_rate))
        mean = float(buffer.mean())
        self._dc = mean if self._dc is None else self._dc + alpha * (mean - self._dc)
        centred = buffer - self._dc

        total = np.empty(centred.size + 1, dtype=np.float64)
        total[0] = 0.0
        np.cumsum(centred, out=total[1:])
        grid = np.arange(total.size, dtype=np.float64)

        keep = centred.size
        for track, frames, assembler in zip(
            self._tracks, self._frames, self._assemblers, strict=True
        ):
            for index, words in frames.feed(track.feed(total, grid)):
                self._collect(track.baud, assembler.batch(index, words))
            self._collect(track.baud, assembler.expire(frames.received))
            keep = min(keep, int(track.position - track.lookbehind))
        if len(self._pages) > MAX_PAGES:
            del self._pages[: len(self._pages) - MAX_PAGES]

        consumed = max(0, min(keep, centred.size))
        if consumed:
            for track in self._tracks:
                track.consume(consumed)
        self._carry = buffer[consumed:].copy()

    def _collect(self, baud: int, pages: list[Page]) -> None:
        if not pages:
            return
        self._baud = baud
        self._pages.extend(pages)

    def snapshot(self) -> PocsagState:
        return PocsagState(
            pages=tuple(self._pages),
            baud=self._baud,
            codewords_ok=sum(frames.codewords_ok for frames in self._frames),
            codewords_bad=sum(frames.codewords_bad for frames in self._frames),
            batches=sum(frames.batches for frames in self._frames),
        )


__all__ = [
    "BATCH_BITS",
    "BAUD_RATES",
    "IDLE_WORD",
    "MAX_PAGES",
    "MIN_IF_RATE_HZ",
    "SYNC_WORD",
    "Page",
    "PocsagReceiver",
    "PocsagState",
    "check",
    "syndrome",
]
