"""RDS: the text a broadcast FM station sends alongside its audio.

This is the piece that turns `FM Radio - 88.5 MHz` into `KQED 88.5 - Forum`.
Everything else in the app describes a signal; this is the first thing that
reads what the signal *says*, which is why the plan calls it the biggest
single win in Phase 4.

The chain, from the top:

    57 kHz subcarrier -> BPSK symbols -> differential decode -> 26-bit blocks
    -> four-block groups -> program service name, radio text, PI code

Three decisions are worth knowing about before reading the code.

**The subcarrier oscillator free-runs rather than locking to the stereo
pilot.** The textbook receiver derives 57 kHz from the third harmonic of the
19 kHz pilot, because the standard guarantees the transmitter locked them
together. Doing that here would tie RDS to stereo and lose it on the mono
stations that carry it perfectly well, so instead the oscillator runs open and
a tracker built from the symbols themselves takes out whatever is left. That
tracker has to follow a *rate*, not just a phase: the multiplex arrives on the
dongle's sample clock rather than the transmitter's, and off a local station
the two measured 98 ppm apart - enough to sit the subcarrier 5.6 Hz off and
rotate the constellation 28 degrees across one DSP block. Assuming the offset
was negligible cost a fifth of the blocks, and made the decoder quietly worse
the larger the blocks got.

**Timing recovery interpolates rather than resampling.** 1187.5 baud does not
divide any sample rate the radio can produce, so there is no whole number of
samples per symbol anywhere in the chain. Rather than put a fractional
resampler in the path, the symbol instants are a floating-point ramp and
`np.interp` reads the matched filter at them; the same 98 ppm goes into the
ramp's slope, and a peak-seeking estimator steers its origin.

**Nothing is shown until it is known.** A block that fails its checkword is
reported as failed, never repaired - the standard allows correcting a burst of
up to five bits and most decoders do it, but a mis-corrected block puts
plausible wrong text on the screen, which is worse than no text for the same
reason the classifier says "Unknown signal". The identifier has to arrive
twice running before it is believed, and the eight-character name is only
published when all four of its pairs have arrived in order and unbroken.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from ..dsp.filters import FirDecimator

SUBCARRIER_HZ = 57_000.0
SYMBOL_RATE = 1187.5
# The subcarrier is +/-2.4 kHz wide. The stereo difference channel ends at
# 53 kHz, which lands 4 kHz below the subcarrier once it is mixed down, so
# this filter is what separates RDS from a signal that is often far stronger.
BANDWIDTH_HZ = 2_400.0
# Roughly ten samples per symbol: enough for the interpolating timing loop to
# have something to interpolate, few enough that the whole decoder costs a
# fraction of the demodulator feeding it.
TARGET_RATE = 11_875.0
# Below this the 57 kHz subcarrier is not inside the IF at all - it has been
# filtered off by the channel filter - so there is nothing to decode and
# attaching a receiver would only burn CPU. 130 kHz leaves headroom above the
# 114 kHz that Nyquist alone demands.
MIN_IF_RATE_HZ = 130_000.0

# Positions within a symbol the timing estimator tries, and how much of the
# answer it acts on each block.
_TIMING_CANDIDATES = 16
_TIMING_GAIN = 0.4
# How much of it goes into the symbol *rate* rather than the phase. The
# transmitter's clock and the dongle's are unrelated, and the difference
# measured 98 ppm off a local station - a whole symbol every eight seconds,
# which a phase-only loop can only chase, never catch.
_RATE_GAIN = 0.1
# Never believe more than this much rate error, whatever the noise says.
_RATE_LIMIT = 0.002
# The same clock ratio moves the 57 kHz subcarrier by 5.6 Hz, which is 28
# degrees of rotation across one DSP block, so the carrier tracker has to
# follow a rate as well as a phase. These are its two loop gains, and the
# limit is far more offset than any real station produces.
_CARRIER_GAIN = 0.3
_CARRIER_RATE_GAIN = 0.1
_CARRIER_RATE_LIMIT = 2.0 * np.pi * 60.0 / SYMBOL_RATE

# g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1, low ten bits only: the
# eleventh is implicit in the shift-register form below.
_POLY = 0x1B9

OFFSET_WORDS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}
_BY_WORD = {word: name for name, word in OFFSET_WORDS.items()}
# Which block may follow which. C' appears instead of C in "version B" groups,
# where the third block repeats the station identifier instead of carrying data.
_NEXT = {
    "A": ("B",),
    "B": ("C", "C'"),
    "C": ("D",),
    "C'": ("D",),
    "D": ("A",),
}
_POSITION = {"A": 0, "B": 1, "C": 2, "C'": 2, "D": 3}

# RBDS, the North American variant. The European table differs from number 5
# onwards, so this belongs with the band plan as regional data the day a
# second region arrives.
PTY_NAMES = (
    "",
    "News",
    "Information",
    "Sports",
    "Talk",
    "Rock",
    "Classic Rock",
    "Adult Hits",
    "Soft Rock",
    "Top 40",
    "Country",
    "Oldies",
    "Soft",
    "Nostalgia",
    "Jazz",
    "Classical",
    "Rhythm and Blues",
    "Soft R&B",
    "Foreign Language",
    "Religious Music",
    "Religious Talk",
    "Personality",
    "Public",
    "College",
    "Spanish Talk",
    "Spanish Music",
    "Hip Hop",
    "",
    "",
    "Weather",
    "Emergency Test",
    "ALERT!",
)


@cache
def checkword(info: int) -> int:
    """The ten-bit remainder of `info` * x^10 divided by the RDS generator."""
    register = 0
    for i in range(15, -1, -1):
        feedback = ((info >> i) & 1) ^ ((register >> 9) & 1)
        register = (register << 1) & 0x3FF
        if feedback:
            register ^= _POLY
    return register


def encode_block(info: int, offset: str) -> int:
    """The 26 bits a transmitter sends for one block. Used by the tests."""
    return ((info & 0xFFFF) << 10) | (checkword(info & 0xFFFF) ^ OFFSET_WORDS[offset])


def block_offset(block: int) -> str | None:
    """Which offset word a 26-bit block carries, or None if it is corrupt.

    The checkword is the block's own CRC exclusive-ored with a constant that
    says which of the four positions in a group this is. Recomputing the CRC
    and cancelling it leaves that constant behind - so one arithmetic step
    both validates the block and locates it.
    """
    seen = (block & 0x3FF) ^ checkword(block >> 10)
    return _BY_WORD.get(seen)


def callsign(pi: int) -> str | None:
    """The US callsign a PI code stands for, where it stands for one.

    RBDS gives most American stations a program identifier that is simply
    their callsign written in base 26, with one block of codes for the K
    prefix and one for W. Codes outside those two ranges belong to
    three-letter callsigns and network identifiers held in a lookup table this
    app does not carry, and the honest answer there is no answer rather than a
    plausible wrong one.
    """
    if 0x1000 <= pi <= 0x54A7:
        prefix, index = "K", pi - 0x1000
    elif 0x54A8 <= pi <= 0x994F:
        prefix, index = "W", pi - 0x54A8
    else:
        return None
    return prefix + "".join(
        chr(ord("A") + (index // place) % 26) for place in (676, 26, 1)
    )


def _character(byte: int) -> str:
    """One character of the RDS alphabet.

    Its printable range coincides with ASCII, and the accented upper half
    does not, so anything outside it becomes a space rather than mojibake.
    """
    return chr(byte) if 0x20 <= byte <= 0x7E else " "


class _Subcarrier:
    """The 57 kHz subcarrier, turned into differentially decoded bits."""

    def __init__(self, if_rate: float) -> None:
        self.if_rate = float(if_rate)
        self.factor = max(1, int(round(self.if_rate / TARGET_RATE)))
        self.rate = self.if_rate / self.factor
        self.nominal_symbol_samples = self.rate / SYMBOL_RATE
        self.samples_per_symbol = self.nominal_symbol_samples
        self.channel = FirDecimator.lowpass(
            self.factor,
            BANDWIDTH_HZ,
            self.if_rate,
            taps_per_phase=40,
            dtype=np.complex64,
        )
        half = max(1, int(round(self.nominal_symbol_samples / 2)))
        # The matched filter for a biphase symbol is that symbol reversed:
        # minus over the first half, plus over the second. Correlating against
        # it peaks once per symbol and rejects anything with no transition in
        # the middle, which is most of what leaks past the filter above.
        self._taps = np.concatenate(
            (-np.ones(half, dtype=np.float32), np.ones(half, dtype=np.float32))
        ) / float(half)
        self.level = 0.0
        self.reset()

    def reset(self) -> None:
        self.channel.reset()
        self.samples_per_symbol = self.nominal_symbol_samples
        self._nco = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(self._taps.size - 1, dtype=np.complex64)
        self._matched = np.zeros(0, dtype=np.complex64)
        # One whole symbol in, so the first retiming can move the cursor in
        # either direction without running off the front of the buffer.
        self._cursor = float(self.samples_per_symbol)
        self._phase = 0.0
        self._rotation = 0.0
        self._last_bit = 0

    def process(self, mpx: np.ndarray) -> np.ndarray:
        """Multiplex in, differentially decoded bits out."""
        samples = np.asarray(mpx, dtype=np.float32)
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))
        usable = samples.size - (samples.size % self.factor)
        self._pending = samples[usable:].copy()
        if usable == 0:
            return np.zeros(0, dtype=np.uint8)

        step = 2.0 * np.pi * SUBCARRIER_HZ / self.if_rate
        phase = self._nco + step * np.arange(usable, dtype=np.float64)
        self._nco = float((self._nco + step * usable) % (2.0 * np.pi))
        baseband = samples[:usable] * np.exp(-1j * phase)

        narrow = self.channel.process(baseband.astype(np.complex64))
        if narrow.size == 0:
            return np.zeros(0, dtype=np.uint8)
        buffered = np.concatenate((self._tail, narrow))
        if buffered.size < self._taps.size:
            self._tail = buffered
            return np.zeros(0, dtype=np.uint8)
        self._matched = np.concatenate(
            (self._matched, np.convolve(buffered, self._taps, mode="valid"))
        )
        self._tail = buffered[buffered.size - self._taps.size + 1 :].copy()

        symbols = self._symbols()
        if symbols.size == 0:
            return np.zeros(0, dtype=np.uint8)
        return self._bits(symbols)

    # -- symbol timing -----------------------------------------------------

    def _retime(self) -> None:
        """Steer the symbol cursor onto the peak of the matched filter.

        This measures the thing it wants to maximise rather than a proxy for
        it: for each of sixteen positions within the symbol, how large the
        matched filter is on average when read there. The right one stands out
        by a factor of four to seven even over the sixteen symbols a block
        holds, and a parabola through its two neighbours puts the answer well
        inside a tenth of a sample.

        An early-late detector was tried first and is what a textbook would
        reach for. It does not work here: a biphase symbol correlates almost
        as strongly against its own inverse half a symbol away, so the loop
        has two places it can sit and only one of them is right. Measuring the
        whole symbol at once has no such ambiguity - there is exactly one
        maximum.
        """
        span = self._matched.size
        count = int((span - 1) / self.samples_per_symbol) - 1
        if count < 3:
            return
        step = self.samples_per_symbol / _TIMING_CANDIDATES
        offsets = step * np.arange(_TIMING_CANDIDATES)
        instants = offsets[:, None] + self.samples_per_symbol * np.arange(count)
        grid = np.arange(span, dtype=np.float64)
        scores = np.interp(
            instants.ravel(), grid, np.abs(self._matched)
        ).reshape(instants.shape).mean(axis=1)

        peak = int(np.argmax(scores))
        left = scores[(peak - 1) % _TIMING_CANDIDATES]
        right = scores[(peak + 1) % _TIMING_CANDIDATES]
        curve = left - 2.0 * scores[peak] + right
        shift = 0.5 * (left - right) / curve if curve < 0 else 0.0
        target = (peak + float(np.clip(shift, -0.5, 0.5))) * step

        # Both are positions within one symbol, so the difference has to be
        # taken the short way round or a cursor sitting just past the end of a
        # symbol gets dragged backwards through the whole of it.
        half = self.samples_per_symbol / 2.0
        error = (target - self._cursor + half) % self.samples_per_symbol - half
        self._cursor += _TIMING_GAIN * error
        # Spread the same error over the symbols it accumulated across and it
        # becomes a rate correction. Without this the loop sits permanently
        # behind a clock that is 98 ppm out and reads every symbol slightly
        # off centre; with it the residual is the jitter alone.
        self.samples_per_symbol = float(
            np.clip(
                self.samples_per_symbol + _RATE_GAIN * error / count,
                self.nominal_symbol_samples * (1.0 - _RATE_LIMIT),
                self.nominal_symbol_samples * (1.0 + _RATE_LIMIT),
            )
        )

    def _symbols(self) -> np.ndarray:
        """Read the matched filter at the symbol instants.

        The instants are a floating-point ramp because no sample rate the
        radio produces holds a whole number of samples per symbol. Only the
        ramp's origin is steered; its slope is the sample clock's, and a TCXO
        moves a symbol boundary by a fiftieth of a symbol per second.
        """
        self._retime()
        last = self._matched.size - 1
        if self._cursor > last:
            return np.zeros(0, dtype=np.complex64)
        count = int((last - self._cursor) // self.samples_per_symbol) + 1
        instants = self._cursor + self.samples_per_symbol * np.arange(count)
        grid = np.arange(self._matched.size, dtype=np.float64)
        on = np.interp(instants, grid, self._matched)
        self.level = float(np.mean(np.abs(on)))

        self._cursor += count * self.samples_per_symbol
        # Leave a whole symbol of history in front of the cursor, so the next
        # block's retiming has somewhere to move it to in either direction.
        keep = int(self._cursor - self.samples_per_symbol)
        if keep > 0:
            self._matched = self._matched[keep:]
            self._cursor -= keep
        return on

    def _bits(self, symbols: np.ndarray) -> np.ndarray:
        """Turn symbols into bits, tracking the subcarrier's phase as it goes.

        Squaring a BPSK constellation collapses both symbols onto one point,
        whose angle is twice the carrier phase error. That is only defined
        modulo pi - which is exactly the ambiguity the differential decoding
        at the end removes, so every correction here is wrapped the same way
        and the tracker never chases it round in circles.

        A phase alone is not enough. The multiplex arrives on the dongle's
        sample clock rather than the transmitter's, and the two measured
        98 ppm apart: enough to put the subcarrier 5.6 Hz off and rotate the
        constellation 28 degrees across a single block. So the rate is
        tracked alongside the phase, and the correction is a ramp.
        """
        squared = symbols * symbols
        if squared.size > 1:
            advance = complex(np.sum(squared[1:] * np.conj(squared[:-1])))
            if advance != 0:
                step = 0.5 * float(np.angle(advance))
                self._rotation = float(
                    np.clip(
                        self._rotation + _CARRIER_RATE_GAIN * (step - self._rotation),
                        -_CARRIER_RATE_LIMIT,
                        _CARRIER_RATE_LIMIT,
                    )
                )
        ramp = self._phase + self._rotation * np.arange(symbols.size)
        residual = complex(np.sum(squared * np.exp(-2j * ramp)))
        if residual != 0:
            self._phase += _CARRIER_GAIN * 0.5 * float(np.angle(residual))
            ramp = self._phase + self._rotation * np.arange(symbols.size)
        data = (symbols * np.exp(-1j * ramp)).real
        self._phase = float((self._phase + self._rotation * symbols.size + np.pi)
                            % (2.0 * np.pi) - np.pi)

        bits = (data > 0).astype(np.uint8)
        previous = np.empty_like(bits)
        previous[0] = self._last_bit
        previous[1:] = bits[:-1]
        self._last_bit = int(bits[-1])
        return bits ^ previous


class _Sync:
    """Find the 26-bit block boundaries and hand back whole groups.

    Acquisition needs two agreeing observations, not one. A random 26-bit
    window carries a valid-looking offset word about once every two hundred
    tries, which at 1187 bits per second is several false locks a second; two
    of them landing 26 bits apart *and* in a legal order essentially never
    happens by chance.
    """

    LOSE_AFTER = 8

    def __init__(self) -> None:
        self.blocks_ok = 0
        self.blocks_bad = 0
        self.reset()

    def reset(self) -> None:
        self._register = 0
        self._filled = 0
        self._lose()

    def _lose(self) -> None:
        """Drop sync without touching the counters that report quality."""
        self.synced = False
        self._candidates: list[list] = []
        self._expect: tuple[str, ...] = ()
        self._since = 0
        self._bad = 0
        self._group: list[int | None] = [None] * 4
        self._valid = [False] * 4

    def feed(self, bits: np.ndarray) -> list[tuple[tuple[int, ...], tuple[bool, ...]]]:
        groups: list[tuple[tuple[int, ...], tuple[bool, ...]]] = []
        for bit in bits.tolist():
            self._register = ((self._register << 1) | int(bit)) & 0x3FFFFFF
            if self._filled < 26:
                self._filled += 1
                continue
            if self.synced:
                self._since += 1
                if self._since >= 26:
                    self._since = 0
                    self._advance(groups)
            else:
                self._search()
        return groups

    def _search(self) -> None:
        for candidate in self._candidates:
            candidate[1] += 1
        self._candidates = [c for c in self._candidates if c[1] <= 26]
        name = block_offset(self._register)
        if name is None:
            return
        for offset, age in self._candidates:
            if age == 26 and name in _NEXT[offset]:
                self._acquire(name)
                return
        self._candidates.append([name, 0])

    def _acquire(self, name: str) -> None:
        self._lose()
        self.synced = True
        self._store(name, True)

    def _advance(self, groups: list) -> None:
        name = block_offset(self._register)
        good = name is not None and name in self._expect
        if good:
            self.blocks_ok += 1
            self._bad = 0
        else:
            self.blocks_bad += 1
            self._bad += 1
            if self._bad >= self.LOSE_AFTER:
                self._lose()
                return
            # Ride out a burst of noise rather than dropping sync for it:
            # re-acquiring costs about a second, and the block boundary is
            # almost certainly still where it was.
            name = self._expect[0]
        self._store(name, good)
        if _POSITION[name] != 3:
            return
        if all(block is not None for block in self._group):
            groups.append((tuple(int(b) for b in self._group), tuple(self._valid)))
        self._group = [None] * 4
        self._valid = [False] * 4

    def _store(self, name: str, good: bool) -> None:
        position = _POSITION[name]
        self._group[position] = self._register >> 10
        self._valid[position] = good
        self._expect = _NEXT[name]


@dataclass(frozen=True)
class RdsState:
    """Everything decoded so far. Immutable, so it can cross threads."""

    pi: int | None = None
    callsign: str | None = None
    station: str = ""
    station_steady: bool = False
    text: str = ""
    # Whether the whole of that message has arrived, or it is still being
    # assembled. Anything reading the text as *data* - the song segmenter -
    # must wait for this; a display showing it fill in need not.
    text_steady: bool = False
    pty: int | None = None
    pty_name: str = ""
    traffic_program: bool = False
    traffic_announcement: bool = False
    stereo: bool | None = None
    groups: int = 0
    blocks_ok: int = 0
    blocks_bad: int = 0
    synced: bool = False

    @property
    def name(self) -> str:
        """The best stable name we have for the station, or an empty string.

        A station sending a fixed name has said what it wants to be called.
        One scrolling a song title through the same field has not, and its
        callsign - which the identifier gives us and which never changes - is
        the better answer there.
        """
        if self.station_steady and self.station.strip():
            return self.station.strip()
        return self.callsign or self.station.strip()

    @property
    def quality(self) -> float:
        """Share of blocks that arrived intact, 0 to 1."""
        total = self.blocks_ok + self.blocks_bad
        return self.blocks_ok / total if total else 0.0


class RdsDecoder:
    """Groups in, station information out."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pi: int | None = None
        # A corrupt block passes its checkword about once in a thousand, which
        # over a minute of a weak station is a near certainty. One of them was
        # enough to label 102.5 MHz with a Los Angeles callsign, so the
        # identifier has to arrive twice running before it is believed.
        self._pi_seen: int | None = None
        self.pty: int | None = None
        self.traffic_program = False
        self.traffic_announcement = False
        self.stereo: bool | None = None
        self.groups = 0
        # The name is eight characters sent two at a time, and it is only
        # published once all four pairs of one pass have arrived. Confirming
        # each character separately looks equivalent and is not: American
        # stations scroll song titles through this field, so characters from
        # neighbouring frames get mixed and the answer comes out as "ren KUow"
        # - measured off 94.9 MHz, where the station is KUOW.
        self._name = bytearray(b" " * 8)
        self._name_frame = bytearray(b" " * 8)
        self._name_have = 0
        self._name_next = 0
        self._name_steady = False
        self._text = bytearray(b" " * 64)
        self._text_flag: bool | None = None
        # Which segments of the message on the air have arrived, as a bit
        # per segment; how many characters each of them carries; and whether
        # one has come round a second time, which is how a station that sends
        # neither a carriage return nor all sixteen segments says it has
        # finished. The tally is cleared the moment a segment we have already
        # had turns up carrying something different, because that is a *new*
        # message arriving over the top of the old one. See `_settled`.
        self._text_have = 0
        self._text_span = 4
        self._text_wrapped = False

    def update(self, blocks: tuple[int, ...], valid: tuple[bool, ...]) -> None:
        a, b, c, d = blocks
        if valid[0]:
            if self._pi_seen == a:
                self.pi = a
            self._pi_seen = a
        if not valid[1]:
            return
        self.groups += 1
        group_type = (b >> 12) & 0xF
        version_b = bool((b >> 11) & 1)
        self.traffic_program = bool((b >> 10) & 1)
        self.pty = (b >> 5) & 0x1F
        if group_type == 0:
            self._program_service(b, d, valid[3])
        elif group_type == 2:
            self._radio_text(b, c, d, version_b, valid)

    def _program_service(self, b: int, d: int, have_d: bool) -> None:
        self.traffic_announcement = bool((b >> 4) & 1)
        segment = b & 0x3
        if segment == 0:
            # The decoder-identification bit is sent one bit per segment, and
            # segment zero's is the one that says stereo.
            self.stereo = bool((b >> 2) & 1)
        if segment != self._name_next or not have_d:
            # The four pairs have to arrive in order and unbroken. Accepting
            # them piecemeal is what put "NP & Now" on screen for KUOW: the
            # station scrolls a song title through this field, so a pair
            # missed to noise gets filled from the next frame along.
            self._name_have = 0
            self._name_next = 0 if segment == 0 and have_d else -1
            if self._name_next != 0:
                return
        self._name_frame[segment * 2] = d >> 8
        self._name_frame[segment * 2 + 1] = d & 0xFF
        self._name_have |= 1 << segment
        self._name_next = segment + 1
        if self._name_have == 0b1111:
            # American stations routinely scroll a song title through the name
            # field, so an eight-character frame is not necessarily a name.
            # Two identical frames running is what tells the two apart.
            self._name_steady = bytes(self._name_frame) == bytes(self._name)
            self._name[:] = self._name_frame
            self._name_have = 0
            self._name_next = -1

    def _radio_text(
        self, b: int, c: int, d: int, version_b: bool, valid: tuple[bool, ...]
    ) -> None:
        flag = bool((b >> 4) & 1)
        if flag != self._text_flag:
            # The A/B flag flips when the message changes. Without clearing,
            # the tail of a long song title stays stuck to the front of a
            # short one.
            self._text_flag = flag
            self._text[:] = b" " * 64
            self._text_have = 0
            self._text_wrapped = False
        segment = b & 0xF
        if version_b:
            # Version B has no room for block C's two characters - it repeats
            # the station identifier there instead - so the message is half
            # the length and arrives two characters at a time.
            self._text_span = 2
            if valid[3]:
                self._note(segment, self._text_chars(segment * 2, d))
            return
        self._text_span = 4
        changed = False
        if valid[2]:
            changed = self._text_chars(segment * 4, c)
        if valid[3]:
            changed = self._text_chars(segment * 4 + 2, d) or changed
        if valid[2] and valid[3]:
            self._note(segment, changed)

    def _text_chars(self, index: int, word: int) -> bool:
        """Write two characters. True where either was not already there."""
        changed = False
        for offset, byte in ((0, word >> 8), (1, word & 0xFF)):
            at = index + offset
            if at < len(self._text):
                changed = changed or self._text[at] != byte
                self._text[at] = byte
        return changed

    def _note(self, segment: int, changed: bool) -> None:
        """Record that a whole segment arrived, and whether it was news.

        A great many stations never toggle the A/B flag - 96.5 MHz here does
        not - so a shorter message arriving over a longer one leaves the tail
        of the old one behind and the two read as one string. Noticing that a
        segment carries something different is what stands in for the flag:
        everything gathered before it describes the message being replaced,
        so the tally starts again and the text is not trusted until a whole
        pass of the new one has arrived.

        Only a segment we have *already had* in this pass counts as news. On
        the first assembly of any message every segment carries something
        different from the spaces underneath it, so resetting on that would
        clear the tally sixteen times running and no message would ever be
        called whole.
        """
        bit = 1 << segment
        if self._text_have & bit:
            if changed:
                self._text_have = 0
                self._text_wrapped = False
            else:
                # Round again with nothing new in it: whatever the station is
                # sending, it has now sent all of it at least once.
                self._text_wrapped = True
        self._text_have |= bit

    def _settled(self) -> bool:
        """Whether the whole of the message on the air has arrived.

        The segments have to run unbroken from zero and reach the end of the
        message - the carriage return where there is one, and otherwise the
        last character anybody wrote. Anything short of that is a half-built
        string, and a half-built string parses: `96.5 Jack FM - The R` is a
        perfectly well-formed artist and title, and a song segmenter fed one
        of those per group starts a new song several times a second and
        finishes none of them.
        """
        run = 0
        while run < 16 and self._text_have & (1 << run):
            run += 1
        if run == 0:
            return False
        # The terminator has to be looked for in the bytes, not in the
        # decoded string: `_character` turns everything outside the printable
        # range into a space, carriage return included, so by the time there
        # is a string to search the end of the message is gone.
        end = self._text.find(0x0D)
        text = "".join(_character(byte) for byte in self._text)
        needed = end + 1 if end >= 0 else len(text.rstrip())
        if run * self._text_span < needed:
            return False
        # Reaching the last character written is not the same as reaching the
        # end of the message: the buffer is spaces underneath, so the first
        # four segments of `96.5 Jack FM - The Real Slim Shady - Eminem` cover
        # every character there is and read as the complete `96.5 Jack FM - T`.
        # Something has to say there is no more coming - the carriage return
        # that ends a message, a whole pass of sixteen segments, or a segment
        # arriving a second time with nothing new in it.
        return end >= 0 or run == 16 or self._text_wrapped

    def snapshot(self, synced: bool, ok: int, bad: int) -> RdsState:
        text = "".join(_character(byte) for byte in self._text)
        # A carriage return ends the message; everything after it is padding
        # the station never meant anybody to read. Found in the bytes for the
        # same reason as in `_settled`: the decoder has already turned it into
        # a space by the time the string exists.
        end = self._text.find(0x0D)
        return RdsState(
            pi=self.pi,
            callsign=None if self.pi is None else callsign(self.pi),
            station="".join(_character(byte) for byte in self._name),
            station_steady=self._name_steady,
            text=(text if end < 0 else text[:end]).rstrip(),
            text_steady=self._settled(),
            pty=self.pty,
            pty_name="" if self.pty is None else PTY_NAMES[self.pty],
            traffic_program=self.traffic_program,
            traffic_announcement=self.traffic_announcement,
            stereo=self.stereo,
            groups=self.groups,
            blocks_ok=ok,
            blocks_bad=bad,
            synced=synced,
        )


class RdsReceiver:
    """The whole chain: multiplex in, `RdsState` out.

    Lives on the demodulator rather than in the engine, because the multiplex
    only exists inside the FM path - between the discriminator and the
    de-emphasis, which would otherwise tilt the subcarrier into the noise.
    """

    def __init__(self, if_rate: float) -> None:
        if if_rate < MIN_IF_RATE_HZ:
            raise ValueError(
                f"IF rate {if_rate:.0f} Hz cannot carry the 57 kHz subcarrier"
            )
        self.subcarrier = _Subcarrier(if_rate)
        self.sync = _Sync()
        self.decoder = RdsDecoder()

    @property
    def if_rate(self) -> float:
        return self.subcarrier.if_rate

    def reset(self) -> None:
        """Start again from nothing - after a retune, there is a new station."""
        self.subcarrier.reset()
        self.sync.reset()
        self.sync.blocks_ok = 0
        self.sync.blocks_bad = 0
        self.decoder.reset()

    def process(self, mpx: np.ndarray) -> None:
        for blocks, valid in self.sync.feed(self.subcarrier.process(mpx)):
            self.decoder.update(blocks, valid)

    def snapshot(self) -> RdsState:
        return self.decoder.snapshot(
            self.sync.synced, self.sync.blocks_ok, self.sync.blocks_bad
        )


__all__ = [
    "MIN_IF_RATE_HZ",
    "OFFSET_WORDS",
    "PTY_NAMES",
    "RdsDecoder",
    "RdsReceiver",
    "RdsState",
    "block_offset",
    "callsign",
    "checkword",
    "encode_block",
]
