"""ADS-B: what aircraft overhead say about themselves.

Every airliner in the sky broadcasts its identity, position, altitude and
speed twice a second on 1090 MHz, in the clear, to nobody in particular. This
is the module that reads it, and it is the second thing in the app - after
RDS - that reports what a signal *says* rather than what it looks like.

The chain, from the top:

    1090 MHz magnitude -> preamble correlation -> pulse-position bit slicing
    -> CRC-24 -> Mode S frame -> ICAO address, callsign, altitude, position

Four decisions are worth knowing about before reading the code.

**The whole decoder works on magnitude, not on IQ.** Mode S is on-off keying:
a pulse is present or it is not, and the phase carries nothing. `np.abs` on
the complex block is therefore the entire "demodulation", and everything
after it is timing and arithmetic. There is no demodulator in `dsp/demod.py`
for this and there should not be - the skeleton there decimates to 48 kHz
audio, which is exactly what a 1 Mbit/s data burst must not go through.

**Every read is the energy in a half-microsecond window, not the value of one
sample, and that is a correctness decision rather than a refinement.** A Mode
S pulse is half a microsecond long and a burst begins wherever it begins -
nothing aligns it to the sample grid. At 2.4 MS/s a pulse spans only 1.2
samples, so a pulse can easily contain exactly one, and reading single
samples means reading a triangle that interpolation invented: at some arrival
phases the read for a bit's first half lands on the skirt of a pulse that
belongs to its second, and the bit comes out inverted. Integrating instead
makes the read the pulse's *energy*, which is where it is regardless of
phase. A running sum answers any interval for the price of two lookups, so
a whole message is sliced from one `np.interp` over a cumulative sum.

That still leaves *where* to put the windows. The preamble search runs on a
coarse grid and places a burst only to within a quarter of a microsecond,
which at 2.4 MS/s is not close enough to be sure which half of a bit a
pulse's single sample belongs to - the marginal arrival phases are a quarter
of all of them, not a rare corner. So the last of the alignment is settled by
slicing at a few sub-step offsets and letting the checkword say which was
right. A wrong offset passes at one in sixteen million. The running sum's
tail carries across blocks, so a message spanning a block boundary is decoded
rather than lost.

**Nothing is believed without its checkword.** The Mode S CRC-24 has enough
strength to correct a single bit error and most decoders use it. This one does
not, for the same reason `decode/rds.py` does not: a mis-corrected frame is a
plausible aircraft at a plausible altitude in the wrong place, which is worse
than no aircraft at all. A frame either passes clean or is counted as bad.

**A position needs two messages, and the pair can be a lie.** Positions are
sent in Compact Position Reporting, which halves the bits by making a frame
ambiguous on its own; an even and an odd frame together resolve it. The two
have to come from the same aircraft *within a few seconds*, because a fast
aircraft moves between them - hence the timestamps on `_Track`, and hence the
latitude-zone cross-check that throws the pair away when they disagree.

Time here is derived from the sample count rather than read off the clock.
The decoder is fed on the DSP thread in blocks whose duration is known
exactly, so counting samples gives a monotonic clock that is right even when
the thread is late, and makes every test deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The detection grid steps twice per microsecond; every window is half a
# microsecond wide, which is one pulse. The width is structural - it is what
# makes a read a pulse's energy rather than a sample of its shape. The step is
# only how finely the preamble is searched for; `_FINE_OFFSETS` does the rest.
STEPS_PER_US = 2
_PER_BIT = STEPS_PER_US
# 8 us of preamble, then 56 or 112 bits at 1 us each.
PREAMBLE_STEPS = 8 * STEPS_PER_US
SHORT_BITS = 56
LONG_BITS = 112
MESSAGE_US = 8 + LONG_BITS
_LONG_STEPS = PREAMBLE_STEPS + _PER_BIT * LONG_BITS

# Steps from the start of the preamble. The four pulses sit at 0, 1, 3.5 and
# 4.5 microseconds; the gaps chosen here are far enough from every pulse that
# no arrival phase the detector can produce lets one leak into them, which is
# what makes the ratio below a fair test rather than a lucky one.
_PULSE_OFFSETS = (0, 2, 7, 9)
_GAP_OFFSETS = (4, 5, 12, 14)
# How far a pulse must stand above the gaps beside it. Generous: a real
# message is 20 dB up, and the job here is only to reject noise that happened
# to cross the gate.
_PULSE_RATIO = 2.0
# How far past a gate crossing to look for the true start. The gate fires on
# the first window holding any part of the first pulse, which is a step early.
_ALIGN_STEPS = 3
# Sub-step offsets, in microseconds, tried when slicing a message. They cover
# the quarter-microsecond the detection grid leaves undecided, finely enough
# that what is left cannot move a pulse across a half-bit boundary. Ordered by
# likelihood, since the first one to pass its checkword ends the search.
_FINE_OFFSETS = (
    0.0,
    0.0625,
    -0.0625,
    0.125,
    -0.125,
    0.1875,
    -0.1875,
    0.25,
    -0.25,
)

# Below this the bit slicer is interpolating detail that is not in the data.
# 2.4 MS/s, the rate the app already runs at, is comfortably above it.
MIN_SAMPLE_RATE_HZ = 2_000_000.0
# Where the traffic is. Worldwide, and the one frequency in this app that is
# not a matter of taste.
FREQUENCY_HZ = 1_090_000_000

# How far above the block's own noise floor the preamble pulses must sit
# before a candidate is worth a CRC. Pure rejection of junk: a real message
# is 20 dB up, and this only keeps the candidate list short enough that the
# per-candidate work stays negligible.
_PULSE_MARGIN = 3.0
# Bound on the work one block can ask for, in case the front end is being
# driven into clipping and every sample looks like a pulse edge.
_MAX_CANDIDATES = 4_000

# An aircraft heard this recently is still on the screen. Two position
# messages more than this far apart are not a pair - a jet moves 4 km in
# ten seconds, which is further than CPR's ambiguity spacing.
CPR_MAX_AGE_S = 10.0
AIRCRAFT_TIMEOUT_S = 60.0
AIRCRAFT_FORGET_S = 300.0

# g(x) = x^24 + x^23 + x^22 + x^21 + x^20 + x^19 + x^18 + x^17 + x^16 + x^15
#      + x^14 + x^13 + x^12 + x^10 + x^3 + 1, low 24 bits.
_POLY = 0xFFF409

# Mode S's own six-bit alphabet. Index 32 is a space; the gaps are unassigned
# and a station transmitting one is saying something we cannot read, so they
# stay as '#' rather than being guessed at.
_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

# Compact Position Reporting divides the globe into this many latitude zones.
_NZ = 15
_CPR_SCALE = 131_072.0  # 2**17


def _crc_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.uint32)
    for i in range(256):
        rem = i << 16
        for _ in range(8):
            rem = ((rem << 1) ^ _POLY) if rem & 0x800000 else (rem << 1)
        table[i] = rem & 0xFFFFFF
    return table


_CRC = _crc_table()


def checksum(frame: bytes) -> int:
    """Mode S CRC-24 over a whole frame, parity bytes included.

    A frame whose parity is a plain checksum - the extended squitters this
    module reads - leaves zero behind. The interrogation formats overlay the
    aircraft address on the parity instead, so their remainder *is* the
    address; that is a different receiver and this one does not attempt it.
    """
    rem = 0
    for byte in frame:
        rem = ((rem << 8) & 0xFFFFFF) ^ int(_CRC[((rem >> 16) ^ byte) & 0xFF])
    return rem


def _nl(lat: float) -> int:
    """How many longitude zones exist at this latitude.

    The whole point of CPR: zones get wider towards the poles, so the same
    seventeen bits keep roughly constant resolution over the whole globe.
    """
    lat = abs(lat)
    if lat >= 87.0:
        return 1
    if lat == 0.0:
        return 59
    inner = 1.0 - (1.0 - math.cos(math.pi / (2.0 * _NZ))) / math.cos(
        math.radians(lat)
    ) ** 2
    return int(math.floor(2.0 * math.pi / math.acos(min(1.0, max(-1.0, inner)))))


def _cpr_global(
    even: tuple[int, int], odd: tuple[int, int], odd_is_newer: bool, surface: bool
) -> tuple[float, float] | None:
    """Resolve an even/odd pair of position frames into one fix.

    Returns None when the two frames disagree about which latitude band they
    are in, which is what a mismatched pair - two different aircraft, or one
    that has moved a zone between them - looks like from here.
    """
    span = 90.0 if surface else 360.0
    lat_even, lon_even = even[0] / _CPR_SCALE, even[1] / _CPR_SCALE
    lat_odd, lon_odd = odd[0] / _CPR_SCALE, odd[1] / _CPR_SCALE

    j = math.floor(59.0 * lat_even - 60.0 * lat_odd + 0.5)
    rlat_even = (span / 60.0) * ((j % 60) + lat_even)
    rlat_odd = (span / 59.0) * ((j % 59) + lat_odd)
    if not surface:
        # The encoding wraps at 270 rather than at 180, so anything above it
        # is a southern latitude.
        if rlat_even >= 270.0:
            rlat_even -= 360.0
        if rlat_odd >= 270.0:
            rlat_odd -= 360.0
    if _nl(rlat_even) != _nl(rlat_odd):
        return None
    # The zone arithmetic runs from -90 to 270, so a pair that does not belong
    # together can resolve to a latitude no aircraft can be at. It can also
    # resolve to a perfectly ordinary wrong one, which nothing here can catch -
    # the pairing window in `_position` is the real defence, and this only
    # stops the obviously impossible reaching the screen.
    if not -90.0 <= rlat_even <= 90.0 or not -90.0 <= rlat_odd <= 90.0:
        return None

    lat = rlat_odd if odd_is_newer else rlat_even
    nl = _nl(lat)
    ni = max(nl - (1 if odd_is_newer else 0), 1)
    m = math.floor(lon_even * (nl - 1) - lon_odd * nl + 0.5)
    lon = (span / ni) * ((m % ni) + (lon_odd if odd_is_newer else lon_even))
    if lon >= 180.0:
        lon -= 360.0
    return lat, lon


def _cpr_local(
    ref: tuple[float, float], cpr: tuple[int, int], odd: bool, surface: bool
) -> tuple[float, float] | None:
    """Resolve one position frame against a position already known.

    Once an aircraft has a fix, every later frame can be decoded on its own -
    the ambiguity is a few hundred kilometres and the aircraft has not moved
    that far. This is what keeps a track updating twice a second rather than
    waiting for the next even/odd pair.
    """
    span = 90.0 if surface else 360.0
    ref_lat, ref_lon = ref
    lat_cpr, lon_cpr = cpr[0] / _CPR_SCALE, cpr[1] / _CPR_SCALE

    dlat = span / (60.0 - (1.0 if odd else 0.0))
    j = math.floor(ref_lat / dlat) + math.floor(0.5 + (ref_lat % dlat) / dlat - lat_cpr)
    lat = dlat * (j + lat_cpr)
    if abs(lat - ref_lat) > (span / 4.0):
        return None

    ni = max(_nl(lat) - (1 if odd else 0), 1)
    dlon = span / ni
    m = math.floor(ref_lon / dlon) + math.floor(0.5 + (ref_lon % dlon) / dlon - lon_cpr)
    lon = dlon * (m + lon_cpr)
    if lon >= 180.0:
        lon -= 360.0
    return lat, lon


def altitude_ft(code: int) -> int | None:
    """The 12-bit altitude field from an airborne position message.

    The Q bit says whether the remaining eleven bits are a count of 25-foot
    steps or the Gillham code used above 50,000 feet. Only the first is
    decoded: Gillham is a reflected-binary encoding of 100-foot steps that no
    airliner in normal flight uses, and returning nothing is better than
    returning a number this module has no way to test against real air.
    """
    if code == 0:
        return None
    if code & 0x10:
        n = ((code & 0xFE0) >> 1) | (code & 0x0F)
        return n * 25 - 1000
    return None


@dataclass
class _Track:
    """Everything known about one aircraft, and when it was learned."""

    icao: int
    first_seen: float
    last_seen: float
    messages: int = 0
    callsign: str | None = None
    altitude_ft: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    ground_speed_kt: float | None = None
    track_deg: float | None = None
    vertical_rate_fpm: int | None = None
    on_ground: bool = False
    rssi_dbfs: float = -100.0
    # The two halves of a position, each with the time it arrived and
    # whether it was sent from the ground. A pair older than `CPR_MAX_AGE_S`
    # is not a pair, and neither is one whose halves disagree about that
    # flag - the two encodings divide the globe differently.
    cpr_even: tuple[int, int] | None = None
    cpr_even_at: float = -1e9
    cpr_even_surface: bool = False
    cpr_odd: tuple[int, int] | None = None
    cpr_odd_at: float = -1e9
    cpr_odd_surface: bool = False


@dataclass(frozen=True)
class Aircraft:
    """One aircraft, as the screen should show it."""

    icao: int
    messages: int
    age_s: float
    callsign: str | None = None
    altitude_ft: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    ground_speed_kt: float | None = None
    track_deg: float | None = None
    vertical_rate_fpm: int | None = None
    on_ground: bool = False
    rssi_dbfs: float = -100.0

    @property
    def address(self) -> str:
        """The ICAO address as the six hex digits everyone quotes it in."""
        return f"{self.icao:06X}"

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def label(self) -> str:
        """What to put on a card. The callsign if it has said one."""
        return self.callsign or self.address


@dataclass(frozen=True)
class AdsbState:
    """A snapshot of the sky, safe to hand across threads."""

    aircraft: tuple[Aircraft, ...] = ()
    messages: int = 0
    bad: int = 0
    seconds: float = 0.0
    positions: int = 0

    @property
    def rate_per_minute(self) -> float:
        if self.seconds <= 0.0:
            return 0.0
        return self.messages * 60.0 / self.seconds


class AdsbDecoder:
    """Mode S frames in, aircraft out. No DSP, no timing, no radio."""

    def __init__(self) -> None:
        self.tracks: dict[int, _Track] = {}
        self.messages = 0
        self.positions = 0
        # Where the receiver is, when anybody has told us. Only used to
        # resolve a single position frame that has no partner; a fix built
        # from a pair needs no reference at all.
        self.reference: tuple[float, float] | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.messages = 0
        self.positions = 0

    def feed(self, frame: bytes, now: float, rssi_dbfs: float = -100.0) -> int | None:
        """Take one CRC-clean frame. Returns the ICAO address if it had one."""
        df = frame[0] >> 3
        if df in (17, 18):
            icao = int.from_bytes(frame[1:4], "big")
            track = self._track(icao, now)
            track.rssi_dbfs = rssi_dbfs
            self._extended_squitter(track, frame[4:11], now)
            self.messages += 1
            return icao
        if df == 11:
            # An all-call reply carries nothing but the address. Worth taking:
            # it keeps an aircraft on the screen between squitters, and it is
            # often the only thing heard at all from one at the edge of range.
            icao = int.from_bytes(frame[1:4], "big")
            track = self._track(icao, now)
            track.rssi_dbfs = rssi_dbfs
            self.messages += 1
            return icao
        return None

    def _track(self, icao: int, now: float) -> _Track:
        track = self.tracks.get(icao)
        if track is None:
            track = _Track(icao=icao, first_seen=now, last_seen=now)
            self.tracks[icao] = track
        track.last_seen = now
        track.messages += 1
        return track

    def _extended_squitter(self, track: _Track, me: bytes, now: float) -> None:
        type_code = me[0] >> 3
        if 1 <= type_code <= 4:
            track.callsign = _callsign(me)
        elif 5 <= type_code <= 8:
            track.on_ground = True
            track.altitude_ft = None
            self._surface(track, me, now)
        elif 9 <= type_code <= 18:
            track.on_ground = False
            track.altitude_ft = altitude_ft((me[1] << 4) | (me[2] >> 4))
            self._position(track, me, now, surface=False)
        elif type_code == 19:
            _velocity(track, me)
        elif 20 <= type_code <= 22:
            # Same position encoding, altitude from GNSS rather than the
            # barometer. Reported as an altitude either way; which instrument
            # produced it is not something a beginner needs on the screen.
            track.on_ground = False
            track.altitude_ft = altitude_ft((me[1] << 4) | (me[2] >> 4))
            self._position(track, me, now, surface=False)

    def _surface(self, track: _Track, me: bytes, now: float) -> None:
        movement = ((me[0] & 0x07) << 4) | (me[1] >> 4)
        track.ground_speed_kt = _surface_speed(movement)
        if me[1] & 0x08:
            track.track_deg = (((me[1] & 0x07) << 4) | (me[2] >> 4)) * 360.0 / 128.0
        self._position(track, me, now, surface=True)

    def _position(self, track: _Track, me: bytes, now: float, surface: bool) -> None:
        odd = bool(me[2] & 0x04)
        lat = ((me[2] & 0x03) << 15) | (me[3] << 7) | (me[4] >> 1)
        lon = ((me[4] & 0x01) << 16) | (me[5] << 8) | me[6]
        if odd:
            track.cpr_odd, track.cpr_odd_at = (lat, lon), now
            track.cpr_odd_surface = surface
        else:
            track.cpr_even, track.cpr_even_at = (lat, lon), now
            track.cpr_even_surface = surface

        fix: tuple[float, float] | None = None
        # Both halves must have been sent from the same place - both from the
        # air, or both from the ground. A surface frame divides the globe into
        # 90 degrees of latitude where an airborne one uses 360, so pairing
        # one of each applies the wrong span to half the arithmetic. It does
        # not produce an obviously broken answer: it produces a perfectly
        # ordinary-looking position somewhere else entirely, which is the one
        # failure this decoder has no way to notice afterwards. Seen off air
        # on 2026-08-28 - an aircraft on approach to Boeing Field appeared at
        # 57 degrees east, with its latitude still right.
        matched = (
            track.cpr_even_surface == surface and track.cpr_odd_surface == surface
        )
        if (
            matched
            and track.cpr_even is not None
            and track.cpr_odd is not None
            and abs(track.cpr_even_at - track.cpr_odd_at) <= CPR_MAX_AGE_S
        ):
            fix = _cpr_global(
                track.cpr_even,
                track.cpr_odd,
                odd_is_newer=track.cpr_odd_at >= track.cpr_even_at,
                surface=surface,
            )
        if fix is None:
            # No usable pair. If this aircraft - or the receiver - already has
            # a position, one frame is enough on its own.
            ref = (
                (track.latitude, track.longitude)
                if track.latitude is not None and track.longitude is not None
                else self.reference
            )
            if ref is not None:
                fix = _cpr_local(ref, (lat, lon), odd, surface)
        if fix is None:
            return
        track.latitude, track.longitude = fix
        self.positions += 1

    def snapshot(self, now: float, seconds: float, bad: int) -> AdsbState:
        """The current sky, most recently heard first, stale entries dropped."""
        for icao, track in list(self.tracks.items()):
            if now - track.last_seen > AIRCRAFT_FORGET_S:
                del self.tracks[icao]
        live = [
            t for t in self.tracks.values() if now - t.last_seen <= AIRCRAFT_TIMEOUT_S
        ]
        live.sort(key=lambda t: t.last_seen, reverse=True)
        return AdsbState(
            aircraft=tuple(
                Aircraft(
                    icao=t.icao,
                    messages=t.messages,
                    age_s=now - t.last_seen,
                    callsign=t.callsign,
                    altitude_ft=t.altitude_ft,
                    latitude=t.latitude,
                    longitude=t.longitude,
                    ground_speed_kt=t.ground_speed_kt,
                    track_deg=t.track_deg,
                    vertical_rate_fpm=t.vertical_rate_fpm,
                    on_ground=t.on_ground,
                    rssi_dbfs=t.rssi_dbfs,
                )
                for t in live
            ),
            messages=self.messages,
            bad=bad,
            seconds=seconds,
            positions=self.positions,
        )


def _callsign(me: bytes) -> str | None:
    """Eight six-bit characters packed into six bytes."""
    bits = int.from_bytes(me[1:7], "big")
    chars = [_CHARSET[(bits >> shift) & 0x3F] for shift in range(42, -1, -6)]
    text = "".join(chars).replace("#", "").strip()
    return text or None


def _surface_speed(movement: int) -> float | None:
    """The surface movement field, which is a piecewise-linear scale."""
    if movement in (0, 127) or movement > 127:
        return None
    if movement == 1:
        return 0.0
    if movement >= 124:
        return 175.0
    for lo, hi, base, step in (
        (2, 8, 0.125, 0.125),
        (9, 12, 1.0, 0.25),
        (13, 38, 2.0, 0.5),
        (39, 93, 15.0, 1.0),
        (94, 108, 70.0, 2.0),
        (109, 123, 100.0, 5.0),
    ):
        if lo <= movement <= hi:
            return base + (movement - lo) * step
    return None


def _velocity(track: _Track, me: bytes) -> None:
    subtype = me[0] & 0x07
    vertical = ((me[4] & 0x07) << 6) | (me[5] >> 2)
    if vertical:
        sign = -1 if me[4] & 0x08 else 1
        track.vertical_rate_fpm = sign * (vertical - 1) * 64

    if subtype not in (1, 2):
        # Subtypes 3 and 4 report airspeed and heading rather than a velocity
        # over the ground, which is a different quantity and only sent when
        # the aircraft has no position reference. Left undecoded rather than
        # written into a field labelled "ground speed".
        return
    east = ((me[1] & 0x03) << 8) | me[2]
    north = ((me[3] & 0x7F) << 3) | (me[4] >> 5)
    if east == 0 or north == 0:
        return
    scale = 4.0 if subtype == 2 else 1.0
    east_kt = (east - 1) * scale * (-1 if me[1] & 0x04 else 1)
    north_kt = (north - 1) * scale * (-1 if me[3] & 0x80 else 1)
    track.ground_speed_kt = math.hypot(east_kt, north_kt)
    track.track_deg = math.degrees(math.atan2(east_kt, north_kt)) % 360.0


def _align(window: np.ndarray, candidate: int) -> int:
    """Pick the step the preamble really starts on, near a gate crossing.

    The gate fires on the first window holding *any* of the first pulse, which
    is up to two steps before the pulse itself. A window's share of a pulse
    falls off with distance from the true start, so the weakest of the four
    pulse reads peaks there and nowhere else: taking its maximum over the next
    few steps recovers the alignment the gate cannot give.
    """
    best, best_score = candidate, -1.0
    for start in range(candidate, candidate + _ALIGN_STEPS):
        score = min(float(window[start + off]) for off in _PULSE_OFFSETS)
        if score > best_score:
            best, best_score = start, score
    return best


class AdsbReceiver:
    """Complex IQ at 1090 MHz in, a list of aircraft out."""

    def __init__(self, sample_rate: float) -> None:
        if sample_rate < MIN_SAMPLE_RATE_HZ:
            raise ValueError(
                f"ADS-B needs at least {MIN_SAMPLE_RATE_HZ / 1e6:g} MS/s, "
                f"got {sample_rate / 1e6:g}"
            )
        self._sample_rate = float(sample_rate)
        # Input samples per microsecond, which is the unit everything in Mode
        # S is naturally expressed in.
        self._per_us = self._sample_rate / 1e6
        self._step = self._per_us / STEPS_PER_US
        self.decoder = AdsbDecoder()
        self.bad = 0
        self._elapsed = 0.0
        self._reset_buffers()

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def _reset_buffers(self) -> None:
        self._raw = np.empty(0, dtype=np.float32)
        self._grid = np.empty(0, dtype=np.float64)
        self._samples_grid = np.empty(0, dtype=np.float64)
        # Absolute sample index of `_raw[0]`, and how far messages have been
        # taken. Both are absolute because the retained tail is scanned again
        # with the next block, and a burst in the overlap must be reported
        # once rather than twice.
        self._origin = 0.0
        self._taken = -1.0

    def reset(self) -> None:
        self._reset_buffers()
        self.decoder.reset()
        self.bad = 0
        self._elapsed = 0.0

    def process(self, iq: np.ndarray) -> None:
        # The clock is the sample count, not the wall: a block's duration is
        # known exactly, so ages and CPR pair windows stay right even when the
        # DSP thread is late, and every test is reproducible.
        self._elapsed += iq.size / self._sample_rate
        if iq.size == 0:
            return
        magnitude = np.abs(iq).astype(np.float32)
        buf = np.concatenate((self._raw, magnitude)) if self._raw.size else magnitude
        # `total[i]` is the sum of everything before sample i, so the energy
        # between two fractional positions is the difference of two readings
        # taken from it - which is what makes an arbitrary window cost two
        # lookups rather than a slice, and the whole message one `np.interp`.
        total = np.concatenate(([0.0], np.cumsum(buf, dtype=np.float64)))
        self._scan(buf, total)

        # Keep enough for a burst that started in this block and has not
        # finished arriving. Without it, a message spanning a block boundary
        # is lost - about one in every 250 at 2.4 MS/s and a 27 ms block, a
        # loss nothing else in the app would report.
        margin = int(MESSAGE_US * self._per_us) + 8
        keep = max(0, buf.size - margin)
        self._raw = buf[keep:]
        self._origin += keep

    # -- finding messages --------------------------------------------------

    def _energy(self, total: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Energy in the half-microsecond window starting at each position.

        Positions are in samples from the start of the buffer and may be
        fractional. The half-sample shift is not a nicety: a running sum puts
        all of a sample's weight after its own instant, while a sample stands
        for the interval around it. At 2.4 MS/s that bias is 0.21 us, which is
        most of a Mode S pulse, and it moves a pulse's whole energy into the
        wrong half of its bit.
        """
        grid = self._samples(total.size)
        begin = np.interp(positions + 0.5, grid, total)
        end = np.interp(positions + 0.5 * self._per_us + 0.5, grid, total)
        return end - begin

    def _samples(self, count: int) -> np.ndarray:
        """0, 1, 2, ... as `np.interp` wants it, cached.

        Rebuilding it each block costs as much as the interpolation that reads
        it. Indexing the running sum directly instead was tried and is three
        times slower than `np.interp`, which is worth writing down because the
        opposite is the obvious guess.
        """
        if self._samples_grid.size < count:
            self._samples_grid = np.arange(count, dtype=np.float64)
        return self._samples_grid[:count]

    def _steps(self, count: int) -> np.ndarray:
        """The detection grid, cached.

        Blocks are all one size after the first, and building the grid afresh
        each time costs as much as the interpolation that reads it.
        """
        if self._grid.size < count:
            self._grid = self._step * np.arange(count)
        return self._grid[:count]

    def _scan(self, buf: np.ndarray, total: np.ndarray) -> None:
        span = int(MESSAGE_US * self._per_us) + 2
        steps = int((buf.size - span) / self._step) if buf.size > span else 0
        if steps < _LONG_STEPS + _ALIGN_STEPS:
            return
        window = self._energy(total, self._steps(steps))

        for candidate in self._candidates(window):
            start = _align(window, int(candidate))
            position = self._origin + start * self._step
            if position <= self._taken:
                continue
            if self._decode(total, start * self._step):
                self._taken = position + MESSAGE_US * self._per_us

    def _candidates(self, window: np.ndarray) -> np.ndarray:
        """Steps where the 8 us preamble might start.

        Two stages, because the second is the expensive one. A cheap threshold
        against the block's own noise floor rules out all but a fraction of a
        percent of the steps in a quiet band; the pulse-and-gap pattern then
        runs on those few by fancy indexing rather than over the whole block.
        Doing it the other way round - a dozen full-length array passes - is
        the difference between a few percent of a core and a fifth of one.
        """
        limit = window.size - _LONG_STEPS - _ALIGN_STEPS
        if limit <= 0:
            return np.empty(0, dtype=np.intp)
        # Every sixteenth step is plenty for a noise floor and turns a sort of
        # the whole block into a sort of a sixteenth of it.
        gate = max(float(np.median(window[::16])) * _PULSE_MARGIN, 1e-9)
        found = np.flatnonzero(window[:limit] > gate)
        if found.size == 0:
            return found
        if found.size > _MAX_CANDIDATES:
            found = found[:_MAX_CANDIDATES]
        weakest = np.min([window[found + off] for off in _PULSE_OFFSETS], axis=0)
        loudest = np.max([window[found + off] for off in _GAP_OFFSETS], axis=0)
        return found[weakest > np.maximum(loudest * _PULSE_RATIO, gate)]

    # -- reading one message -----------------------------------------------

    def _decode(self, total: np.ndarray, position: float) -> int:
        """Slice, check and hand over one candidate. Returns the bits taken.

        The detection grid places a burst to within an eighth of a
        microsecond, and at 2.4 MS/s that is not close enough to be sure which
        half of a bit a pulse's single sample belongs to - the marginal cases
        are a quarter of all arrival phases, not a rare corner. So the last of
        the alignment is settled by trying a few sub-step offsets and letting
        the checkword say which was right. A wrong offset passes at one in
        sixteen million, so this costs nothing in confidence, and a candidate
        still counts as at most one bad frame however many offsets it took.
        """
        for offset in _FINE_OFFSETS:
            bits = self._bits(total, position + offset * self._per_us)
            if bits is None:
                break
            # The first five bits say which format this is, and formats 16 and
            # up are the long ones. Reading the length out of the message
            # rather than trying both keeps it to one CRC per offset.
            df = int(np.packbits(bits[:8])[0]) >> 3
            length = LONG_BITS if df >= 16 else SHORT_BITS
            frame = np.packbits(bits[:length]).tobytes()
            if checksum(frame) != 0:
                continue
            self.decoder.feed(frame, self._elapsed, self._rssi(total, position))
            return length
        self.bad += 1
        return 0

    def _bits(self, total: np.ndarray, position: float) -> np.ndarray | None:
        """The 112 data bits, read as adjacent half-microsecond energies."""
        body = position + 8.0 * self._per_us
        halves = body + np.arange(2 * LONG_BITS) * (0.5 * self._per_us)
        if halves[-1] + self._per_us >= total.size - 1:
            return None
        energy = self._energy(total, halves)
        return energy[0::2] > energy[1::2]

    def _rssi(self, total: np.ndarray, position: float) -> float:
        """How loud the four preamble pulses were, in dBFS.

        An energy over the window, so dividing by the window's width in
        samples turns it back into the amplitude every other level in the app
        is quoted as.
        """
        offsets = np.array(_PULSE_OFFSETS, dtype=np.float64) * self._step
        level = float(np.mean(self._energy(total, position + offsets)))
        return 20.0 * math.log10(max(level / (0.5 * self._per_us), 1e-9))

    def snapshot(self) -> AdsbState:
        return self.decoder.snapshot(self._elapsed, self._elapsed, self.bad)


__all__ = [
    "AdsbDecoder",
    "AdsbReceiver",
    "AdsbState",
    "Aircraft",
    "FREQUENCY_HZ",
    "MIN_SAMPLE_RATE_HZ",
    "altitude_ft",
    "checksum",
]
