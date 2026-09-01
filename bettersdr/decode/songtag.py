"""Reading a song out of RadioText, and knowing when it is not one.

`decode/rds.py` recovers the 64 characters a station is scrolling through its
RadioText field. What it cannot say is what those characters *mean*, because
the standard does not: RT is a free text field, and stations put the song in
it, their slogan in it, their phone number in it, and the name of whoever is
on air after the news in it, usually in rotation.

So this module answers two questions, and keeps them separate because they
fail differently:

* **Does this text name a song?** A pure function over one string. The
  message is cut on whatever separator the station used - *every* occurrence
  of it, which matters more than it sounds - the fields that are the station
  naming itself are set aside, and what has to be left is exactly two, with
  a short list of the shapes that are never a song: a web address, a phone
  number, a text-in short code.
* **Is this text the station rather than the programme?** That cannot be
  answered from one string, because "The Best Music Variety" and
  "Rush - Tom Sawyer" are equally well-formed. It is answered from *time*:
  the station's own messages come back every few minutes, and a song does
  not. `TextTracker` is that memory.

96.5 MHz here is the station that shaped most of this. It transmits
`96.5 Jack FM - The Real Slim Shady - Eminem`, and splitting on the first
separator alone gives an artist called `96.5 Jack FM` playing a song called
`The Real Slim Shady - Eminem` - a perfectly well-formed answer that is wrong
about every song the station plays, and one that no amount of confidence in
the audio would catch. It also shows why the *order* of the two survivors is
a real question rather than a convention: it writes the title first, and
nothing in the string says so.

The one thing this module cannot do from a single message is tell an artist
from a title. `Artist - Title` is the default because it is what almost every
station in the world writes; where a station wrote more fields than a song
has, `TextTracker` weighs the only evidence there is - an artist comes round
again with a different song, and a title does not.

The reason the second question needs care is the five-copies rule in
`audio/library.py`. A rule as simple as "text we have seen three times before
is the station talking" would work all afternoon and then quietly refuse to
record the third play of somebody's favourite song, which is precisely the
copy they were promised. So the test is not how often a string has appeared,
it is how *fast* it came back: a slogan returns within minutes, and a station
replaying a song within a quarter of an hour is doing something else wrong.

Neither question is trusted on its own. `audio/repro.py` also requires the
audio to have been music for most of the segment, and not to have been
somebody talking for much of it, before it saves anything - which is what
actually keeps advertisements out. Measured off air: an advertisement reads
`Accident? Boohoff Law. Better Off With Boohoff! - 96.5 Jack FM`, which is
refused here because only one field survives, and it does not sound like
music either. It has to pass both and it fails both.

No Qt, no numpy, no device: text in, a tag or nothing out.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

# Where a station splits the two halves, most common first. ` - ` with spaces
# rather than a bare hyphen, because hyphens are inside titles far more often
# than they are between them: "Jack-Ass", "Spider-Man", "Ob-La-Di".
SEPARATORS: tuple[str, ...] = (" - ", " – ", " — ", " -- ", " / ", " | ", " * ")
# `by` reads the other way round: title first, artist second.
REVERSED_SEPARATORS: tuple[str, ...] = (" by ",)

# Neither half of a song is this long. The longest real title in common
# rotation is under fifty characters; anything past this is a sentence, which
# means it is the station talking.
MAX_FIELD = 60
# ...nor this short. A one-character half is a separator that was not one.
MIN_FIELD = 2
# No RadioText message is longer than this - the field itself is 64
# characters - so anything past it is not a message we assembled correctly.
MAX_MESSAGE = 128

# How many different sets of companions a field has to have turned up with
# before it is taken to be the station rather than the programme. Two: the
# slogan on a station that prefixes it comes back with every song, and the
# title and artist beside it are different every time, so the second distinct
# song is the first moment there is any evidence at all. Below that there is
# none, and a guess here names somebody's file after a radio station.
STATION_SIBLINGS = 2
# ...and how large a share of the messages it has to have been in. This is the
# half that stops an artist being mistaken for the station, and without it the
# two rules are the same rule: after the second Eminem song, "Eminem" has two
# sets of companions exactly as the slogan does, and the field that is meant
# to become the artist gets thrown away as the station instead. What the
# slogan does and an artist does not is turn up in *nearly every* message.
STATION_SHARE = 0.75
# ...and how many before a field is taken to be the artist rather than the
# title. Same counter, different question: an artist comes round again with a
# different song and a title does not, which is the only evidence in the
# stream about which way round a station writes the two.
ARTIST_SIBLINGS = 2
# Which of the two survivors is the artist when the station wrote its own
# name as well. 1, meaning the artist is second and the title first, which is
# the opposite of the bare two-field convention - see `parse` for the
# measurements behind it.
STATION_FIRST_ARTIST = 1

# Shapes that are never a song, whichever half they land in. Deliberately
# short: every entry here is a thing that can also appear in a real title, so
# each one is a small bet, and the audio check behind this is what makes the
# bets affordable.
_PROMO = re.compile(
    r"(https?://|www\.|\.com\b|\.org\b|\.net\b|\.fm\b|@\w|\btext\s+\d|"
    r"\bcall\s+\d|\bstudio line\b|\badvert|\bsponsored\b|\bnow playing on\b)",
    re.IGNORECASE,
)
# A run of digits this long is a phone number or a short code, not a title.
_LONG_NUMBER = re.compile(r"\d{5,}")
# Something has to be pronounceable in each half.
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# How long a text has to keep coming back before it is the station rather
# than the programme. This is a *lifetime*, not a repeat count, and the
# difference is the whole rule: a great many stations alternate their slogan
# with the song title every few seconds, so both recur, and counting
# recurrences brands the song title just as fast as the slogan. What the
# slogan does and the title cannot is go on doing it after the song has
# ended. Fifteen minutes is longer than any song a broadcaster will play, so
# no title can reach it - and a station whose text really has sat still for a
# quarter of an hour is handled by `audio/repro.py`'s own length cap.
IDLE_LIFETIME_S = 900.0
# How long a text has to be off the air before that lifetime starts again.
# Ten minutes: longer than a song, so a slogan shown between two songs never
# resets, and shorter than any sensible rotation, so a track replayed later in
# the afternoon starts with a clean sheet and can still be recorded.
IDLE_FORGET_S = 600.0
# ...and the second half of the rule, which is what makes the first half safe.
# A text that has ever been left up for this long has been *announcing*
# something, which is not what a slogan flashed between titles does. Without
# it, a song played twice within `IDLE_FORGET_S` accumulates a lifetime like a
# slogan and is branded as one - and the third, fourth and fifth copies
# `audio/library.py` promises are silently never recorded.
SONG_STRETCH_S = 45.0


def _tidy(text: str) -> str:
    """One line, single spaces, no control characters.

    RDS pads its text field with spaces to the next segment boundary and
    terminates it with a carriage return, so almost every string arrives with
    something on the end that is not part of what the station wrote.
    """
    cleaned = "".join(
        " " if unicodedata.category(ch)[0] == "C" else ch for ch in str(text)
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _plausible_half(part: str) -> bool:
    return (
        MIN_FIELD <= len(part) <= MAX_FIELD
        and _HAS_LETTER.search(part) is not None
        and _LONG_NUMBER.search(part) is None
    )


def _fold(text: str) -> str:
    """One string for one field value, whatever case the playout used."""
    return _tidy(text).casefold()


def split_fields(text: str) -> tuple[str, ...] | None:
    """The message cut on whatever separator the station used, or None.

    Every occurrence of that separator splits, not just the first, and that
    is the difference between reading 96.5 MHz correctly and not reading it
    at all. It transmits `96.5 Jack FM - The Real Slim Shady - Eminem`, and
    splitting once gives an artist called `96.5 Jack FM` - a perfectly
    well-formed answer that is wrong about every song the station plays.

    The separator is chosen by priority rather than by position, so a title
    containing a slash is not cut on it while a ` - ` is sitting there.
    """
    line = _tidy(text)
    if not line or len(line) > MAX_MESSAGE or _PROMO.search(line):
        return None
    for separator in SEPARATORS:
        if separator in line:
            parts = tuple(part.strip() for part in line.split(separator))
            return parts if all(parts) else None
    for separator in REVERSED_SEPARATORS:
        if separator in line:
            left, _, right = line.partition(separator)
            # `by` reads the other way round: title first, artist second.
            return (right.strip(), left.strip())
    return None


# A dial position, which no song title carries and almost every station
# identification does.
_DIAL = re.compile(r"\b\d{2,4}[.,]\d\b")
# ...and the words that turn one into a station identification rather than a
# number that happens to be in a title.
_BAND_WORD = re.compile(r"\b(fm|am|mhz|khz|radio|hd\d?)\b", re.IGNORECASE)


def is_station_field(value: str, station: str = "", frequency_hz: float = 0.0) -> bool:
    """Whether this field is the station naming itself.

    Three rules, all of which answer immediately rather than needing a
    session's worth of evidence: the field carries the frequency the receiver
    is tuned to, it carries some other dial position alongside a band word,
    or it is the station's own name. `TextTracker` learns the rest - a slogan
    with no frequency in it can only be recognised by its coming back.

    The name has to be most of the field rather than merely somewhere in it.
    A station whose program service name is a word - JACK, MIX, KISS - would
    otherwise throw away `Jack Johnson` and `Kiss` as itself, which is the
    same class of mistake as reading a half-assembled message: confident,
    well-formed and wrong about one artist for ever.
    """
    folded = _fold(value)
    if not folded:
        return True
    if frequency_hz:
        dial = f"{frequency_hz / 1e6:.1f}"
        if dial in folded or dial.replace(".", ",") in folded:
            return True
    if _DIAL.search(folded) and (_BAND_WORD.search(folded) or _DIAL.fullmatch(folded)):
        return True
    name = _fold(station)
    if not name or len(name) < 0.5 * len(folded):
        return False
    return re.search(rf"\b{re.escape(name)}\b", folded) is not None


def key_for(artist: str, title: str) -> str:
    """What counts as the same song, for the five-copies rule.

    Case, punctuation and spacing all vary between plays of the same track -
    stations re-key their playout metadata, and one of them writes "Guns N'
    Roses" where another writes "Guns n Roses". Accents are folded for the
    same reason. What is deliberately *not* folded is anything in brackets:
    a remix and a radio edit are different recordings and a listener choosing
    between five copies would want to be able to tell.
    """
    joined = f"{artist} {title}"
    folded = unicodedata.normalize("NFKD", joined.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    # Two passes, and the second is not cosmetic: dropping an apostrophe
    # leaves a space where it was, so "Guns N' Roses" folds to two spaces
    # where "Guns n Roses" folds to one - and the five-copies rule would give
    # each spelling five copies of its own.
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", folded)).strip()


@dataclass(frozen=True)
class SongTag:
    """A song a station said it was playing, and why we believed it."""

    artist: str
    title: str
    reasons: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return key_for(self.artist, self.title)

    @property
    def display(self) -> str:
        return f"{self.artist} - {self.title}"

    @property
    def explanation(self) -> str:
        return f"{', '.join(self.reasons)} -> {self.display}"


def song_fields(
    text: str,
    station: str = "",
    frequency_hz: float = 0.0,
    known_station: Callable[[str], bool] | None = None,
) -> tuple[tuple[str, str], tuple[str, ...]] | None:
    """The two fields that could be a song, and the ones set aside.

    None where the message does not split, where nothing plausible is left,
    or where more than two fields survive. That last is a refusal rather than
    a guess: `A - B - C` is `Station - Title - Artist` on one station and
    `Artist - Title - Part 2` on another, and there is nothing in the string
    that says which.
    """
    fields = split_fields(text)
    if fields is None or len(fields) < 2:
        return None
    dropped = tuple(
        value
        for value in fields
        if is_station_field(value, station, frequency_hz)
        or (known_station is not None and known_station(value))
    )
    kept = [value for value in fields if value not in dropped]
    if len(kept) != 2:
        return None
    first, second = kept
    if not (_plausible_half(first) and _plausible_half(second)):
        return None
    return (first, second), dropped


def parse(
    text: str,
    station: str = "",
    frequency_hz: float = 0.0,
    *,
    known_station: Callable[[str], bool] | None = None,
    known_artist: Callable[[str], bool] | None = None,
    artist_index: int | None = None,
) -> SongTag | None:
    """`Fleetwood Mac - Dreams` into its two halves, or None.

    None is the common answer and a perfectly good one: most of what crosses
    RadioText is not a song, and saying so is cheaper than being confidently
    wrong about a file somebody will find in their music folder next week.

    Which of the two halves is the artist is the one question the string
    cannot answer. `Artist - Title` is the default because it is what almost
    every station in the world writes; `artist_index` and `known_artist` are
    what `TextTracker` has learnt about this particular station, and 96.5 MHz
    here writes `Slogan - Title - Artist`, so the default is not always right
    and there is no punctuation anywhere that says so.
    """
    found = song_fields(text, station, frequency_hz, known_station)
    if found is None:
        return None
    (first, second), dropped = found

    reasons = [f'RadioText reads "{_tidy(text)}"']
    if dropped:
        reasons.append(f'set aside "{dropped[0]}" as the station')
    # Two shapes, two conventions, and they are not the same one.
    #
    # `A - B` with nothing set aside is `Artist - Title`, which is what
    # almost every station in the world writes and what this module used to
    # assume everywhere. `Slogan - A - B` is a station announcing what is on
    # rather than labelling a file, and that reads the other way round -
    # "on 96.5 Jack FM: Seven Nation Army, by the White Stripes". Measured on
    # the one station available: every one of `The Real Slim Shady - Eminem`,
    # `Come Out And Play - Offspring`, `Edge Of Seventeen - Stevie Nicks`,
    # `Seven Nation Army - White Stripes` and `With Or Without You - U2` puts
    # the title first. Defaulting to `Artist - Title` there named every file
    # on the station backwards.
    #
    # It is a prior rather than a fact, which is why the learner below can
    # still overturn it: a station writing `Slogan - Artist - Title` is
    # corrected the first time an artist comes round with a second song.
    if not dropped:
        index = 0
    elif artist_index is not None:
        index = artist_index
    else:
        index = STATION_FIRST_ARTIST
    why = "a station announcing what is on names the song first" if index == 1 else None
    if dropped and known_artist is not None:
        if known_artist(second) and not known_artist(first):
            index, why = 1, f'"{second}" has been on with another song'
        elif known_artist(first) and not known_artist(second):
            index = 0
    artist, title = (second, first) if index == 1 else (first, second)
    reasons.append(why or "read as artist then title")
    return SongTag(artist=artist, title=title, reasons=tuple(reasons))


@dataclass
class _Seen:
    """The stretch of afternoon over which one string has kept turning up."""

    first_seen: float = 0.0
    last_seen: float = 0.0
    longest: float = 0.0

    @property
    def lifetime(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class TextTracker:
    """The station's RadioText over time: what changed, and what is a song.

    Fed the current text on every update - which is what the decoder produces,
    the same string over and over - and reports only the changes. The memory
    it keeps is what separates the station's own messages from the programme;
    see the module docstring for why that cannot be done one string at a time.
    """

    text: str = ""
    since: float = 0.0
    _seen: dict[str, _Seen] = field(default_factory=dict)
    # For each field value the station has sent, the different sets of
    # companions it turned up beside. One counter, two questions: a value with
    # several companions is the station naming itself, and a value with two is
    # an artist rather than a title. Both are recorded whether or not the
    # message parsed as a song, because the message that teaches us which
    # field is the slogan is usually one we refused.
    _siblings: dict[str, set[str]] = field(default_factory=dict)
    # The distinct messages those companions were counted over. Distinct, not
    # a running total: a station that alternates its slogan with the song
    # sends the same message back every few seconds, and counting each arrival
    # would make every real field look rare by comparison.
    _messages: set[str] = field(default_factory=set)
    # Which of the two surviving fields this station puts the artist in, once
    # something has shown us. A playout template does not change during an
    # afternoon, so one song by an artist we have heard before settles the
    # order for every song after it.
    _artist_at: int | None = None

    def update(self, text: str, now: float) -> str | None:
        """Feed the current RadioText. Returns the new text on a change.

        Called on every block rather than only when something moves, because
        the memory being kept is about *how long* a string has gone on
        appearing, and that cannot be measured from the changes alone.
        """
        line = _tidy(text)
        if not line:
            return None
        if self.text and line != self.text:
            leaving = self._seen.get(self.text)
            if leaving is not None:
                leaving.longest = max(leaving.longest, now - self.since)
        record = self._seen.get(line)
        if record is None:
            record = self._seen[line] = _Seen(first_seen=now, last_seen=now)
        elif now - record.last_seen > IDLE_FORGET_S:
            # Gone long enough to be a different occasion entirely.
            record.first_seen = now
        record.last_seen = now
        if line == self.text:
            return None
        self.text, self.since = line, now
        self._remember_fields(line)
        return line

    def _remember_fields(self, line: str) -> None:
        """Note which field values keep company with which others."""
        fields = split_fields(line)
        if fields is None or len(fields) < 2:
            return
        self._messages.add(_fold(line))
        for index, value in enumerate(fields):
            company = " | ".join(
                _fold(other) for position, other in enumerate(fields) if position != index
            )
            self._siblings.setdefault(_fold(value), set()).add(company)

    def _companions(self, value: str) -> int:
        return len(self._siblings.get(_fold(value), ()))

    def is_station_text(self, value: str) -> bool:
        """Whether this field has come back beside too many songs to be one.

        The learnt half of `is_station_field`, and the only thing that can
        recognise a slogan with no frequency in it. It costs two songs to
        find out, which is why the immediate rules exist beside it.
        """
        seen = self._companions(value)
        return (
            seen >= STATION_SIBLINGS
            and seen >= STATION_SHARE * len(self._messages)
        )

    def is_artist_text(self, value: str) -> bool:
        """Whether this field has been on the air beside a *different* song."""
        return self._companions(value) >= ARTIST_SIBLINGS

    def is_idle(self, text: str) -> bool:
        """Whether this string is the station rather than the programme.

        Both halves are needed. Going on appearing for a quarter of an hour is
        what a slogan does; never once being left up for as long as it takes
        to announce a song is the other thing it does, and a real title -
        however often it comes round - will have held the field for minutes at
        a time at least once.
        """
        record = self._seen.get(_tidy(text))
        return (
            record is not None
            and record.lifetime >= IDLE_LIFETIME_S
            and record.longest < SONG_STRETCH_S
        )

    def tag(
        self, text: str, station: str = "", frequency_hz: float = 0.0
    ) -> SongTag | None:
        """The song this text names, if it names one and is not the station."""
        if self.is_idle(text):
            return None
        self._learn_order(text, station, frequency_hz)
        return parse(
            text,
            station,
            frequency_hz,
            known_station=self.is_station_text,
            known_artist=self.is_artist_text,
            artist_index=self._artist_at,
        )

    def _learn_order(self, text: str, station: str, frequency_hz: float) -> None:
        """Note which side this station writes the artist on, once shown."""
        if self._artist_at is not None:
            return
        found = song_fields(text, station, frequency_hz, self.is_station_text)
        if found is None or not found[1]:
            return
        first, second = found[0]
        if self.is_artist_text(second) and not self.is_artist_text(first):
            self._artist_at = 1
        elif self.is_artist_text(first) and not self.is_artist_text(second):
            self._artist_at = 0

    def reset(self) -> None:
        """Forget everything. A different station is a different vocabulary."""
        self.text = ""
        self.since = 0.0
        self._seen.clear()
        self._siblings.clear()
        self._messages.clear()
        self._artist_at = None


__all__ = [
    "ARTIST_SIBLINGS",
    "IDLE_FORGET_S",
    "IDLE_LIFETIME_S",
    "SONG_STRETCH_S",
    "STATION_FIRST_ARTIST",
    "STATION_SHARE",
    "STATION_SIBLINGS",
    "MAX_FIELD",
    "MAX_MESSAGE",
    "SEPARATORS",
    "SongTag",
    "TextTracker",
    "is_station_field",
    "key_for",
    "parse",
    "song_fields",
    "split_fields",
]
