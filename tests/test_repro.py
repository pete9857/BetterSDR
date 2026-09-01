"""Repro-Radio: the gate, the caps, and songs cut out of a broadcast.

All of it runs against a clock the test moves by hand and synthetic audio from
`tests/synth_audio.py`, so a four-hour session cap and a three-minute song are
both a few milliseconds - the same bargain `tests/test_monitor.py` made, and
for the same reason: every rule here is a statement about elapsed time.

What none of this is, is a radio station. The boundaries between songs, the
lag in a station's RadioText and whether an advertisement break really does
sound like speech are all questions about real air, and are listed as such in
docs/PLAN.md.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from bettersdr.audio import encode
from bettersdr.audio import repro as rr
from bettersdr.audio.library import MAX_COPIES
from tests import synth_audio as sa

RATE = 48_000
# A block a second. The state machine does not care how big a block is, and
# eighty times fewer iterations makes the difference between a test file that
# runs in seconds and one nobody waits for.
BLOCK_S = 1.0

pytestmark = pytest.mark.skipif(
    not encode.available(), reason="the MP3 encoder is not installed"
)

MUSIC = sa.music(12.0, seed=1).astype(np.float32)
SPEECH = sa.speech(12.0, seed=2).astype(np.float32)
QUIET = np.zeros(int(12.0 * RATE), dtype=np.float32)


class Clock:
    """A clock that only moves when the test says so."""

    def __init__(self, now: float = 1_756_600_000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


class Station:
    """Somewhere to point the radio, and a cursor into its audio."""

    def __init__(self, radio: rr.ReproRadio, clock: Clock) -> None:
        self.radio = radio
        self.clock = clock
        self.at = 0

    def play(
        self,
        seconds: float,
        source: np.ndarray = MUSIC,
        *,
        text: str = "",
        squelch: bool | None = None,
        center_hz: float = 94.9e6,
        station: str = "KUOW",
        stereo: bool = False,
        block_s: float = BLOCK_S,
    ) -> None:
        frames = int(block_s * RATE)
        for _ in range(int(round(seconds / block_s))):
            index = np.arange(self.at, self.at + frames) % source.size
            block = source[index]
            self.at += frames
            if stereo:
                block = np.stack([block, block], axis=1)
            self.radio.feed(
                block,
                squelch_open=squelch,
                center_hz=center_hz,
                station=station,
                radio_text=text,
                genre="Rock",
            )
            self.clock.now += block_s


@pytest.fixture
def station(tmp_path):
    """A running Repro-Radio with no session cap, and a way to feed it."""

    def build(**settings):
        settings.setdefault("max_session_s", 0.0)
        clock = Clock()
        radio = rr.ReproRadio(
            tmp_path, rr.ReproSettings(**settings), rate=RATE, clock=clock
        )
        radio.start()
        return Station(radio, clock), radio, tmp_path

    return build


def clips(folder):
    return sorted(p.name for p in folder.glob("RR-*.mp3"))


def songs(folder):
    return sorted(p.name for p in (folder / "Songs").glob("*.mp3"))


def seconds_of(path):
    """How long an MP3 is, from its size. CBR is what makes this possible."""
    rate = encode.STEREO_BITRATE_KBPS if "Songs" in str(path) else None
    kbps = rate or encode.MONO_BITRATE_KBPS
    return path.stat().st_size / (kbps * 1000 / 8)


# -- naming ------------------------------------------------------------------


def test_the_name_says_where_and_when():
    """Month, day, hour, minute and second, two digits each, local time.

    Local rather than UTC, and stated as a wall-clock time here rather than
    converted, because the whole point of the short form is that somebody
    reads the time off it and matches it to what they remember hearing.
    """
    from datetime import datetime

    start = datetime(2026, 8, 31, 14, 30, 12).astimezone()
    end = datetime(2026, 8, 31, 14, 31, 45).astimezone()
    assert (
        rr.clip_name(98.5e6, start, end)
        == "RR-98.500MHz-0831143012-0831143145.mp3"
    )


def test_a_recording_in_progress_says_so_in_its_name():
    from datetime import datetime

    start = datetime(2026, 8, 31, 14, 30, 12).astimezone()
    assert rr.clip_name(162.55e6, start).endswith("-recording.mp3")


# -- the rolling buffer ------------------------------------------------------


def test_the_buffer_hands_back_the_stretch_asked_for():
    """The arithmetic the whole backwards step rests on."""
    buffer = rr._Preroll(seconds=10.0, rate=1000)
    for value in range(10):
        # A block is timestamped where it *ends*, so block 1 covers 1.0 to 2.0.
        buffer.push(np.full(1000, float(value), dtype=np.float32), now=value + 1.0)
    span = buffer.span(start=1.0, end=2.0)
    assert span.shape == (1000, 2)
    assert np.allclose(span, 1.0)
    assert np.allclose(buffer.tail(1.0), 9.0)


def test_the_buffer_answers_against_its_own_newest_sample():
    """A sweep stops the audio without stopping the clock. Asking for the
    last two seconds afterwards must give the last two seconds of audio, not
    a window shifted into a silence that was never recorded."""
    buffer = rr._Preroll(seconds=10.0, rate=1000)
    for value in range(5):
        buffer.push(np.full(1000, float(value), dtype=np.float32), now=value + 1.0)
    # The caller's clock has moved on four seconds; the buffer has not, and
    # the end it is asked for is past everything it holds.
    span = buffer.span(start=3.0, end=9.0)
    assert span.shape == (2000, 2)
    # The newest audio, not a window shifted into the silence.
    assert np.allclose(span[-1000:], 4.0)


def test_the_buffer_is_always_two_channels():
    buffer = rr._Preroll(seconds=2.0, rate=1000)
    buffer.push(np.ones(1000, dtype=np.float32))
    buffer.push(np.ones((1000, 2), dtype=np.float32))
    assert buffer.tail(2.0).shape == (2000, 2)


def test_asking_for_more_than_is_held_gives_what_there_is():
    buffer = rr._Preroll(seconds=2.0, rate=1000)
    buffer.push(np.ones(1000, dtype=np.float32))
    assert buffer.tail(30.0).shape == (1000, 2)


# -- clips -------------------------------------------------------------------


def test_a_recording_opens_when_the_squelch_does(station):
    play, radio, folder = station()
    play.play(3.0, SPEECH, squelch=False)
    assert not radio.status.recording
    play.play(3.0, SPEECH, squelch=True)
    assert radio.status.recording
    assert radio.status.clip_path.endswith("-recording.mp3")


def test_a_pause_for_breath_is_not_two_files(station):
    """The hang time. Without it the radio is left recording one half of an
    exchange per file, which is the same failure as the monitor's release."""
    play, radio, folder = station(hang_s=3.0)
    play.play(4.0, SPEECH, squelch=True)
    play.play(2.0, QUIET, squelch=False)
    assert radio.status.recording
    play.play(4.0, SPEECH, squelch=True)
    radio.stop()
    assert len(clips(folder)) == 1


def test_the_recording_closes_after_the_hang_and_is_given_its_end(station):
    play, radio, folder = station(hang_s=2.0)
    play.play(4.0, SPEECH, squelch=True)
    play.play(5.0, QUIET, squelch=False)
    assert not radio.status.recording
    names = clips(folder)
    assert len(names) == 1
    assert "-recording.mp3" not in names[0]
    # 94.900 MHz, and two full timestamps.
    assert names[0].startswith("RR-94.900MHz-")
    # ...and two full ten-digit stamps, neither of them the in-progress name.
    assert re.fullmatch(r"RR-94\.900MHz-\d{10}-\d{10}\.mp3", names[0])


def test_a_noise_burst_that_opened_the_squelch_is_thrown_away(station):
    """Measured on how long the gate was open, not on how long the file is:
    with a hang time the file is always at least the hang long, so a length
    test on the file could never discard anything."""
    play, radio, folder = station(hang_s=3.0)
    play.play(0.2, SPEECH, squelch=True, block_s=0.2)
    play.play(6.0, QUIET, squelch=False)
    assert not radio.status.recording
    assert clips(folder) == []


def test_a_short_transmission_is_still_a_transmission(station):
    """The other side of the same threshold. Somebody saying "roger" is a
    second of audio, and it is the whole reason a scanner was left running."""
    play, radio, folder = station(hang_s=3.0)
    play.play(1.0, SPEECH, squelch=True, block_s=0.5)
    play.play(6.0, QUIET, squelch=False)
    assert len(clips(folder)) == 1


def test_the_block_that_opened_the_recording_is_in_it(station):
    """It used to be dropped, which is a fraction of a second of somebody
    starting to speak - and a whole recording, when the file was closed
    before a second block ever arrived."""
    play, radio, folder = station(hang_s=0.0)
    play.play(1.0, SPEECH, squelch=True, block_s=1.0)
    play.play(1.0, QUIET, squelch=False)
    names = clips(folder)
    assert len(names) == 1
    assert (folder / names[0]).stat().st_size > 1_000


def test_a_recording_nothing_was_ever_written_to_closes_cleanly(station):
    """`lameenc.flush` raises on an encoder that was never handed a sample,
    and pressing the button and changing your mind is not a corner case."""
    play, radio, folder = station()
    play.play(1.0, SPEECH, squelch=True)
    radio._close_clip()
    # Open one and close it without a single block going through.
    radio._open_clip(94.9e6, np.zeros(0, dtype=np.float32))
    radio.stop()
    assert radio.status.message is None or "reached" not in radio.status.message


def test_one_long_recording_is_split_at_the_cap(station):
    play, radio, folder = station(max_clip_s=10.0)
    play.play(35.0, SPEECH, squelch=True)
    radio.stop()
    assert len(clips(folder)) == 4


def test_the_session_stops_itself_at_its_limit(station):
    play, radio, folder = station(max_session_s=20.0)
    play.play(40.0, SPEECH, squelch=True)
    assert not radio.status.enabled
    assert radio.finished
    assert "session limit" in radio.status.message
    # ...and what it had open was closed and named properly on the way out.
    assert len(clips(folder)) == 1
    assert "-recording.mp3" not in clips(folder)[0]


def test_nothing_is_recorded_once_it_has_stopped(station):
    play, radio, folder = station(max_session_s=20.0)
    play.play(40.0, SPEECH, squelch=True)
    before = clips(folder)
    play.play(30.0, SPEECH, squelch=True)
    assert clips(folder) == before


def test_no_squelch_means_record_everything(station):
    """A broadcast band sets no squelch, and the honest reading of that is
    "nothing is gating this", not "record nothing"."""
    play, radio, folder = station()
    play.play(6.0, MUSIC, squelch=None)
    assert radio.status.recording
    radio.stop()
    assert len(clips(folder)) == 1


def test_a_retune_closes_what_belonged_to_the_old_frequency(station):
    """A file whose name says one frequency must not contain another."""
    play, radio, folder = station()
    play.play(5.0, SPEECH, center_hz=162.55e6)
    play.play(5.0, SPEECH, center_hz=94.9e6)
    radio.stop()
    names = clips(folder)
    assert len(names) == 2
    assert any(name.startswith("RR-162.550MHz-") for name in names)
    assert any(name.startswith("RR-94.900MHz-") for name in names)


def test_something_else_borrowing_the_radio_closes_what_was_open(station):
    """A sweep, the aircraft screen and a monitor session all simply stop
    feeding this. Noticing the gap here means no call at each of those sites,
    and none to remember when the next such feature is written."""
    play, radio, folder = station()
    play.play(5.0, SPEECH)
    assert radio.status.recording
    # Five seconds of the radio being somewhere else entirely.
    play.clock.now += 5.0
    play.play(5.0, SPEECH)
    radio.stop()
    assert len(clips(folder)) == 2


def test_a_gain_probe_is_not_an_interruption(station):
    """340 ms of no capture is normal and must not split a recording."""
    play, radio, folder = station()
    play.play(5.0, SPEECH)
    play.clock.now += 0.4
    play.play(5.0, SPEECH)
    radio.stop()
    assert len(clips(folder)) == 1


def test_a_stereo_station_is_recorded_in_stereo(station):
    play, radio, folder = station()
    play.play(6.0, MUSIC, stereo=True)
    radio.stop()
    written = sorted(folder.glob("RR-*.mp3"))[0]
    # Stereo is twice the bitrate, so six seconds is about 96 kB rather than 48.
    assert written.stat().st_size > 80_000


# -- songs -------------------------------------------------------------------


def test_a_song_is_saved_named_and_tagged(station):
    play, radio, folder = station(songs=True)
    play.play(20.0, SPEECH, text="The Mix 94.9")
    play.play(90.0, MUSIC, text="Fleetwood Mac - Dreams")
    play.play(40.0, SPEECH, text="The Mix 94.9")
    radio.stop()

    assert songs(folder) == ["Fleetwood Mac - Dreams.mp3"]
    tags = encode.read_tags(folder / "Songs" / "Fleetwood Mac - Dreams.mp3")
    assert tags.artist == "Fleetwood Mac"
    assert tags.title == "Dreams"
    assert "94.9" in tags.album
    assert tags.genre == "Rock"
    assert "94.900 MHz" in tags.comment


def test_the_song_starts_before_the_station_says_so(station):
    """RadioText is late, so the boundary steps back to where the sound
    changed. Without it the intro of every song is missing."""
    play, radio, folder = station(songs=True)
    play.play(30.0, SPEECH, text="The Mix 94.9")
    # The music starts here, but the station does not say so for ten seconds.
    play.play(10.0, MUSIC, text="The Mix 94.9")
    play.play(80.0, MUSIC, text="Fleetwood Mac - Dreams")
    play.play(40.0, SPEECH, text="The Mix 94.9")
    radio.stop()

    saved = folder / "Songs" / "Fleetwood Mac - Dreams.mp3"
    assert saved.exists()
    # 90 seconds of music, not the 80 the station admitted to.
    assert seconds_of(saved) > 85.0


def test_an_advertisement_is_not_saved_even_when_it_parses(station):
    """Two independent checks, and the advertisement fails both. This one is
    given a title so that only the audio can reject it."""
    play, radio, folder = station(songs=True)
    play.play(20.0, MUSIC, text="The Mix 94.9")
    play.play(90.0, SPEECH, text="Bobs Motors - Best Deals In Town")
    play.play(40.0, MUSIC, text="The Mix 94.9")
    radio.stop()

    assert songs(folder) == []
    assert "did not sound like music" in radio.status.message


def test_the_encoder_is_handed_nothing_above_the_broadcast_limit():
    """Analog FM stereo stops at 15 kHz - the 19 kHz pilot has to sit above
    the audio - so everything a demodulator produces above that is hiss the
    transmitter never sent. It is the most expensive thing a lossy encoder
    can be given: noise looks like signal at every frequency and there is
    nothing to mask it behind, so the bits go there instead of on the record.
    """
    from bettersdr.audio.encode import BROADCAST_AUDIO_HZ
    from bettersdr.dsp.filters import LowPass

    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.1, (RATE, 2)).astype(np.float32)
    out = LowPass(RATE, BROADCAST_AUDIO_HZ).process(noise)
    freqs = np.fft.rfftfreq(RATE, 1 / RATE)

    def level(block, low, high):
        spectrum = np.abs(np.fft.rfft(block[:, 0]))
        return 20 * np.log10(spectrum[(freqs >= low) & (freqs < high)].mean() + 1e-12)

    # Everything the broadcast carries is untouched...
    kept = level(noise, 1_000, 14_000)
    assert level(out, 1_000, 14_000) == pytest.approx(kept, abs=0.5)
    # ...and everything above it is gone.
    assert level(out, 19_000, 24_000) < level(noise, 19_000, 24_000) - 40


def test_the_thresholds_sit_between_what_was_measured_off_air():
    """The readings the two thresholds were set from, written down.

    Five real songs and two real breaks off 96.5 MHz. The point of keeping
    them here is that the music shares *overlap* - a rap record read as music
    less often than a news bulletin did - so anybody raising
    `MIN_MUSIC_SHARE` to catch a break will find they have thrown away songs,
    and will find out from this rather than from an empty folder a week
    later. The speech shares do not overlap at all.
    """
    songs = ((0.30, 0.05), (0.53, 0.00), (0.67, 0.03), (0.53, 0.00), (0.46, 0.04))
    breaks = ((0.14, 0.18), (0.37, 0.25))
    for music, speech in songs:
        assert music >= rr.MIN_MUSIC_SHARE
        assert speech <= rr.MAX_SPEECH_SHARE
    for music, speech in breaks:
        assert music < rr.MIN_MUSIC_SHARE or speech > rr.MAX_SPEECH_SHARE


def test_a_jingle_is_too_short_to_be_a_song(station):
    """...and what ends it is the music stopping, not the title going away.

    A jingle is followed by somebody talking, which is why the segment closes
    where it does. Uninterrupted music after it would be a station that had
    simply stopped naming what it was playing, and running on is the right
    answer to that - see `test_a_title_that_goes_away_mid_record`.
    """
    play, radio, folder = station(songs=True)
    play.play(20.0, MUSIC, text="The Mix 94.9")
    play.play(30.0, MUSIC, text="Somebody - A Jingle")
    play.play(40.0, SPEECH, text="The Mix 94.9")
    radio.stop()
    assert songs(folder) == []


def test_a_title_that_goes_away_mid_record_does_not_split_the_song(station):
    """Measured off 96.5 MHz: a play of `Teenage Dirtbag` lost its own title
    for 32 seconds in the middle while the record kept playing. Closing on
    that gap made two files out of one play - copy 1 and copy 2 of a song
    played once - which is worse than either keeping it whole or losing it."""
    play, radio, folder = station(songs=True)
    play.play(60.0, MUSIC, text="Wheatus - Teenage Dirtbag")
    play.play(32.0, MUSIC, text="Playing What We Want")
    play.play(60.0, MUSIC, text="Wheatus - Teenage Dirtbag")
    play.play(40.0, SPEECH, text="Playing What We Want")
    radio.stop()
    assert songs(folder) == ["Wheatus - Teenage Dirtbag.mp3"]


def test_a_station_that_stops_naming_anything_still_closes_the_song(station):
    """The backstop. Without it a segue-heavy hour is one file named after
    whatever was on when the titles dried up."""
    play, radio, folder = station(songs=True)
    play.play(60.0, MUSIC, text="Wheatus - Teenage Dirtbag")
    play.play(90.0, MUSIC, text="Playing What We Want")
    assert radio.status.song_title == ""
    radio.stop()
    assert songs(folder) == ["Wheatus - Teenage Dirtbag.mp3"]


def test_a_station_that_alternates_its_slogan_still_yields_a_song(station):
    """The failure nobody would report: a great many stations flip between
    their slogan and the title every few seconds, and treating each flip as a
    boundary produces nothing but eight-second fragments."""
    play, radio, folder = station(songs=True)
    play.play(20.0, SPEECH, text="The Mix 94.9")
    for _ in range(10):
        play.play(5.0, MUSIC, text="Fleetwood Mac - Dreams")
        play.play(5.0, MUSIC, text="The Mix 94.9")
    play.play(40.0, SPEECH, text="The Mix 94.9")
    radio.stop()

    assert songs(folder) == ["Fleetwood Mac - Dreams.mp3"]
    assert seconds_of(folder / "Songs" / "Fleetwood Mac - Dreams.mp3") > 80.0


def test_five_copies_and_then_no_more(station):
    play, radio, folder = station(songs=True)
    for number in range(MAX_COPIES + 2):
        # Far enough apart that the tracker never mistakes it for a slogan.
        play.play(60.0, MUSIC, text=f"Somebody {number} - Song {number}")
        play.play(60.0, MUSIC, text="Rush - Tom Sawyer")
    radio.stop()

    kept = [name for name in songs(folder) if "Tom Sawyer" in name]
    assert len(kept) == MAX_COPIES
    assert "Rush - Tom Sawyer (5).mp3" in kept
    assert radio.status.songs_skipped >= 2
    assert "Already have 5" in radio.status.message


def test_a_fragmenting_station_writes_nothing_at_all(station):
    """The lazy open. A segment that never lasts long enough to be a song
    must not create and delete a file every few seconds."""
    play, radio, folder = station(songs=True)
    for number in range(20):
        play.play(5.0, MUSIC, text=f"Artist {number} - Song {number}")
    radio.stop()
    assert songs(folder) == []
    assert not list((folder / "Songs").glob(".*")) or all(
        path.name.endswith(".json")
        for path in (folder / "Songs").glob(".*")
    )


def test_radiotext_that_has_stopped_moving_records_one_song_only(
    station, monkeypatch
):
    """A stuck title is closed and kept at the cap, and must not then start
    itself over. The claim stays, marked refused, until the title changes."""
    monkeypatch.setattr(rr, "MAX_SONG_S", 60.0)
    play, radio, folder = station(songs=True)
    play.play(20.0, SPEECH, text="The Mix 94.9")
    play.play(240.0, MUSIC, text="Fleetwood Mac - Dreams")
    radio.stop()
    kept = [name for name in songs(folder) if "Dreams" in name]
    assert kept == ["Fleetwood Mac - Dreams.mp3"]


def test_a_station_with_no_song_information_is_just_recorded(station):
    """The answer for AM, and for FM without RadioText: no songs, and the
    frequency still recorded exactly as any other."""
    play, radio, folder = station(songs=True)
    play.play(120.0, MUSIC, text="")
    radio.stop()
    assert songs(folder) == []
    assert len(clips(folder)) == 1
    assert radio.status.songs_saved == 0


def test_songs_are_off_unless_asked_for(station):
    play, radio, folder = station(songs=False)
    play.play(120.0, MUSIC, text="Fleetwood Mac - Dreams")
    radio.stop()
    assert not (folder / "Songs").exists() or songs(folder) == []
    assert len(clips(folder)) == 1


def test_a_retune_mid_song_keeps_what_it_had(station):
    """What was recorded before the retune really was that song."""
    play, radio, folder = station(songs=True)
    play.play(20.0, SPEECH, text="The Mix 94.9")
    play.play(90.0, MUSIC, text="Fleetwood Mac - Dreams")
    play.play(5.0, MUSIC, text="Fleetwood Mac - Dreams", center_hz=98.1e6)
    radio.stop()
    assert songs(folder) == ["Fleetwood Mac - Dreams.mp3"]


def test_the_status_reports_the_song_being_recorded(station):
    play, radio, folder = station(songs=True)
    play.play(20.0, SPEECH, text="The Mix 94.9")
    play.play(60.0, MUSIC, text="Fleetwood Mac - Dreams")
    status = radio.status
    assert status.song_title == "Fleetwood Mac - Dreams"
    assert status.song_seconds > 55.0
    assert status.songs_enabled
