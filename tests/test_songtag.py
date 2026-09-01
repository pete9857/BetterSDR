"""Reading a song out of RadioText, and knowing when it is not one.

The parsing half is a pure function and is tested as one. The tracking half
is entirely about elapsed time - which text has gone on turning up, and for
how long it was left up each time - so it runs against a clock the test moves
by hand, the same bargain `tests/test_monitor.py` made.
"""

from __future__ import annotations

import pytest

from bettersdr.decode import songtag


@pytest.mark.parametrize(
    ("text", "artist", "title"),
    [
        ("Fleetwood Mac - Dreams", "Fleetwood Mac", "Dreams"),
        ("Rush – Tom Sawyer", "Rush", "Tom Sawyer"),
        ("Queen — Bohemian Rhapsody", "Queen", "Bohemian Rhapsody"),
        ("The Beatles / Hey Jude", "The Beatles", "Hey Jude"),
        ("a-ha - Take On Me", "a-ha", "Take On Me"),
        ("Spider-Man Theme - Michael Buble", "Spider-Man Theme", "Michael Buble"),
    ],
)
def test_the_usual_shapes_are_read(text, artist, title):
    tag = songtag.parse(text)
    assert tag is not None
    assert (tag.artist, tag.title) == (artist, title)


def test_by_reads_the_other_way_round():
    """`Title by Artist` puts the artist second, which is the whole point."""
    tag = songtag.parse("Tom Sawyer by Rush")
    assert tag is not None
    assert (tag.artist, tag.title) == ("Rush", "Tom Sawyer")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The Best Music Variety",
        "Listen at www.example.com - always",
        "Text 55512 - to win",
        "Call 5551000 - now",
        "Studio line - 555 1000",
        "A - ",
        " - B",
        "x - y",
        "Sponsored - by somebody",
    ],
)
def test_what_is_not_a_song_is_refused(text):
    """None is the common answer, and much better than a confident wrong one."""
    assert songtag.parse(text) is None


def test_the_station_is_not_its_own_artist():
    assert songtag.parse("KUOW - Dreams", station="KUOW") is None
    assert songtag.parse("Fleetwood Mac - Dreams", station="KUOW") is not None


def test_padding_and_control_characters_are_stripped():
    """RDS pads to a segment boundary and ends with a carriage return."""
    tag = songtag.parse("  Rush\r-\tTom  Sawyer   \r")
    assert tag is not None
    assert (tag.artist, tag.title) == ("Rush", "Tom Sawyer")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (("Guns N' Roses", "Sweet Child"), ("Guns n Roses", "Sweet Child")),
        (("Beyonce", "Halo"), ("Beyoncé", "Halo")),
        (("ABBA", "Waterloo"), ("abba", "  waterloo ")),
    ],
)
def test_the_same_song_written_differently_is_one_song(left, right):
    """Stations re-key their playout metadata; the five-copies rule must not
    hand out five more copies because one of them dropped an apostrophe."""
    assert songtag.key_for(*left) == songtag.key_for(*right)


def test_a_remix_is_not_the_original():
    assert songtag.key_for("Rush", "Tom Sawyer") != songtag.key_for(
        "Rush", "Tom Sawyer (Live)"
    )


# -- the tracker -------------------------------------------------------------


def test_only_a_change_is_reported():
    tracker = songtag.TextTracker()
    assert tracker.update("Rush - Tom Sawyer", 0.0) == "Rush - Tom Sawyer"
    assert tracker.update("Rush - Tom Sawyer", 1.0) is None
    assert tracker.update("Queen - Radio Ga Ga", 2.0) == "Queen - Radio Ga Ga"


def test_a_slogan_that_keeps_flashing_up_becomes_the_station():
    """The alternating station: slogan, title, slogan, title, all afternoon.

    Both texts recur, so counting recurrences would brand them both. What
    separates them is that the slogan goes on doing it after the song ends.
    """
    tracker = songtag.TextTracker()
    now = 0.0
    # Eight seconds each, for rather longer than a song.
    while now < songtag.IDLE_LIFETIME_S + 60.0:
        tracker.update("The Mix 94.9", now)
        now += 8.0
        tracker.update("Rush - Tom Sawyer", now)
        now += 8.0
    assert tracker.is_idle("The Mix 94.9")


def test_a_title_left_up_for_a_whole_song_is_never_branded():
    """The rule that protects the third, fourth and fifth copies.

    A song replayed inside the forgetting window accumulates a lifetime just
    like a slogan does. What it also does, and a slogan does not, is hold the
    field for minutes at a time.
    """
    tracker = songtag.TextTracker()
    now = 0.0
    for _ in range(12):
        # Up for two minutes - a song - then away for four.
        for _ in range(24):
            tracker.update("Rush - Tom Sawyer", now)
            now += 5.0
        for _ in range(48):
            tracker.update("Somebody Else - Another", now)
            now += 5.0
    assert not tracker.is_idle("Rush - Tom Sawyer")
    assert tracker.tag("Rush - Tom Sawyer") is not None


def test_a_long_absence_starts_the_clock_again():
    """A track played again in the evening is a fresh occasion."""
    tracker = songtag.TextTracker()
    tracker.update("Rush - Tom Sawyer", 0.0)
    tracker.update("Something Else", 10.0)
    tracker.update("Rush - Tom Sawyer", songtag.IDLE_FORGET_S + 100.0)
    assert not tracker.is_idle("Rush - Tom Sawyer")


def test_an_idle_text_yields_no_song_even_though_it_parses():
    """The hardest station: its slogan is shaped exactly like a song.

    `The Mix - Best Variety` parses as well as any title does, so nothing
    about the string itself can separate them. What separates them is that the
    songs keep changing and the slogan does not, and the tracker has to run
    for longer than a song before it can know that. Titles here are three
    minutes each, which is what a station actually does; a test that held one
    title up for a quarter of an hour would be asserting against a station
    that does not exist.
    """
    tracker = songtag.TextTracker()
    now = 0.0
    titles = [f"Artist {n} - Song {n}" for n in range(8)]
    for title in titles:
        for _ in range(11):  # 176 seconds, alternating every eight
            tracker.update("The Mix - Best Variety", now)
            now += 8.0
            tracker.update(title, now)
            now += 8.0
    assert songtag.parse("The Mix - Best Variety") is not None
    assert tracker.is_idle("The Mix - Best Variety")
    assert tracker.tag("The Mix - Best Variety") is None
    # ...and no title was caught by the same net.
    assert all(not tracker.is_idle(title) for title in titles)
    assert tracker.tag(titles[-1]) is not None


def test_the_station_is_only_learnt_after_a_song_has_gone_by():
    """The known limitation, asserted rather than left to be discovered.

    Until the slogan has been on the air for longer than a song could be,
    there is no evidence that separates it from one - so on a station whose
    slogan also parses, the first quarter of an hour produces fragments. It is
    written down here so that changing it is a decision rather than a
    surprise.
    """
    tracker = songtag.TextTracker()
    now = 0.0
    while now < songtag.IDLE_LIFETIME_S / 2:
        tracker.update("The Mix - Best Variety", now)
        now += 8.0
        tracker.update("Rush - Tom Sawyer", now)
        now += 8.0
    assert not tracker.is_idle("The Mix - Best Variety")


def test_resetting_forgets_the_station():
    tracker = songtag.TextTracker()
    tracker.update("The Mix 94.9", 0.0)
    tracker.reset()
    assert tracker.text == ""
    assert not tracker.is_idle("The Mix 94.9")


# -- a station that writes three fields --------------------------------------
#
# 96.5 MHz here transmits `96.5 Jack FM - The Real Slim Shady - Eminem`. Read
# as two halves on the first separator that is an artist called `96.5 Jack FM`
# playing a song called `The Real Slim Shady - Eminem`, which is a perfectly
# well-formed answer and is wrong about every song the station plays.


def test_every_separator_splits_not_just_the_first():
    assert songtag.split_fields("96.5 Jack FM - The Real Slim Shady - Eminem") == (
        "96.5 Jack FM",
        "The Real Slim Shady",
        "Eminem",
    )


def test_the_station_naming_itself_is_set_aside():
    tag = songtag.parse(
        "96.5 Jack FM - The Real Slim Shady - Eminem", "KCZC", 96.5e6
    )
    assert tag is not None
    assert {tag.artist, tag.title} == {"The Real Slim Shady", "Eminem"}


def test_the_dial_the_receiver_is_on_is_recognised_immediately():
    """The frequency needs no learning: a field carrying it is the station."""
    assert songtag.is_station_field("96.5", frequency_hz=96.5e6)
    assert songtag.is_station_field("Jack 96.5 FM", frequency_hz=96.5e6)
    assert not songtag.is_station_field("Eminem", frequency_hz=96.5e6)


def test_a_dial_position_with_a_band_word_is_the_station_anywhere():
    assert songtag.is_station_field("102.5 FM")
    assert songtag.is_station_field("KZOK 102.5 Radio")
    # ...and a number in a title is not.
    assert not songtag.is_station_field("Summer of 69")


def test_a_station_named_after_a_word_does_not_eat_an_artist():
    """A program service name of JACK, MIX or KISS is a word, and a rule that
    dropped any field containing it would throw away `Jack Johnson` for ever.
    The name has to be most of the field, not merely somewhere in it."""
    assert songtag.is_station_field("JACK", "JACK")
    assert songtag.is_station_field("KCZC-FM", "KCZC")
    assert not songtag.is_station_field("Jack Johnson", "JACK")
    tag = songtag.parse("Jack Johnson - Better Together", "JACK", 96.5e6)
    assert tag is not None
    assert (tag.artist, tag.title) == ("Jack Johnson", "Better Together")


def test_three_unrecognisable_fields_are_refused_rather_than_guessed():
    """`A - B - C` is `Station - Title - Artist` on one station and
    `Artist - Title - Part 2` on another, and the string does not say."""
    assert songtag.parse("Aaa - Bbb - Ccc") is None


def test_a_slogan_with_no_dial_in_it_is_learnt_from_its_company():
    tracker = songtag.TextTracker()
    for when, text in enumerate(
        (
            "Best Variety - The Real Slim Shady - Eminem",
            "Best Variety - Come Out And Play - Offspring",
        )
    ):
        tracker.update(text, when * 300.0)
    tag = tracker.tag(tracker.text, "KCZC")
    assert tag is not None
    assert {tag.artist, tag.title} == {"Come Out And Play", "Offspring"}


def test_an_artist_that_comes_back_is_not_mistaken_for_the_slogan():
    """The two rules share one counter and must not share one threshold.

    After a second Eminem song, `Eminem` has been seen beside two sets of
    companions exactly as the slogan has. What separates them is that the
    slogan is in nearly every message and an artist is not; without that,
    the field that is about to become the artist is thrown away instead.
    """
    tracker = songtag.TextTracker()
    for when, text in enumerate(
        (
            "96.5 Jack FM - The Real Slim Shady - Eminem",
            "96.5 Jack FM - Come Out And Play - Offspring",
            "96.5 Jack FM - Lose Yourself - Eminem",
        )
    ):
        tracker.update(text, when * 300.0)
    assert tracker.is_station_text("96.5 Jack FM")
    assert not tracker.is_station_text("Eminem")


def test_a_station_that_names_itself_first_names_the_song_next():
    """`Slogan - Title - Artist`, which is the opposite of the bare
    two-field convention and is what 96.5 MHz actually transmits. Reading it
    the conventional way round named every file on the station backwards."""
    tag = songtag.parse(
        "96.5 Jack FM - Seven Nation Army - White Stripes", "KCZC", 96.5e6
    )
    assert tag is not None
    assert (tag.artist, tag.title) == ("White Stripes", "Seven Nation Army")


def test_the_order_is_overturned_by_an_artist_heard_twice():
    """The prior above is a prior, not a fact. A station writing
    `Slogan - Artist - Title` is corrected by the only evidence in the
    stream: an artist comes round again with a different song, a title
    does not.
    """
    tracker = songtag.TextTracker()
    songs = (
        "96.5 Jack FM - Eminem - The Real Slim Shady",
        "96.5 Jack FM - Offspring - Come Out And Play",
        "96.5 Jack FM - Eminem - Lose Yourself",
        "96.5 Jack FM - Fleetwood Mac - Dreams",
    )
    read = []
    for when, text in enumerate(songs):
        tracker.update(text, when * 300.0)
        tag = tracker.tag(tracker.text, "KCZC", 96.5e6)
        read.append(None if tag is None else tag.display)
    # The first two are read the way the station that was measured writes
    # them, and are wrong about this one; the third carries the evidence and
    # every one after it is right. That is the honest limitation, asserted
    # rather than hidden.
    assert read[0] == "The Real Slim Shady - Eminem"
    assert read[2] == "Eminem - Lose Yourself"
    assert read[3] == "Fleetwood Mac - Dreams"


def test_a_plain_two_field_message_keeps_the_convention():
    """Only a station that wrote more fields than a song has gets its order
    questioned. `A - B` with nothing set aside is what almost every station
    in the world writes, and there is nothing to weigh against it - so a
    title that happens to recur must not be able to flip a whole session."""
    tracker = songtag.TextTracker()
    for when, text in enumerate(
        (
            "Somebody One - Another Song",
            "Somebody Two - Another Song",
            "Rush - Tom Sawyer",
        )
    ):
        tracker.update(text, when * 300.0)
    tag = tracker.tag("Rush - Tom Sawyer", "KUOW", 94.9e6)
    assert tag is not None
    assert (tag.artist, tag.title) == ("Rush", "Tom Sawyer")
