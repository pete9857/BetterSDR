"""Tests for the HD Radio (NRSC-5) child process and its metadata.

The decoder itself is somebody else's program, so there is nothing here that
tries to verify HDC audio. What is ours is the boundary: the log lines turned
into state, the pipes that must not block the DSP thread, and the process
started and stopped without leaving anything running.

The log transcript below is a real one, taken from nrsc5 decoding upstream's
own sample recording of KUT Austin. Testing the parser against a recording
rather than against invented lines is the point: the format is nrsc5's and we
do not get to choose it.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from bettersdr.decode.hdradio import (
    AUDIO_RATE_HZ,
    SAMPLE_RATE_HZ,
    HdMetadata,
    HdRadio,
    HdState,
    available,
    executable,
)

# Verbatim from a session on 2026-08-28, including the audio driver's warning
# and the negative MER of a decoder that has not settled yet.
TRANSCRIPT = """ao_raw WARNING: Driver raw does not support automatic channel mapping;
\tRouting only L/R channels to output.
19:32:27 Synchronized
19:32:27 Frequency offset: 99 Hz
19:32:27 Primary service mode: 1
19:32:27 Station name: KUT
19:32:27 Country: AB, FCC facility ID: 21
19:32:27 MER: -13.4 dB (lower), -13.0 dB (upper)
19:32:27 BER: 0.197337, avg: 0.197337, min: 0.197337, max: 0.197337
19:32:27 Lost synchronization
19:32:27 Synchronized
19:32:27 MER: 13.4 dB (lower), 12.4 dB (upper)
19:32:27 BER: 0.000186, avg: 0.098762, min: 0.000186, max: 0.197337
19:32:27 Audio service 0: public, type: None, codec: 0, blend: 2, gain: 0 dB, \
delay: 96, latency: 8
19:32:27 Title: You're Listening to Q with Jian Ghomeshi
19:32:27 Artist:
19:32:27 Audio service 1: restricted, type: Public, codec: 0, blend: 0, \
gain: 0 dB, delay: 0, latency: 8
19:32:27 Slogan: The University of Texas at Austin
19:32:27 Audio bit rate: 63.7 kbps
"""


def _played(transcript: str = TRANSCRIPT, program: int = 0) -> HdState:
    metadata = HdMetadata(program)
    for line in transcript.splitlines():
        metadata.feed(line)
    return metadata.state


# -- the log ---------------------------------------------------------------


def test_transcript_yields_everything_the_screen_needs() -> None:
    state = _played()
    assert state.synced
    assert state.station == "KUT"
    assert state.slogan == "The University of Texas at Austin"
    assert state.title == "You're Listening to Q with Jian Ghomeshi"
    assert state.bit_rate_kbps == pytest.approx(63.7)
    assert state.ber == pytest.approx(0.000186)


def test_unknown_lines_are_ignored_rather_than_failing() -> None:
    """The log carries traffic maps, alerts and driver warnings we never read.

    A parser that objected to any of them would be broken by the next nrsc5
    release, so everything unrecognised has to pass through silently.
    """
    metadata = HdMetadata()
    for line in (
        "ao_raw WARNING: Driver raw does not support automatic channel mapping;",
        "19:32:27 HERE Image: type=TRAFFIC, seq=3, n1=1, n2=2",
        "19:32:27 Alert: [12345] Severe thunderstorm",
        "19:32:27 Something nrsc5 has not invented yet",
        "",
        "   ",
    ):
        metadata.feed(line)
    assert metadata.state == HdState()


def test_mer_reports_the_weaker_sideband() -> None:
    """Hybrid IBOC has two, and the weaker one is what fails first."""
    state = _played()
    assert state.mer_db == pytest.approx(12.4)


def test_losing_sync_keeps_the_station_but_clears_the_flag() -> None:
    """Sync comes and goes at the edge of coverage.

    Blanking the screen each time says "nothing is here" about a station that
    is still there, so only the flag and the counter move.
    """
    state = _played(TRANSCRIPT + "19:32:28 Lost synchronization\n")
    assert not state.synced
    assert state.lost_sync == 2
    assert state.station == "KUT"
    assert state.title


def test_programmes_are_listed_in_order_with_access() -> None:
    state = _played()
    assert [program.label for program in state.programs] == ["HD1", "HD2"]
    assert not state.programs[0].restricted
    assert state.programs[1].restricted


def test_a_programme_with_no_declared_type_has_no_type() -> None:
    """nrsc5 prints "None" there, which is a name for the absence of a type."""
    state = _played()
    assert state.programs[0].kind == ""
    assert state.programs[1].kind == "Public"


def test_a_repeated_service_announcement_does_not_duplicate_it() -> None:
    state = _played(TRANSCRIPT + TRANSCRIPT)
    assert len(state.programs) == 2


def test_track_joins_artist_and_title_and_survives_either_being_empty() -> None:
    assert _played().track == "You're Listening to Q with Jian Ghomeshi"
    metadata = HdMetadata()
    metadata.feed("19:32:27 Artist: Steely Dan")
    metadata.feed("19:32:27 Title: Peg")
    assert metadata.state.track == "Steely Dan - Peg"
    assert HdState().track == ""


def test_label_names_the_subchannel_the_way_a_car_radio_does() -> None:
    assert HdState(program=0).label == "HD1"
    assert HdState(program=2).label == "HD3"


# -- finding the decoder ---------------------------------------------------


def test_a_missing_binary_is_a_missing_feature_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Everything HD is gated on this, so it must never raise."""
    monkeypatch.setenv("BETTERSDR_NRSC5", os.path.join("nowhere", "nrsc5.exe"))
    assert executable() is None
    assert not available()
    radio = HdRadio()
    assert not radio.start()
    assert "not installed" in radio.snapshot().error
    assert not radio.running
    # And it stays usable: nothing queued, nothing to collect, no exception.
    radio.feed(b"\x80" * 1024)
    assert radio.audio().shape == (0, 2)
    radio.stop()


# -- the pipe boundary -----------------------------------------------------


def test_feed_before_start_is_a_no_op() -> None:
    radio = HdRadio()
    radio.feed(b"\x80" * 4096)
    assert radio.dropped_blocks == 0
    assert radio.audio().shape == (0, 2)


class _StalledProcess:
    """A child that is alive and reading nothing, which is the case that hurts.

    A real decoder never fills the queue - nrsc5 consumes many times faster
    than real time - so the overflow path would otherwise never be tested at
    all, and it is the path that decides whether a wedged child process stops
    the radio.
    """

    stdin = stdout = stderr = None

    def poll(self) -> None:
        return None


def test_the_input_queue_drops_the_oldest_rather_than_growing() -> None:
    """A full pipe must cost the past, not stall the DSP thread."""
    radio = HdRadio()
    radio._process = _StalledProcess()  # type: ignore[assignment]
    radio._input_cap = 4096
    assert radio.running
    for index in range(20):
        began = time.perf_counter()
        radio.feed(bytes([index]) + b"\x80" * 1023)
        assert time.perf_counter() - began < 0.05
    assert radio._input_bytes <= radio._input_cap
    assert radio.dropped_blocks > 0
    # What survived is the newest, so the decoder resumes on current air.
    assert radio._input[-1][0] == 19
    radio._process = None


def test_snapshot_is_safe_before_anything_has_started() -> None:
    state = HdRadio().snapshot()
    assert not state.running
    assert not state.playing
    assert state.audio_seconds == 0.0


# -- against the real decoder ----------------------------------------------

needs_nrsc5 = pytest.mark.skipif(
    not available(), reason="bundled nrsc5 not present in this checkout"
)


@needs_nrsc5
def test_the_child_process_starts_consumes_iq_and_stops_cleanly() -> None:
    """Noise will never sync, and that is the point.

    What is being tested is the plumbing either side of the decoder: it
    starts, it swallows real-time IQ without the feeding thread blocking, and
    `stop` leaves no process and no threads behind.
    """
    radio = HdRadio()
    assert radio.start()
    try:
        assert radio.running
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, size=SAMPLE_RATE_HZ // 4, dtype=np.uint8)
        for _ in range(4):
            began = time.perf_counter()
            radio.feed(noise.tobytes())
            # The whole contract of `feed`: it returns at once whatever the
            # child process is doing.
            assert time.perf_counter() - began < 0.05
        assert radio.dropped_blocks == 0
    finally:
        radio.stop()
    assert not radio.running
    assert not radio.snapshot().running
    assert not any(thread.is_alive() for thread in radio._threads)


@needs_nrsc5
def test_stop_is_idempotent_and_survives_never_having_started() -> None:
    radio = HdRadio()
    radio.stop()
    assert radio.start()
    radio.stop()
    radio.stop()
    assert not radio.running


@needs_nrsc5
def test_the_programme_number_reaches_the_command_line() -> None:
    radio = HdRadio(program=1)
    assert radio.start()
    try:
        assert radio.snapshot().label == "HD2"
    finally:
        radio.stop()


def test_a_programme_number_outside_the_standard_is_clamped() -> None:
    assert HdRadio(program=99).program == 7
    assert HdRadio(program=-3).program == 0


def test_the_rates_are_the_ones_nrsc5_fixes() -> None:
    """Both are constants of the standard, and both are awkward on purpose.

    1,488,375 is not a whole multiple of 48 kHz - 31.0078 - which is why an HD
    session cannot run the `dsp/demod.py` skeleton and why the audio has to be
    resampled from 44,100 on the way back.
    """
    assert SAMPLE_RATE_HZ == 1_488_375
    assert AUDIO_RATE_HZ == 44_100
    assert SAMPLE_RATE_HZ % 48_000 != 0
