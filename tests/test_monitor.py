"""The activity ledger and the scanner behaviour built on top of it.

Everything here runs against a clock the test moves by hand. That is not
convenience: the release timer, the revisit policy and the "come back sooner
if somebody was talking" rule are all statements about elapsed time, and a
test that waited for real seconds would either be slow or would assert
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.scan import monitor as mon
from bettersdr.scan import voice
from bettersdr.scan.classifier import Signal, classify
from bettersdr.scan.detector import Detection
from tests import synth_audio as sa


class Clock:
    """A clock that only moves when the test says so."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += float(seconds)


def signal(hz: float, snr_db: float = 25.0, width_hz: float = 12_500.0) -> Signal:
    return classify(
        Detection(
            center_hz=hz,
            bandwidth_hz=width_hz,
            peak_hz=hz,
            peak_dbfs=-30.0 + snr_db,
            floor_dbfs=-30.0,
        )
    )


def verdict(kind: str) -> voice.Verdict:
    """A real verdict, off real audio, rather than a stub of one.

    The monitor branches on `carries_audio` and `is_voice`, and both are
    `voice.Verdict`'s own answers - so a hand-built stand-in here would be a
    test of a copy of the rule rather than of the rule.
    """
    made = {
        voice.VOICE: lambda: sa.speech(0.8, seed=1),
        voice.MUSIC: lambda: sa.music(0.8, seed=1),
        voice.DATA: lambda: sa.data(0.8, seed=1),
        voice.NOISE: lambda: sa.static(0.8, seed=1),
        voice.TONE: lambda: sa.tone(0.8),
        voice.SILENCE: lambda: sa.silence(0.8),
    }[kind]()
    answer = voice.classify(made)
    assert answer.kind == kind, f"the material for {kind} now reads as {answer.kind}"
    return answer


def build(**kwargs) -> tuple[mon.Monitor, Clock]:
    clock = Clock()
    return mon.Monitor(150e6, 160e6, clock=clock, **kwargs), clock


# -- the ledger -------------------------------------------------------------


def test_a_channel_heard_once_is_not_yet_a_channel():
    """The persistence gate, in the place the monitor keeps it.

    A single sweep's threshold crossing is a noise peak as often as it is a
    transmitter, which is why the sweep repeats itself three times. Here the
    sighting count does that job and is also the number the user came for.
    """
    watcher, _ = build()
    watcher.note_pass([signal(155e6)])
    assert watcher.snapshot().channels == ()
    watcher.note_pass([signal(155e6)])
    assert len(watcher.snapshot().channels) == 1


def test_sightings_accumulate_across_passes():
    watcher, _ = build()
    for _ in range(5):
        watcher.note_pass([signal(155e6)])
    (channel,) = watcher.snapshot().channels
    assert channel.sightings == 5
    assert channel.passes == 5
    assert channel.duty == 1.0


def test_a_channel_up_half_the_time_reports_half():
    watcher, _ = build()
    for index in range(6):
        watcher.note_pass([signal(155e6)] if index % 2 == 0 else [])
    (channel,) = watcher.snapshot().channels
    assert channel.sightings == 3
    assert channel.duty == pytest.approx(0.5)


def test_duty_is_measured_from_when_a_channel_was_first_heard():
    """A channel found late and busy ever since is busy, not idle.

    Dividing by the whole session would report it at a few percent and take
    an hour to correct itself, which is the same fault as a filter that
    persists between sittings: the number would be defensible and wrong.
    """
    watcher, _ = build()
    for _ in range(20):
        watcher.note_pass([signal(150.5e6)])
    for _ in range(4):
        watcher.note_pass([signal(150.5e6), signal(155e6)])
    latecomer = next(
        c for c in watcher.snapshot().channels if c.frequency_hz == pytest.approx(155e6)
    )
    assert latecomer.duty == 1.0


def test_only_what_was_heard_this_pass_is_active():
    watcher, _ = build()
    watcher.note_pass([signal(155e6), signal(156e6)])
    watcher.note_pass([signal(155e6)])
    active = [c.frequency_hz for c in watcher.snapshot().channels if c.active]
    assert active == [pytest.approx(155e6)]


def test_the_most_confident_view_of_a_channel_is_the_one_kept():
    """One transmitter classifies differently from pass to pass.

    Same rule as the sweeper's `_remember`, and the same fault without it: a
    station reported correctly on one pass and as a bare carrier on the next
    would flicker between two descriptions of itself.
    """
    watcher, _ = build()
    watcher.note_pass([signal(155e6, width_hz=800.0)])
    watcher.note_pass([signal(155e6, width_hz=12_500.0)])
    (channel,) = watcher.snapshot().channels
    assert channel.signal.bandwidth_hz == 12_500.0


def test_readings_a_few_kilohertz_apart_are_one_channel():
    watcher, _ = build()
    watcher.note_pass([signal(155_000_000)])
    watcher.note_pass([signal(155_001_000)])
    assert len(watcher.snapshot().channels) == 1


# -- choosing where to listen ----------------------------------------------


def test_nothing_is_auditioned_before_it_has_been_heard_twice():
    watcher, _ = build()
    watcher.note_pass([signal(155e6)])
    assert watcher.choose_target() is None
    watcher.note_pass([signal(155e6)])
    assert watcher.choose_target() == pytest.approx(155e6)


def test_a_channel_that_has_gone_quiet_is_not_auditioned():
    watcher, _ = build()
    watcher.note_pass([signal(155e6)])
    watcher.note_pass([signal(155e6)])
    watcher.note_pass([])
    assert watcher.choose_target() is None


def test_the_strongest_never_heard_channel_goes_first():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6, snr_db=20.0), signal(156e6, snr_db=35.0)])
    assert watcher.choose_target() == pytest.approx(156e6)


def test_the_radio_does_not_lock_onto_the_strongest_channel():
    """Without a revisit delay a scanner visits one frequency for ever.

    The failure is quiet: everything on the list keeps updating from the
    sweep, so the screen looks entirely correct while the loudest channel is
    the only one anything is ever actually heard on.
    """
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6, snr_db=20.0), signal(156e6, snr_db=35.0)])
    first = watcher.choose_target()
    watcher.begin_audition(first)
    watcher.note_audition(first, verdict(voice.NOISE))
    clock.tick(0.9)
    second = watcher.choose_target()
    assert second != first
    assert second == pytest.approx(155e6)


def test_a_channel_is_left_alone_until_it_is_due_again():
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.note_audition(155e6, verdict(voice.NOISE))
    clock.tick(1.0)
    assert watcher.choose_target() is None
    clock.tick(mon.DEFAULT_REVISIT_S)
    assert watcher.choose_target() == pytest.approx(155e6)


def test_a_channel_that_carried_a_voice_is_revisited_sooner():
    """A conversation is short transmissions with gaps between them.

    Waiting the full revisit delay after each over means arriving back after
    the reply has finished, every time.
    """
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.note_audition(155e6, verdict(voice.VOICE))
    watcher.note_hold(155e6, verdict(voice.NOISE))
    clock.tick(mon.DEFAULT_RELEASE_S + 0.1)
    watcher.note_hold(155e6, verdict(voice.NOISE))
    clock.tick(mon.VOICE_REVISIT_S + 0.1)
    assert watcher.choose_target() == pytest.approx(155e6)


def test_nothing_is_auditioned_when_listening_is_off():
    watcher, _ = build(listen=False)
    for _ in range(3):
        watcher.note_pass([signal(155e6)])
    assert watcher.choose_target() is None
    assert watcher.phase == mon.SWEEPING


# -- stopping, holding and letting go --------------------------------------


def test_a_voice_stops_the_sweep_and_static_does_not():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    assert watcher.phase == mon.AUDITIONING
    assert watcher.note_audition(155e6, verdict(voice.VOICE))
    assert watcher.phase == mon.HOLDING

    watcher.resume()
    watcher.begin_audition(155e6)
    assert not watcher.note_audition(155e6, verdict(voice.NOISE))
    assert watcher.phase == mon.SWEEPING


@pytest.mark.parametrize(
    ("kind", "stops"),
    [
        (voice.VOICE, True),
        (voice.MUSIC, True),
        (voice.DATA, False),
        (voice.TONE, False),
        (voice.NOISE, False),
        (voice.SILENCE, False),
    ],
)
def test_only_something_worth_hearing_stops_the_sweep(kind, stops):
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    assert watcher.note_audition(155e6, verdict(kind)) is stops


def test_a_hold_rides_out_the_gap_between_two_overs():
    """Releasing on the first quiet window sends the radio away mid-exchange."""
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    watcher.note_audition(155e6, verdict(voice.VOICE))

    clock.tick(0.8)
    assert watcher.note_hold(155e6, verdict(voice.NOISE))
    clock.tick(0.8)
    assert watcher.note_hold(155e6, verdict(voice.NOISE))
    # ...and the other party answers.
    clock.tick(0.8)
    assert watcher.note_hold(155e6, verdict(voice.VOICE))
    assert watcher.phase == mon.HOLDING


def test_a_hold_ends_once_the_channel_has_been_quiet_long_enough():
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    watcher.note_audition(155e6, verdict(voice.VOICE))
    clock.tick(mon.DEFAULT_RELEASE_S + 0.1)
    assert not watcher.note_hold(155e6, verdict(voice.NOISE))
    assert watcher.phase == mon.SWEEPING
    assert watcher.target_hz is None


def test_a_user_hold_never_releases_on_its_own():
    """The one thing that overrides the state machine, and it has to.

    Somebody can hear that a channel is worth staying on when the classifier
    cannot - a weak signal, a language it has no opinion about, a pause while
    somebody thinks.
    """
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    watcher.note_audition(155e6, verdict(voice.VOICE))
    watcher.hold(155e6)
    for _ in range(20):
        clock.tick(mon.DEFAULT_RELEASE_S)
        assert watcher.note_hold(155e6, verdict(voice.NOISE))
    assert watcher.phase == mon.HOLDING


def test_holding_a_second_channel_releases_the_first():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6), signal(156e6)])
    watcher.hold(155e6)
    watcher.hold(156e6)
    assert watcher.held_hz == pytest.approx(156e6)
    held = [c.frequency_hz for c in watcher.snapshot().channels if c.held]
    assert held == [pytest.approx(156e6)]


def test_skipping_leaves_the_channel_now_and_stops_going_back():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    watcher.note_audition(155e6, verdict(voice.VOICE))
    watcher.skip(155e6)
    assert watcher.phase == mon.SWEEPING
    assert watcher.choose_target() is None
    (channel,) = watcher.snapshot().channels
    assert channel.skipped
    # ...and it is still on the list, with everything known about it.
    assert channel.sightings == 2
    assert channel.heard_voice


def test_a_skip_can_be_undone():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.skip(155e6)
    watcher.unskip(155e6)
    assert watcher.choose_target() == pytest.approx(155e6)


def test_what_was_heard_is_remembered_per_channel():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6), signal(156e6)])
    watcher.note_audition(155e6, verdict(voice.VOICE))
    watcher.resume()
    watcher.note_audition(156e6, verdict(voice.DATA))
    lower, upper = watcher.snapshot().channels
    assert lower.heard_voice
    assert lower.sound == "Voice"
    assert not upper.heard_voice
    assert upper.sound == "Data"


def test_static_on_a_channel_is_not_counted_as_hearing_it():
    """The sweep found something; listening found nothing there.

    Refreshing `last_heard` on a static verdict would have a channel that has
    stopped transmitting reporting itself as heard a moment ago for ever.
    """
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    heard = watcher.snapshot().channels[0].last_heard
    clock.tick(5.0)
    watcher.note_audition(155e6, verdict(voice.NOISE))
    assert watcher.snapshot().channels[0].last_heard == heard


# -- what the screen is told ------------------------------------------------


def test_the_state_says_what_the_radio_is_doing():
    watcher, _ = build()
    assert watcher.snapshot().phase == mon.SWEEPING
    assert not watcher.snapshot().listening
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    watcher.begin_audition(155e6)
    watcher.note_audition(155e6, verdict(voice.VOICE))
    state = watcher.snapshot()
    assert state.holding and state.listening
    assert state.target_hz == pytest.approx(155e6)


def test_the_state_counts_what_is_busy_and_what_has_talked():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6), signal(156e6)])
    watcher.note_audition(155e6, verdict(voice.VOICE))
    state = watcher.snapshot()
    assert state.busy == 2
    assert state.voices == 1


def test_channels_come_back_in_frequency_order():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(156e6), signal(151e6), signal(155e6)])
    order = [c.frequency_hz for c in watcher.snapshot().channels]
    assert order == sorted(order)


# -- ordering the list ------------------------------------------------------


def test_busiest_first_puts_the_busiest_first():
    watcher, _ = build()
    for index in range(10):
        found = [signal(155e6)]
        if index % 3 == 0:
            found.append(signal(156e6))
        watcher.note_pass(found)
    ordered = mon.sort_activities(watcher.snapshot().channels, "activity")
    assert ordered[0].frequency_hz == pytest.approx(155e6)


def test_recently_heard_puts_the_newest_first():
    watcher, clock = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6), signal(156e6)])
    clock.tick(10.0)
    for _ in range(2):
        watcher.note_pass([signal(156e6)])
    ordered = mon.sort_activities(watcher.snapshot().channels, "recent")
    assert ordered[0].frequency_hz == pytest.approx(156e6)


def test_the_sweep_orders_still_work_on_a_monitor_list():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6, snr_db=15.0), signal(156e6, snr_db=35.0)])
    channels = watcher.snapshot().channels
    assert mon.sort_activities(channels, "strength")[0].snr_db == pytest.approx(35.0)
    assert mon.sort_activities(channels, "frequency")[0].frequency_hz == pytest.approx(
        155e6
    )


def test_every_order_is_fully_determined():
    """Two equal entries must not swap between updates.

    A card carries an expanded explanation, and a list that reorders itself
    under the cursor slams it shut - which is exactly the failure the sweep's
    orders were made total to avoid.
    """
    watcher, _ = build()
    for _ in range(3):
        watcher.note_pass([signal(hz) for hz in (151e6, 155e6, 156e6)])
    channels = watcher.snapshot().channels
    for order in ("activity", "recent", "strength", "frequency", "kind"):
        first = [c.frequency_hz for c in mon.sort_activities(channels, order)]
        second = [c.frequency_hz for c in mon.sort_activities(channels[::-1], order)]
        assert first == second, order


def test_the_voice_filter_keeps_only_what_has_talked():
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6), signal(156e6)])
    watcher.note_audition(155e6, verdict(voice.VOICE))
    kept = mon.with_voice(watcher.snapshot().channels)
    assert [c.frequency_hz for c in kept] == [pytest.approx(155e6)]


def test_the_signal_behind_a_channel_is_available_to_demodulate_it():
    """The audition uses the mode the classifier chose, never one of its own."""
    watcher, _ = build()
    for _ in range(2):
        watcher.note_pass([signal(155e6)])
    found = watcher.signal_at(155e6)
    assert found is not None
    assert found.mode
    assert found.demod_bandwidth_hz > 0
    assert watcher.signal_at(120e6) is None


def test_an_activity_describes_itself_in_plain_english():
    watcher, _ = build()
    for _ in range(4):
        watcher.note_pass([signal(155e6)])
    (channel,) = watcher.snapshot().channels
    assert "4 times" in channel.activity_phrase
    assert "100%" in channel.activity_phrase


def test_a_ledger_with_nothing_in_it_is_not_an_error():
    watcher, _ = build()
    watcher.note_pass([])
    state = watcher.snapshot()
    assert state.channels == ()
    assert state.busy == 0
    assert np.isfinite(state.elapsed_s)
