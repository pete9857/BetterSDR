"""Tests for the audio jitter buffer.

These drive `AudioSink` without opening a sound card: the producer side and the
PortAudio callback are both plain methods, and the policies worth pinning down
- silence on underrun, drop the oldest when the backlog grows - live entirely
in them.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.audio.output import AudioSink, ClockSync


def sink(**kwargs: object) -> AudioSink:
    kwargs.setdefault("drift_correction", False)
    return AudioSink(rate=48_000, **kwargs)


def pull(audio_sink: AudioSink, frames: int) -> np.ndarray:
    """Run one PortAudio callback and return the left channel of what it made.

    The sink is opened with two channels whatever the radio is doing, and a
    mono block is duplicated into both, so reading one channel back is the
    same audio - see `test_a_mono_block_reaches_both_ears`, which is the test
    that pins that down rather than assuming it.
    """
    out = np.zeros((frames, audio_sink.channels), dtype=np.float32)
    audio_sink._callback(out, frames, None, None)
    return out[:, 0]


def test_audio_comes_back_in_order():
    s = sink()
    s.write(np.arange(100, dtype=np.float32))
    s.write(np.arange(100, 200, dtype=np.float32))
    np.testing.assert_array_equal(pull(s, 200), np.arange(200, dtype=np.float32))


def test_callback_spanning_several_written_blocks():
    s = sink()
    for i in range(4):
        s.write(np.full(50, float(i), dtype=np.float32))
    out = pull(s, 200)
    assert out[0] == 0.0 and out[60] == 1.0 and out[199] == 3.0


def test_partial_block_is_resumed_next_callback():
    s = sink()
    s.write(np.arange(100, dtype=np.float32))
    np.testing.assert_array_equal(pull(s, 30), np.arange(30, dtype=np.float32))
    np.testing.assert_array_equal(pull(s, 70), np.arange(30, 100, dtype=np.float32))


def test_underrun_emits_silence_not_stale_audio():
    """A gap must sound like a gap, never like a repeated fragment."""
    s = sink()
    s.write(np.ones(10, dtype=np.float32))
    out = pull(s, 50)

    assert np.all(out[:10] == 1.0)
    assert np.all(out[10:] == 0.0)
    assert s.underruns == 1


def test_underrun_on_empty_buffer_is_fully_silent():
    s = sink()
    assert np.all(pull(s, 128) == 0.0)
    assert s.underruns == 1


def test_latency_tracks_what_is_queued():
    s = sink()
    s.write(np.zeros(4_800, dtype=np.float32))
    assert s.latency_s == 0.1
    pull(s, 2_400)
    assert s.latency_s == 0.05


def test_backlog_is_trimmed_from_the_oldest_end():
    """Latency that creeps up and never recovers is worse than one glitch."""
    s = sink(max_latency_s=0.1)  # 4800 samples
    for i in range(10):
        s.write(np.full(1_000, float(i), dtype=np.float32))

    assert s.latency_s <= 0.1
    assert s.dropped_blocks > 0
    # What survives is the newest audio, so playback stays current: draining
    # the buffer ends on the most recently written block.
    drained = pull(s, 4_000)
    assert drained[-1] == 9.0
    assert drained[0] > 0.0  # the earliest blocks are the ones discarded


def test_empty_write_is_ignored():
    s = sink()
    s.write(np.zeros(0, dtype=np.float32))
    assert s.latency_s == 0.0


def test_write_accepts_any_float_input():
    s = sink()
    s.write(np.array([0.25, -0.5], dtype=np.float64))
    np.testing.assert_allclose(pull(s, 2), [0.25, -0.5])


# -- Clock drift -----------------------------------------------------------


def test_clock_sync_stretches_when_the_buffer_is_low():
    clock = ClockSync(target_samples=1_000)
    out = clock.resample(np.zeros(1_000, dtype=np.float32), buffered=500)
    assert clock.ratio > 1.0
    assert out.size > 1_000


def test_clock_sync_shortens_when_the_buffer_is_high():
    clock = ClockSync(target_samples=1_000)
    out = clock.resample(np.zeros(1_000, dtype=np.float32), buffered=2_000)
    assert clock.ratio < 1.0
    assert out.size < 1_000


def test_clock_sync_correction_stays_inaudible():
    """0.5% is under a tenth of a semitone; more would be heard as pitch."""
    clock = ClockSync(target_samples=1_000, max_correction=0.005)
    clock.resample(np.zeros(1_000, dtype=np.float32), buffered=0)
    assert clock.ratio == pytest.approx(1.005)
    clock.resample(np.zeros(1_000, dtype=np.float32), buffered=1_000_000)
    assert clock.ratio == pytest.approx(0.995)


def test_clock_sync_preserves_the_waveform_endpoints():
    clock = ClockSync(target_samples=1_000)
    audio = np.sin(np.linspace(0, 4 * np.pi, 480, dtype=np.float32))
    out = clock.resample(audio, buffered=100)
    assert out[0] == pytest.approx(audio[0], abs=1e-5)
    assert out[-1] == pytest.approx(audio[-1], abs=1e-5)


def test_drift_correction_prevents_starvation_when_the_radio_runs_slow():
    """The measured failure: capture at 99.8% drains the buffer within minutes."""
    s = AudioSink(rate=48_000, drift_correction=True)
    s.write(np.zeros(s.target_samples, dtype=np.float32))
    for _ in range(5_000):  # ~100 s of audio, well past the drain point below
        s.write(np.zeros(958, dtype=np.float32))  # 0.2% short, as measured
        pull(s, 960)
    assert s.underruns == 0


def test_without_drift_correction_the_same_stream_starves():
    """The buffer holds 7200 samples and loses 2 per cycle, so it dies at ~3600."""
    s = AudioSink(rate=48_000, drift_correction=False)
    s.write(np.zeros(s.target_samples, dtype=np.float32))
    for _ in range(5_000):
        s.write(np.zeros(958, dtype=np.float32))
        pull(s, 960)
    assert s.underruns > 0


def test_flush_drops_the_backlog():
    s = sink()
    s.write(np.ones(5_000, dtype=np.float32))
    assert s.latency_s > 0
    s.flush()
    assert s.latency_s == 0
    np.testing.assert_array_equal(pull(s, 64), np.zeros(64, dtype=np.float32))


def test_flush_clears_a_partly_consumed_block():
    """The read offset has to go with it, or the next block starts mid-way."""
    s = sink()
    s.write(np.arange(100, dtype=np.float32))
    pull(s, 40)
    s.flush()
    s.write(np.full(10, 7.0, dtype=np.float32))
    np.testing.assert_array_equal(pull(s, 10), np.full(10, 7.0, dtype=np.float32))


def test_parking_the_sink_does_not_add_latency_every_time():
    """Measured on air before this was fixed: one gain probe took the buffer
    from 190 ms to 369 ms and it never came back, because `stop` kept the
    queued audio and `start` primed a fresh target buffer on top of it. A few
    probes later it sat against the 400 ms cap discarding blocks - a third of
    a second of audio lagging behind the display for the rest of the session.

    `start` opens a real stream, so this exercises the two halves that matter:
    the queue is emptied on stop, and priming refills it to the target once.
    """
    s = sink()
    s.write(np.ones(s.target_samples, dtype=np.float32))
    settled = s.latency_s

    for _ in range(5):
        s.stop()  # no stream open, but the flush must still happen
        assert s.latency_s == 0
        s.write(np.ones(s.target_samples, dtype=np.float32))

    assert s.latency_s == pytest.approx(settled)
    assert s.dropped_blocks == 0


# -- stereo ----------------------------------------------------------------


def test_a_mono_block_reaches_both_ears():
    """Silence in one ear is the failure a stereo-capable sink invites."""
    s = sink()
    s.write(np.arange(64, dtype=np.float32))
    out = np.zeros((64, 2), dtype=np.float32)
    s._callback(out, 64, None, None)
    np.testing.assert_array_equal(out[:, 0], np.arange(64, dtype=np.float32))
    np.testing.assert_array_equal(out[:, 1], out[:, 0])


def test_a_stereo_block_keeps_its_channels_apart():
    s = sink()
    block = np.stack(
        [np.arange(64, dtype=np.float32), -np.arange(64, dtype=np.float32)], axis=1
    )
    s.write(block)
    out = np.zeros((64, 2), dtype=np.float32)
    s._callback(out, 64, None, None)
    np.testing.assert_array_equal(out, block)


def test_a_stereo_block_is_mixed_down_for_a_mono_device():
    """A device that will only do one channel must still hear both of them."""
    s = sink(channels=1)
    block = np.stack(
        [np.ones(32, dtype=np.float32), np.full(32, 3.0, dtype=np.float32)], axis=1
    )
    s.write(block)
    np.testing.assert_allclose(pull(s, 32), np.full(32, 2.0), atol=1e-6)


def test_latency_counts_frames_not_samples():
    """A stereo block is not half a second of audio because it has two
    channels in it - which is exactly what a `size`-based count would say."""
    s = sink()
    s.write(np.zeros((4_800, 2), dtype=np.float32))
    assert s.latency_s == pytest.approx(0.1)


def test_drift_correction_stretches_both_channels_the_same_way():
    """Resampling the channels independently is an image that wanders."""
    clock = ClockSync(target_samples=1_000)
    ramp = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    block = np.stack([ramp, ramp], axis=1)
    out = clock.resample(block, buffered=100)
    assert out.shape[0] > 512
    np.testing.assert_array_equal(out[:, 0], out[:, 1])
