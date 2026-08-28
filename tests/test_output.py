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
    """Run one PortAudio callback and return what it produced."""
    out = np.zeros((frames, 1), dtype=np.float32)
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
