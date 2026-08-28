"""Tests for audio and IQ recording.

The claim that matters most is that an IQ file is byte-exact: it is the only
artefact the app produces that another program might measure, and a recorder
that quietly rescales or resamples would make every such measurement wrong
without ever looking wrong.
"""

from __future__ import annotations

import wave
from datetime import UTC, datetime

import numpy as np
import pytest

from bettersdr.audio.record import (
    AudioRecorder,
    IqRecorder,
    RecordingLimits,
    bytes_per_second,
    timestamped_name,
)


def _read_wav(path):
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.readframes(handle.getnframes()),
        )


def test_iq_recording_is_byte_exact(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, 65_536, dtype=np.uint8)
    path = tmp_path / "capture.wav"

    with IqRecorder(path, sample_rate=2_400_000) as recorder:
        for start in range(0, raw.size, 8_192):
            recorder.write(raw[start : start + 8_192])

    channels, width, rate, payload = _read_wav(path)
    assert (channels, width, rate) == (2, 1, 2_400_000)
    np.testing.assert_array_equal(np.frombuffer(payload, dtype=np.uint8), raw)


def test_audio_recording_round_trips_at_full_scale(tmp_path):
    t = np.arange(48_000) / 48_000
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "audio.wav"

    with AudioRecorder(path) as recorder:
        for start in range(0, audio.size, 1_024):
            recorder.write(audio[start : start + 1_024])

    channels, width, rate, payload = _read_wav(path)
    assert (channels, width, rate) == (1, 2, 48_000)
    restored = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0
    np.testing.assert_allclose(restored, audio, atol=1e-4)


def test_audio_recording_clips_rather_than_wrapping(tmp_path):
    """An integer overflow in a WAV is a square wave, not a loud recording."""
    path = tmp_path / "loud.wav"
    with AudioRecorder(path) as recorder:
        recorder.write(np.full(1_024, 4.0, dtype=np.float32))
        recorder.write(np.full(1_024, -4.0, dtype=np.float32))

    _, _, _, payload = _read_wav(path)
    samples = np.frombuffer(payload, dtype="<i2")
    assert samples[:1_024].min() == 32_767
    assert samples[1_024:].max() == -32_767


def test_recording_stops_at_its_size_limit(tmp_path):
    path = tmp_path / "capped.wav"
    recorder = IqRecorder(
        path, 2_400_000, RecordingLimits(max_bytes=16_384, min_free_bytes=0)
    ).start()
    for _ in range(10):
        recorder.write(np.zeros(8_192, dtype=np.uint8))

    assert not recorder.active
    assert recorder.bytes_written == 16_384
    assert "size limit" in (recorder.stopped_reason or "")


def test_recording_stops_at_its_time_limit(tmp_path):
    path = tmp_path / "timed.wav"
    recorder = AudioRecorder(
        path, 48_000, RecordingLimits(max_seconds=0.5, max_bytes=None, min_free_bytes=0)
    ).start()
    for _ in range(100):
        recorder.write(np.zeros(4_800, dtype=np.float32))

    assert not recorder.active
    assert recorder.seconds == pytest.approx(0.5, abs=0.11)
    assert "time limit" in (recorder.stopped_reason or "")


def test_refuses_to_start_without_free_space(tmp_path):
    recorder = IqRecorder(
        tmp_path / "no.wav",
        2_400_000,
        RecordingLimits(min_free_bytes=1 << 62),
    ).start()
    assert not recorder.active
    assert "free space" in (recorder.stopped_reason or "")
    # Writing to a recorder that never opened must be harmless, not a crash:
    # the caller is a DSP thread that has no way to report an exception.
    recorder.write(np.zeros(1_024, dtype=np.uint8))
    assert recorder.bytes_written == 0


def test_seconds_tracks_the_two_formats_correctly(tmp_path):
    iq = IqRecorder(tmp_path / "a.wav", 2_400_000).start()
    iq.write(np.zeros(2_400_000 * 2, dtype=np.uint8))
    assert iq.seconds == pytest.approx(1.0)
    iq.stop()

    audio = AudioRecorder(tmp_path / "b.wav", 48_000).start()
    audio.write(np.zeros(48_000, dtype=np.float32))
    assert audio.seconds == pytest.approx(1.0)
    audio.stop()


def test_filename_carries_the_frequency_and_a_utc_stamp():
    when = datetime(2026, 8, 28, 14, 30, 0, tzinfo=UTC)
    assert (
        timestamped_name(98_500_000, "IQ", when=when)
        == "BetterSDR_20260828_143000Z_98500000Hz_IQ.wav"
    )


def test_iq_costs_fifty_times_what_audio_does():
    assert bytes_per_second(2_400_000) == 4_800_000
    assert bytes_per_second(48_000) == 96_000
