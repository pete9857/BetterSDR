"""Recording: demodulated audio to WAV, and raw baseband IQ to WAV.

Two recorders because they capture two genuinely different things, and the
difference is worth being explicit about:

* **Audio** is what you heard. It has been through the demodulator, the
  squelch, noise reduction and the AGC, and - importantly - through the clock
  drift correction in `audio/output.py`, which resamples by up to 0.5% to keep
  the sound card fed. That is inaudible but it is not bit-exact, so an audio
  recording is a record of the listening session, not a measurement.
* **IQ** is what the antenna received. It is taken from the ring buffer ahead
  of everything, so it can be replayed later through a different demodulator,
  a decoder that does not exist yet, or somebody else's software. This is the
  format that matters for anything scientific.

The IQ file is a plain WAV: two channels of unsigned 8-bit PCM at the SDR's
own sample rate, which is exactly what the RTL2832U delivers and exactly what
SDR#, HDSDR and GNU Radio expect to read back. No conversion, no loss - the
bytes in the file are the bytes off the USB.

Both recorders guard against the two ways an unattended recording ruins
somebody's day: they stop at a size or duration limit, and they stop before
the disk fills. **Raw IQ is 4.8 MB per second at 2.4 MS/s** - 17 GB an hour -
so this is not a theoretical concern.

That guard is `_Recorder`, and it is a base class rather than part of the WAV
writer because `audio/encode.py` needs exactly the same one. A recorder that
runs unattended for hours - which is the whole of Repro-Radio - is the case
the limits were written for, and a second copy of them would be a second copy
to get wrong.
"""

from __future__ import annotations

import shutil
import time
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# One second of IQ at full rate is 4.8 MB; the default cap is about ten
# minutes of it, or many hours of audio.
DEFAULT_MAX_BYTES = 3 * 1024**3
DEFAULT_MIN_FREE_BYTES = 512 * 1024**2
# Checking free space on every block would be a filesystem call at the block
# rate. Every few seconds is soon enough to stop before a disk fills.
FREE_SPACE_CHECK_S = 5.0


@dataclass(frozen=True)
class RecordingLimits:
    """When to stop on our own, before something else stops us badly."""

    max_seconds: float | None = None
    max_bytes: int | None = DEFAULT_MAX_BYTES
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES


def timestamped_name(
    center_hz: float, kind: str, extension: str = "wav", when: datetime | None = None
) -> str:
    """`BetterSDR_20260828_143000Z_98500000Hz_AF.wav`.

    The frequency is in the filename because a folder of recordings with only
    timestamps is useless a week later, and because this is the convention
    SDR# established - files sort together and other tools can parse them.
    """
    moment = when or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%d_%H%M%SZ")
    return f"BetterSDR_{stamp}_{int(round(center_hz))}Hz_{kind}.{extension}"


class _Recorder:
    """Limits, the disk-space guard, and the numbers a status line needs.

    Deliberately knows nothing about a file format. Duration is counted in
    **frames**, never derived from the byte count, because a compressed
    recording's bytes say nothing about how long it is - the one assumption a
    WAV writer is allowed to make and an MP3 writer is not.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        limits: RecordingLimits | None = None,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.limits = limits or RecordingLimits()
        self.bytes_written = 0
        self.frames_written = 0
        self.stopped_reason: str | None = None
        self._started = 0.0
        self._last_space_check = 0.0

    # -- lifecycle ---------------------------------------------------------

    def _prepare(self) -> bool:
        """Make the folder and check there is room. False means do not open."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.path.parent).free < self.limits.min_free_bytes:
            self.stopped_reason = "There is not enough free space to record."
            return False
        self._started = time.monotonic()
        self._last_space_check = self._started
        return True

    def _close(self) -> None:
        """Release whatever the subclass opened. Must be safe to call twice."""

    def stop(self, reason: str | None = None) -> None:
        if self.active:
            self._close()
        if reason is not None and self.stopped_reason is None:
            self.stopped_reason = reason

    def start(self) -> _Recorder:  # pragma: no cover - subclasses provide it
        raise NotImplementedError

    def __enter__(self) -> _Recorder:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- state -------------------------------------------------------------

    @property
    def active(self) -> bool:  # pragma: no cover - subclasses provide it
        raise NotImplementedError

    @property
    def seconds(self) -> float:
        return self.frames_written / self.sample_rate if self.sample_rate else 0.0

    def _over_limit(self) -> str | None:
        limits = self.limits
        if limits.max_bytes is not None and self.bytes_written >= limits.max_bytes:
            return "The recording reached its size limit and was stopped."
        if limits.max_seconds is not None and self.seconds >= limits.max_seconds:
            return "The recording reached its time limit and was stopped."
        now = time.monotonic()
        if now - self._last_space_check >= FREE_SPACE_CHECK_S:
            self._last_space_check = now
            if shutil.disk_usage(self.path.parent).free < limits.min_free_bytes:
                return "The disk is nearly full, so recording was stopped."
        return None

    def _account(self, written: int, frames: int) -> None:
        """Count what was just written, and stop if that crossed a limit."""
        self.bytes_written += int(written)
        self.frames_written += int(frames)
        reason = self._over_limit()
        if reason is not None:
            self.stop(reason)


class _WavRecorder(_Recorder):
    """Shared open/write/close for the two WAV formats."""

    channels = 1
    sample_width = 2

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        limits: RecordingLimits | None = None,
    ) -> None:
        super().__init__(path, sample_rate, limits)
        self._wave: wave.Wave_write | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> _WavRecorder:
        if not self._prepare():
            return self
        # Held open for the life of the recording rather than reopened per
        # block, so it is closed by `stop` instead of by a `with`.
        handle = wave.open(str(self.path), "wb")  # noqa: SIM115
        handle.setnchannels(self.channels)
        handle.setsampwidth(self.sample_width)
        handle.setframerate(self.sample_rate)
        self._wave = handle
        return self

    def _close(self) -> None:
        if self._wave is not None:
            self._wave.close()
            self._wave = None

    @property
    def active(self) -> bool:
        return self._wave is not None

    # -- writing -----------------------------------------------------------

    def _write_bytes(self, payload: bytes, frames: int) -> None:
        if self._wave is None:
            return
        self._wave.writeframes(payload)
        self._account(len(payload), frames)


class AudioRecorder(_WavRecorder):
    """16-bit WAV of the audio as it was heard, mono or stereo.

    The channel count is fixed when the file is opened, because a WAV header
    is written once. FM stereo, though, comes and goes with the pilot, so a
    recording started on a stereo station can find itself handed mono blocks
    partway through and the other way round. `write` conforms each block to
    what the header promised rather than letting the file go out of step with
    itself - a WAV whose channel count changes halfway is not a recoverable
    file, it is a burst of noise at double speed.
    """

    sample_width = 2

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = 48_000,
        limits: RecordingLimits | None = None,
        channels: int = 1,
    ) -> None:
        super().__init__(path, sample_rate, limits)
        self.channels = max(1, int(channels))

    def write(self, audio: np.ndarray) -> None:
        """Queue float32 audio in [-1, 1]. Anything louder is clipped, not
        wrapped: an integer overflow in a WAV is a full-scale square wave and
        sounds like the recording is destroyed rather than merely loud."""
        if self._wave is None or audio.size == 0:
            return
        block = np.asarray(audio, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] != self.channels:
            block = (
                np.repeat(block, self.channels, axis=1)
                if block.shape[1] == 1
                else block.mean(axis=1, keepdims=True)
            )
        clipped = np.clip(block, -1.0, 1.0)
        self._write_bytes((clipped * 32767.0).astype("<i2").tobytes(), block.shape[0])


class IqRecorder(_WavRecorder):
    """Two-channel unsigned 8-bit WAV: the bytes exactly as the dongle sent
    them.

    Written from the ring buffer rather than from anywhere downstream. The
    audio path resamples slightly to track the sound card's clock, which is
    the right thing for listening and the wrong thing for a capture that might
    later be measured or decoded.
    """

    channels = 2
    sample_width = 1

    def write(self, raw: np.ndarray) -> None:
        """Queue interleaved uint8 IQ, exactly as `Device.read` returns it."""
        if self._wave is None or raw.size == 0:
            return
        payload = np.ascontiguousarray(raw, dtype=np.uint8).tobytes()
        self._write_bytes(payload, len(payload) // 2)


def bytes_per_second(sample_rate: float, channels: int = 1) -> float:
    """What a recording costs on disk, for a warning the user can act on.

    Two bytes per sample either way, which is a coincidence worth spelling
    out: IQ is two 8-bit channels at the SDR rate, audio is one 16-bit channel
    at 48 kHz. The rate is what makes the difference - 4.8 MB/s for IQ at
    2.4 MS/s against 96 kB/s for mono audio, a factor of fifty. Stereo audio
    doubles its own figure and is still nowhere near.
    """
    return float(sample_rate) * 2.0 * max(1, int(channels))


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MIN_FREE_BYTES",
    "AudioRecorder",
    "IqRecorder",
    "RecordingLimits",
    "bytes_per_second",
    "timestamped_name",
]
