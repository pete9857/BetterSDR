"""Sound card output with a jitter buffer.

The DSP thread produces audio in bursts whose timing follows USB transfers,
while the sound card consumes it on a rigid clock. A buffer between them
absorbs that mismatch. Two policies matter more than they look:

* An underrun emits **silence**, never stale or repeated audio. A brief gap
  reads as "the signal dropped"; a repeated fragment reads as "this program is
  broken", and we would rather a beginner blame the airwaves than the app.
* When the producer runs ahead - which happens after a stall - we discard the
  oldest audio rather than letting the backlog grow. Latency that creeps
  upward and never recovers is worse than one audible glitch.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import sounddevice as sd

DEFAULT_RATE = 48_000
# Enough slack to ride out USB scheduling and a Windows timer hiccup, while
# staying well under the point where tuning feels laggy.
DEFAULT_TARGET_LATENCY_S = 0.15
DEFAULT_MAX_LATENCY_S = 0.4


class ClockSync:
    """Stretch or squeeze audio slightly so it tracks the sound card's clock.

    The dongle and the sound card are timed by different crystals, and no
    amount of buffering fixes a rate mismatch - it only delays the moment the
    buffer runs dry. Left alone, a listener hears a tick every few seconds
    after the first couple of minutes.

    So the buffer depth becomes a control signal: when it sits below target we
    resample each block very slightly longer, and when it climbs we shorten it.
    The correction is capped at 0.5%, which is under a tenth of a semitone and
    inaudible, but is around twenty times larger than any real crystal error.
    """

    def __init__(
        self,
        target_samples: int,
        max_correction: float = 0.005,
        gain: float = 0.05,
    ) -> None:
        self.target_samples = max(1, int(target_samples))
        self.max_correction = float(max_correction)
        self.gain = float(gain)
        self.ratio = 1.0

    def resample(self, audio: np.ndarray, buffered: int) -> np.ndarray:
        frames = audio.shape[0]
        if frames < 2:
            return audio
        error = (self.target_samples - buffered) / self.target_samples
        self.ratio = 1.0 + float(
            np.clip(self.gain * error, -self.max_correction, self.max_correction)
        )
        wanted = max(2, int(round(frames * self.ratio)))
        if wanted == frames:
            return audio
        # Both endpoints are preserved, so blocks still join without a step.
        source = np.arange(frames, dtype=np.float32)
        target = np.linspace(0.0, frames - 1, wanted, dtype=np.float32)
        if audio.ndim == 1:
            return np.interp(target, source, audio).astype(np.float32)
        # Every channel is stretched by the same ratio onto the same grid, so
        # the two ears stay sample-aligned. Resampling them independently -
        # even by rounding to a different length - is an image that wanders.
        out = np.empty((wanted, audio.shape[1]), dtype=np.float32)
        for channel in range(audio.shape[1]):
            out[:, channel] = np.interp(target, source, audio[:, channel])
        return out


def output_devices() -> list[tuple[int, str]]:
    """Indexes and names of every device that can play audio."""
    return [
        (index, device["name"])
        for index, device in enumerate(sd.query_devices())
        if device["max_output_channels"] > 0
    ]


def default_output_device() -> str:
    try:
        return str(sd.query_devices(kind="output")["name"])
    except Exception:  # noqa: BLE001 - no audio device is not fatal here
        return "none"


class AudioSink:
    """A float32 output stream fed from another thread.

    Opened with two channels whatever the radio is doing, and a mono block is
    duplicated into both on the way in. That costs one copy of a 48 kHz stream
    and buys the thing that matters: FM stereo comes and goes with the pilot,
    several times a minute on a marginal station, and reopening the sound card
    at each transition would put a gap in the audio every time.
    """

    def __init__(
        self,
        rate: int = DEFAULT_RATE,
        device: int | str | None = None,
        target_latency_s: float = DEFAULT_TARGET_LATENCY_S,
        max_latency_s: float = DEFAULT_MAX_LATENCY_S,
        drift_correction: bool = True,
        channels: int = 2,
    ) -> None:
        self.rate = int(rate)
        self.device = device
        self.channels = max(1, int(channels))
        self.target_samples = int(self.rate * target_latency_s)
        self.max_samples = int(self.rate * max_latency_s)
        self.underruns = 0
        self.dropped_blocks = 0
        self.clock = ClockSync(self.target_samples) if drift_correction else None

        self._blocks: deque[np.ndarray] = deque()
        self._buffered = 0
        self._offset = 0
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> AudioSink:
        if self._stream is not None:
            return self
        # Priming with silence means the first real audio arrives into a
        # buffer that already has depth, instead of underrunning immediately.
        self.write(np.zeros(self.target_samples, dtype=np.float32))
        self._stream = self._open()
        self._stream.start()
        return self

    def _open(self) -> sd.OutputStream:
        """Open the stream, dropping to mono if the device refuses two.

        Essentially every sound card does two channels, but a virtual or
        telephony device may not, and being unable to play anything at all is
        a far worse outcome than being unable to play it in stereo.
        """
        try:
            return self._stream_with(self.channels)
        except Exception:  # noqa: BLE001 - fall back rather than go silent
            if self.channels == 1:
                raise
        self.channels = 1
        self.flush()
        self.write(np.zeros(self.target_samples, dtype=np.float32))
        return self._stream_with(1)

    def _stream_with(self, channels: int) -> sd.OutputStream:
        return sd.OutputStream(
            samplerate=self.rate,
            channels=channels,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Outside the branch on purpose: stopping an already-stopped sink must
        # still leave it empty, so a caller cannot end up with a primed buffer
        # it believes it discarded.
        self.flush()

    def flush(self) -> None:
        """Throw away anything still queued.

        Called on every stop, and that is the whole point of it. The sink is
        parked for a gain probe, a sample-rate change or a sweep, and in every
        one of those cases the audio still sitting in the buffer was captured
        under conditions that no longer apply. Worse, `start` primes with a
        fresh target buffer of silence, so keeping the old contents *adds*
        150 ms of latency to every park and never gives it back.

        Measured on air: one gain probe took the buffer from 190 ms to 369 ms,
        where it stayed - and after a few more it sat against the 400 ms cap
        discarding blocks to stay there, which is a third of a second of audio
        lagging behind the display for the rest of the session.
        """
        with self._lock:
            self._blocks.clear()
            self._buffered = 0
            self._offset = 0

    def __enter__(self) -> AudioSink:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- producer side -----------------------------------------------------

    @property
    def latency_s(self) -> float:
        return self._buffered / self.rate

    def _conform(self, audio: np.ndarray) -> np.ndarray:
        """Reshape a block to the channel count the stream was opened with.

        Mono is duplicated rather than placed on the left, and stereo is
        averaged rather than truncated: either shortcut would play half the
        broadcast on a device that cannot do better.
        """
        block = np.ascontiguousarray(audio, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] == self.channels:
            return block
        if block.shape[1] == 1:
            return np.ascontiguousarray(np.repeat(block, self.channels, axis=1))
        return np.ascontiguousarray(
            np.repeat(block.mean(axis=1, keepdims=True), self.channels, axis=1)
        )

    def write(self, audio: np.ndarray) -> None:
        """Queue audio for playback. Safe to call from the DSP thread.

        Takes mono `(frames,)` or `(frames, channels)`; either is conformed to
        what the stream was opened with.
        """
        if audio.size == 0:
            return
        block = self._conform(audio)
        if self.clock is not None:
            block = self.clock.resample(block, self._buffered)
        with self._lock:
            self._blocks.append(block)
            self._buffered += block.shape[0]
            while self._buffered > self.max_samples and len(self._blocks) > 1:
                oldest = self._blocks.popleft()
                self._buffered -= oldest.shape[0] - self._offset
                self._offset = 0
                self.dropped_blocks += 1

    # -- consumer side -----------------------------------------------------

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        filled = 0
        with self._lock:
            while filled < frames and self._blocks:
                block = self._blocks[0]
                take = min(frames - filled, block.shape[0] - self._offset)
                outdata[filled : filled + take] = block[
                    self._offset : self._offset + take
                ]
                filled += take
                self._offset += take
                if self._offset >= block.shape[0]:
                    self._blocks.popleft()
                    self._offset = 0
            self._buffered -= filled
        if filled < frames:
            outdata[filled:] = 0.0
            self.underruns += 1


__all__ = ["AudioSink", "default_output_device", "output_devices"]
