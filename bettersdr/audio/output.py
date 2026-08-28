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
        if audio.size < 2:
            return audio
        error = (self.target_samples - buffered) / self.target_samples
        self.ratio = 1.0 + float(
            np.clip(self.gain * error, -self.max_correction, self.max_correction)
        )
        wanted = max(2, int(round(audio.size * self.ratio)))
        if wanted == audio.size:
            return audio
        # Both endpoints are preserved, so blocks still join without a step.
        source = np.arange(audio.size, dtype=np.float32)
        target = np.linspace(0.0, audio.size - 1, wanted, dtype=np.float32)
        return np.interp(target, source, audio).astype(np.float32)


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
    """A mono float32 output stream fed from another thread."""

    def __init__(
        self,
        rate: int = DEFAULT_RATE,
        device: int | str | None = None,
        target_latency_s: float = DEFAULT_TARGET_LATENCY_S,
        max_latency_s: float = DEFAULT_MAX_LATENCY_S,
        drift_correction: bool = True,
    ) -> None:
        self.rate = int(rate)
        self.device = device
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
        self._stream = sd.OutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def __enter__(self) -> AudioSink:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- producer side -----------------------------------------------------

    @property
    def latency_s(self) -> float:
        return self._buffered / self.rate

    def write(self, audio: np.ndarray) -> None:
        """Queue audio for playback. Safe to call from the DSP thread."""
        if audio.size == 0:
            return
        block = np.ascontiguousarray(audio, dtype=np.float32)
        if self.clock is not None:
            block = self.clock.resample(block, self._buffered)
        with self._lock:
            self._blocks.append(block)
            self._buffered += block.size
            while self._buffered > self.max_samples and len(self._blocks) > 1:
                oldest = self._blocks.popleft()
                self._buffered -= oldest.size - self._offset
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
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._blocks:
                block = self._blocks[0]
                take = min(frames - filled, block.size - self._offset)
                out[filled : filled + take] = block[self._offset : self._offset + take]
                filled += take
                self._offset += take
                if self._offset >= block.size:
                    self._blocks.popleft()
                    self._offset = 0
            self._buffered -= filled
        if filled < frames:
            out[filled:] = 0.0
            self.underruns += 1


__all__ = ["AudioSink", "default_output_device", "output_devices"]
