"""MP3 encoding, and the ID3 tags that make a folder of files a collection.

Everything else the app writes is WAV, because WAV is what a measurement
wants: no codec, no decisions, byte-exact where it matters. Repro-Radio is the
first thing here that is not a measurement. It runs unattended for hours and
it fills a music folder, so the two things that matter are that the files are
small enough to leave running overnight and that every device the user owns
can play them without being told how.

That is MP3, and the choice is about the ecosystem rather than the codec.
Opus is better per bit and AAC is better per bit; neither shows a title in
Windows Explorer, and one of them will not play in a car. At 128 kbps stereo
the encoder is nowhere near the limiting factor anyway - an FM broadcast
arrives with 15 kHz of audio bandwidth and its own noise floor, both of which
are coarser than anything the codec is doing.

Three properties of MP3 are load-bearing here and are worth naming:

* **There is no header to fix at the end.** A file is a run of frames, so a
  recording interrupted by a crash, a power cut or a pulled dongle is still a
  playable file that is simply shorter than intended. That is why the
  in-progress name is a *rename* at the end rather than a rewrite, and it is
  the property the WAV recorders do not have.
* **The encoder is cheap.** Measured on this machine: **6.8 ms per second of
  48 kHz stereo audio** on noise, which is the worst case a lossy encoder can
  be handed - 0.7% of a core, against the WFM demodulator's 6.3%. So it runs
  inline on the DSP thread exactly like the WAV writers, with no queue and no
  thread of its own to get wrong.
* **The channel count is fixed when the encoder is built**, the same
  constraint as a WAV header. FM stereo comes and goes with the pilot, so
  `write` conforms every block to what the file was opened as rather than
  letting the two drift apart.

`lameenc` and `mutagen` are ordinary PyPI wheels - one a 157 KB statically
built LAME, the other pure Python - so they keep the property the packaging
phase turned on: nothing on the install path is a new unsigned binary. They
are still imported defensively, because an environment built before they were
added would otherwise raise on the DSP thread, where there is no user to show
it to. `available()` is the question to ask first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dsp.filters import LowPass
from .record import RecordingLimits, _Recorder

try:  # pragma: no cover - exercised by the environment, not by the tests
    import lameenc
except Exception as exc:  # noqa: BLE001 - any import fault means the same thing
    lameenc = None
    _LAME_ERROR: str | None = str(exc)
else:
    _LAME_ERROR = None

try:  # pragma: no cover - as above
    from mutagen.id3 import COMM, ID3, TALB, TCON, TDRC, TIT2, TPE1
except Exception as exc:  # noqa: BLE001
    ID3 = None  # type: ignore[assignment]
    _ID3_ERROR: str | None = str(exc)
else:
    _ID3_ERROR = None

EXTENSION = "mp3"
# CBR rather than VBR, and the reason is the disk guard: a recording that may
# run for hours has to be able to say what it will cost before it starts, and
# a variable rate cannot.
#
# The rate and the band limit are one decision and they are both taken from
# the broadcast, not from a preference. **Analog FM stereo is band-limited to
# 15 kHz** - the 19 kHz pilot has to sit above the audio, and the standard
# leaves it nowhere else - so everything a demodulator hands over above that
# is hiss the transmitter never sent. Feeding it to a lossy encoder is the
# worst thing that can be done to one: noise looks like signal at every
# frequency and there is nothing to mask it behind, so the bits go there
# instead of on the record. That was audible at 128 kbps as cymbals and
# reverb tails turning to a warble.
#
# So the input is cut at 15 kHz first, and then the rate only has to carry
# what the broadcast actually contains. For scale, an HD Radio hybrid
# simulcast carries its *whole* digital payload in about 100 kbps of HDC and
# the local HD1 measured 92 - see the HD Radio facts - so 160 kbps of MP3
# over a 15 kHz source is well clear of the thing it is recording. 20 kB a
# second, against the mono WAV recorder's 96.
STEREO_BITRATE_KBPS = 160
MONO_BITRATE_KBPS = 96
# The band limit above, as a number this module can be asked about.
BROADCAST_AUDIO_HZ = 15_000.0
# LAME's own scale, 0 best and 9 fastest. 2 is its "high quality" setting and
# costs about a millisecond per second of audio more than the default.
QUALITY = 2


def available() -> bool:
    """Whether anything can be encoded at all in this environment."""
    return lameenc is not None


def unavailable_reason() -> str | None:
    """Why not, in words that name the fix rather than the exception.

    The remedy is the same one the setup script gives for everything else,
    because an environment missing these was built before they were listed.
    """
    if lameenc is not None:
        return None
    return (
        "The MP3 encoder is not installed in this environment. Close "
        "BetterSDR and run BetterSDR.cmd --update to install it."
    )


def bytes_per_second(channels: int = 1) -> float:
    """What a recording costs on disk, for a warning the user can act on.

    20 kB/s in stereo and 12 in mono, against 96 kB/s for the mono WAV the
    Record audio button writes and 4.8 MB/s for raw IQ. An overnight session
    is a few hundred megabytes, which is the whole reason this file exists.
    """
    return bitrate_for(channels) * 1000.0 / 8.0


def bitrate_for(channels: int) -> int:
    return STEREO_BITRATE_KBPS if int(channels) >= 2 else MONO_BITRATE_KBPS


class Mp3Recorder(_Recorder):
    """A streaming MP3 file, with the same limits and guards as a WAV one.

    Duration comes from the frames handed in rather than from the bytes on
    disk, which is what `_Recorder` counts separately for exactly this case:
    at 128 kbps a byte count says nothing about how long the recording is.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = 48_000,
        limits: RecordingLimits | None = None,
        channels: int = 1,
        bitrate_kbps: int | None = None,
    ) -> None:
        super().__init__(path, sample_rate, limits)
        self.channels = 2 if int(channels) >= 2 else 1
        self.bitrate_kbps = int(
            bitrate_kbps if bitrate_kbps is not None else bitrate_for(self.channels)
        )
        self._encoder: object | None = None
        self._band: LowPass | None = None
        self._file: object | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> Mp3Recorder:
        if lameenc is None:
            self.stopped_reason = unavailable_reason()
            return self
        if not self._prepare():
            return self
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.bitrate_kbps)
        encoder.set_in_sample_rate(self.sample_rate)
        encoder.set_out_sample_rate(self.sample_rate)
        encoder.set_channels(self.channels)
        encoder.set_quality(QUALITY)
        # LAME's own lowpass is chosen from the bitrate and sits above the
        # broadcast limit at every rate this uses, so the band has to be
        # removed before the encoder sees it. `lameenc` exposes no lowpass
        # setter, which is the other half of the reason it is done here.
        self._band = LowPass(self.sample_rate, BROADCAST_AUDIO_HZ)
        # Held open for the life of the recording, like the WAV writers, so it
        # is closed by `stop` rather than by a `with`.
        handle = open(self.path, "wb")  # noqa: SIM115
        self._encoder = encoder
        self._file = handle
        return self

    def _close(self) -> None:
        """Flush the encoder's tail, then the file. Safe to call twice.

        LAME holds most of a frame back - it needs the next block to finish
        the one it is on - so skipping the flush loses up to about 25 ms off
        the end of every recording. Inaudible on a three-minute song and
        precisely the last word of a two-second radio transmission.

        It is only asked for when something was actually encoded. `flush` on
        an encoder that has never been handed a sample raises `RuntimeError:
        Not currently encoding`, and a recording opened and closed without a
        block in between is not a corner case - it is what happens whenever
        somebody presses the button and changes their mind.
        """
        handle, encoder = self._file, self._encoder
        self._file = self._encoder = None
        if handle is None:
            return
        try:
            if encoder is not None and self.frames_written:
                tail = bytes(encoder.flush())
                if tail:
                    handle.write(tail)
                    self.bytes_written += len(tail)
        finally:
            handle.close()

    @property
    def active(self) -> bool:
        return self._file is not None

    # -- writing -----------------------------------------------------------

    def write(self, audio: np.ndarray) -> None:
        """Queue float32 audio in [-1, 1], mono or stereo.

        Conformed to the channel count the file was opened with, for the same
        reason `AudioRecorder.write` does it: a stereo station dropping its
        pilot halfway through must not change what the file is.
        """
        if self._file is None or self._encoder is None or audio.size == 0:
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
        frames = block.shape[0]
        if self._band is not None:
            block = self._band.process(block)
        pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2")
        payload = bytes(self._encoder.encode(pcm.tobytes()))
        if payload:
            self._file.write(payload)
        # Accounted even when the encoder returned nothing this block, because
        # the frames went in and the duration limit is about them, not about
        # when LAME chose to hand a frame back.
        self._account(len(payload), frames)


@dataclass(frozen=True)
class Tags:
    """What a file says about itself once it is somewhere else.

    Everything here comes from something the station actually transmitted or
    from something the app actually measured. There is no field for a guess:
    an empty tag is a better answer than an invented one, and it is the same
    rule the classifier follows when it says "Unknown signal".
    """

    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    date: str = ""
    comment: str = ""


def write_tags(path: str | Path, tags: Tags) -> bool:
    """Attach ID3v2.4 to a finished file. False if it could not be done.

    Written after the recording is closed rather than reserved at the start,
    because on FM the title arrives some seconds *into* the song - so at the
    moment the file is opened there is nothing yet to write.

    A failure here is deliberately not an error anywhere upstream. An untagged
    MP3 is a slightly worse file, not a lost recording, and losing the
    recording over a metadata problem would be the wrong trade in a feature
    whose whole job is to be running unattended.
    """
    if ID3 is None:
        return False
    try:
        frames = ID3()
        if tags.title:
            frames.add(TIT2(encoding=3, text=[tags.title]))
        if tags.artist:
            frames.add(TPE1(encoding=3, text=[tags.artist]))
        if tags.album:
            frames.add(TALB(encoding=3, text=[tags.album]))
        if tags.genre:
            frames.add(TCON(encoding=3, text=[tags.genre]))
        if tags.date:
            frames.add(TDRC(encoding=3, text=[tags.date]))
        if tags.comment:
            frames.add(COMM(encoding=3, lang="eng", desc="", text=[tags.comment]))
        frames.save(str(path), v2_version=4)
    except Exception:  # noqa: BLE001 - see the docstring: never fatal
        return False
    return True


def read_tags(path: str | Path) -> Tags:
    """What a file already says. Used by the tests, and by nothing else."""
    if ID3 is None:
        return Tags()
    try:
        frames = ID3(str(path))
    except Exception:  # noqa: BLE001
        return Tags()

    def first(key: str) -> str:
        frame = frames.getall(key)
        return str(frame[0].text[0]) if frame and frame[0].text else ""

    return Tags(
        title=first("TIT2"),
        artist=first("TPE1"),
        album=first("TALB"),
        genre=first("TCON"),
        date=first("TDRC"),
        comment=first("COMM"),
    )


__all__ = [
    "EXTENSION",
    "BROADCAST_AUDIO_HZ",
    "MONO_BITRATE_KBPS",
    "STEREO_BITRATE_KBPS",
    "Mp3Recorder",
    "Tags",
    "available",
    "bitrate_for",
    "bytes_per_second",
    "read_tags",
    "unavailable_reason",
    "write_tags",
]
