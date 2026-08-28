"""Tests for the engine's thread-crossing pieces.

The engine itself needs a dongle, but the two things that carry data between
threads do not, and they are where a mistake would be least visible: a mailbox
that queued instead of dropping would build unbounded latency, and a frame
whose bin frequencies were off by half a bin would put every signal on screen
at the wrong place.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from bettersdr.core.device import MAX_TUNE_HZ, MIN_TUNE_HZ
from bettersdr.core.engine import (
    DSP_BLOCK_BYTES,
    DSP_BLOCK_SECONDS,
    SPECTRUM_HZ,
    DisplayFrame,
    Engine,
    Mailbox,
    dsp_block_bytes_for,
)
from bettersdr.dsp.psd import DEFAULT_FFT_SIZE


def frame(bins: int = 8, center_hz: float = 100e6, rate: float = 2.4e6) -> DisplayFrame:
    return DisplayFrame(
        spectrum_db=np.zeros(bins, dtype=np.float32),
        center_hz=center_hz,
        sample_rate=rate,
        bin_width_hz=rate / bins,
        channel_power_dbfs=-30.0,
        bandwidth_hz=200_000.0,
        squelch_open=None,
        audio_latency_s=0.15,
        underruns=0,
        ring_overruns=0,
    )


# -- Mailbox ---------------------------------------------------------------


def test_empty_mailbox_reads_as_nothing():
    assert Mailbox().peek() is None


def test_newest_value_replaces_the_old_one():
    """Dropping frames is correct for a display; queueing them is not."""
    box: Mailbox[int] = Mailbox()
    for value in range(100):
        box.put(value)
    assert box.peek() == 99


def test_peek_leaves_the_value_in_place():
    box: Mailbox[str] = Mailbox()
    box.put("frame")
    assert box.peek() == "frame"
    assert box.peek() == "frame"


def test_a_writer_thread_and_a_reader_thread_do_not_tear():
    """The GUI must never see a half-written slot while the DSP thread writes."""
    box: Mailbox[int] = Mailbox()
    box.put(0)
    stop = threading.Event()
    seen: list[int | None] = []

    def writer() -> None:
        for value in range(20_000):
            box.put(value)
        stop.set()

    thread = threading.Thread(target=writer)
    thread.start()
    while not stop.is_set():
        seen.append(box.peek())
    thread.join()

    assert all(value is not None for value in seen)
    assert box.peek() == 19_999


# -- DisplayFrame ----------------------------------------------------------

def test_frame_bin_frequencies_match_the_fft_convention():
    """Must agree with psd.Spectrum, or the display and detector disagree."""
    bins, rate, center = 8, 2.4e6, 100e6
    expected = center + np.fft.fftshift(np.fft.fftfreq(bins, 1.0 / rate))
    np.testing.assert_allclose(frame(bins, center, rate).frequencies(), expected)


def test_frame_is_centred_on_the_tuned_frequency():
    freqs = frame(bins=1024).frequencies()
    assert freqs[512] == 100e6
    assert freqs[0] < 100e6 < freqs[-1]


def test_frame_cannot_be_mutated_after_it_crosses_threads():
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        frame().center_hz = 1.0


def test_dsp_block_reproduces_the_default_at_full_rate():
    assert dsp_block_bytes_for(2_400_000) == DSP_BLOCK_BYTES


def test_dsp_block_covers_the_same_span_of_time_at_every_rate():
    for rate in (240_000, 960_000, 2_400_000):
        seconds = dsp_block_bytes_for(rate) / 2 / rate
        assert 0.5 * DSP_BLOCK_SECONDS <= seconds <= 1.5 * DSP_BLOCK_SECONDS, rate


@pytest.mark.parametrize("rate", [240_000, 288_000, 960_000, 1_200_000, 2_400_000])
def test_a_display_frame_can_always_be_filled_in_time(rate):
    """A block is eight FFT frames at 2.4 MS/s and less than one at 240 kS/s.

    Where it is less than one the engine carries the remainder; this checks
    the carry can never take longer than a display interval to fill, which is
    what would turn "the spectrum updates slowly" into "the spectrum stops".
    """
    samples_per_block = dsp_block_bytes_for(rate) / 2
    blocks_needed = math.ceil(DEFAULT_FFT_SIZE / samples_per_block)
    seconds = blocks_needed * samples_per_block / rate
    assert seconds <= 1.0 / SPECTRUM_HZ * 1.5, f"{rate}: {seconds*1000:.0f} ms"


# -- the tuning choke point ------------------------------------------------


def test_tuning_out_of_range_is_clamped_rather_than_raised():
    """Every tuning path goes through here, and the one that used to bypass
    the display's own clamp - click-to-tune - killed the reader thread."""
    engine = Engine()
    engine.tune(400_000)
    assert engine.center_hz == MIN_TUNE_HZ
    engine.tune(2_000_000_000)
    assert engine.center_hz == MAX_TUNE_HZ


def test_tuning_in_range_is_untouched():
    engine = Engine()
    engine.tune(94_900_000)
    assert engine.center_hz == 94_900_000


def test_asking_for_a_gain_measurement_without_a_radio_does_nothing():
    Engine().auto_gain()  # must not raise


# -- gain measurement scheduling -------------------------------------------


class FakeReader:
    """Captures submitted commands instead of touching a device."""

    def __init__(self) -> None:
        self.commands = []

    def submit(self, command) -> None:
        self.commands.append(command)


def test_two_callers_asking_at_once_produce_one_probe():
    """A band change asks, and the window change it triggers asks again. The
    probe holds the reader thread, so running it twice is a real cost - it
    emptied the audio buffer for 23 underruns on a single hop to the AM band.
    """
    engine = Engine()
    engine.reader = FakeReader()
    engine.auto_gain()
    engine.auto_gain()
    engine.auto_gain()
    assert len(engine.reader.commands) == 1


def test_a_later_ask_is_honoured_once_the_probe_has_run():
    engine = Engine()
    engine.reader = FakeReader()
    engine.auto_gain()

    class Dev:
        sample_rate = 2_400_000
        gains_db = [0.0]

        def set_manual_gain(self, on): pass
        def reset_buffer(self): pass
        def read(self, n): return np.full(n, 128, dtype=np.uint8)

    engine.reader.commands[0](Dev())
    engine.auto_gain()
    assert len(engine.reader.commands) == 2


def test_a_failing_probe_does_not_wedge_later_ones():
    """`_gain_pending` must clear even when the measurement raises, or every
    subsequent band change silently skips its gain measurement."""
    engine = Engine()
    engine.reader = FakeReader()
    engine.auto_gain()

    class Broken:
        sample_rate = 2_400_000

        @property
        def gains_db(self):
            raise RuntimeError("dongle went away")

        def set_manual_gain(self, on): pass

    with pytest.raises(RuntimeError):
        engine.reader.commands[0](Broken())

    engine.auto_gain()
    assert len(engine.reader.commands) == 2


# -- Phase 3 controls ------------------------------------------------------


def _drain(engine: Engine) -> None:
    """Run whatever the control surface queued for the DSP thread.

    The engine defers anything that touches a stateful DSP object so it
    happens on the thread that owns it. Without a running thread the tests
    have to turn the handle themselves.
    """
    engine._drain_commands()


def test_the_demodulator_is_built_at_unity_with_no_limiter():
    """Volume and the clip belong to the audio chain, at the end of the path.

    An AGC placed behind either of them would spend its range undoing a fixed
    attenuator, or would be unable to recover what a limiter had flattened.
    """
    engine = Engine()
    assert engine._demod.volume == 1.0
    assert engine._demod.clip is False


def test_volume_reaches_the_audio_chain_rather_than_the_demodulator():
    engine = Engine()
    engine.set_volume(0.9)
    assert engine.audio.volume == 0.9
    assert engine._demod.volume == 1.0


def test_deemphasis_override_replaces_the_stage_the_mode_chose():
    engine = Engine()
    assert engine._demod.deemphasis is not None  # wfm defaults to 75 us

    engine.set_deemphasis(None)
    _drain(engine)
    assert engine._demod.deemphasis is None

    engine.set_deemphasis(50.0)
    _drain(engine)
    assert engine._demod.deemphasis is not None


def test_deemphasis_override_survives_a_rebuild():
    """Changing bandwidth rebuilds the demodulator from the mode's defaults."""
    engine = Engine()
    engine.set_deemphasis(None)
    _drain(engine)
    engine.set_bandwidth(150_000.0)
    _drain(engine)
    assert engine._demod.deemphasis is None


def test_if_noise_reduction_is_installed_at_the_if_rate():
    """Not on the raw stream: 33% of a core there against 3% here."""
    engine = Engine()
    engine.set_if_noise_reduction(True)
    _drain(engine)
    assert engine._demod.if_stage is not None
    assert engine._demod.if_stage.sample_rate == engine._demod.if_rate
    assert engine._demod.if_stage.complex_input is True

    engine.set_if_noise_reduction(False)
    _drain(engine)
    assert engine._demod.if_stage is None


def test_offset_tuning_moves_the_tuner_but_not_the_display():
    engine = Engine()
    engine.tune(94_900_000)
    assert engine.device_center_hz == 94_900_000

    engine.set_offset_tuning(250_000.0)
    assert engine.center_hz == 94_900_000
    assert engine.device_center_hz == 95_150_000


def test_offset_tuning_is_clamped_like_every_other_tuning_path():
    engine = Engine()
    engine.tune(MAX_TUNE_HZ)
    engine.set_offset_tuning(1_000_000.0)
    assert engine.device_center_hz == MAX_TUNE_HZ


def test_display_settings_rebuild_the_spectrum():
    engine = Engine()
    engine.set_display(fft_size=1024, window="blackmanharris", smoothing=0.5)
    _drain(engine)
    assert engine.fft_size == 1024
    assert engine._spectrum.window_name == "blackmanharris"
    assert engine._spectrum.smoothing == 0.5


def test_display_settings_keep_the_ones_not_given():
    engine = Engine()
    engine.set_display(fft_size=8192)
    _drain(engine)
    engine.set_display(smoothing=0.25)
    _drain(engine)
    assert engine.fft_size == 8192
    assert engine._spectrum.smoothing == 0.25


def test_an_unknown_window_is_refused_at_the_call_site():
    """Rejected here rather than on the DSP thread, where the exception would
    unwind a worker nobody is watching."""
    with pytest.raises(ValueError, match="unknown window"):
        Engine().set_display(window="not-a-window")


def test_a_rate_change_rebuilds_the_front_end_around_the_new_rate():
    """A quarter-second DC time constant at 2.4 MS/s is two and a half
    seconds at 240 kS/s if the stage is left as it was."""
    engine = Engine()
    engine._apply_sample_rate(240_000)
    assert engine.front.sample_rate == 240_000.0
    assert engine._spectrum.sample_rate == 240_000.0


def test_a_rate_change_keeps_the_display_settings():
    engine = Engine()
    engine.set_display(fft_size=1024, window="flattop", smoothing=0.3)
    _drain(engine)
    engine._apply_sample_rate(240_000)
    assert engine.fft_size == 1024
    assert engine._spectrum.window_name == "flattop"
    assert engine._spectrum.smoothing == 0.3


def test_recording_reports_nothing_when_nothing_is_recording():
    status = Engine().recording
    assert not status.active
    assert status.audio_path is None


def test_recording_opens_and_closes_files(tmp_path):
    engine = Engine()
    engine.recording_dir = tmp_path
    engine.tune(94_900_000)

    engine.start_recording(audio=True, iq=True)
    _drain(engine)
    status = engine.recording
    assert status.active
    assert "94900000Hz_AF" in (status.audio_path or "")
    assert "94900000Hz_IQ" in (status.iq_path or "")

    engine._service_recorders(
        np.full(2_048, 128, dtype=np.uint8), np.zeros(512, dtype=np.float32)
    )
    engine.stop_recording()
    _drain(engine)
    assert not engine.recording.active
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_stopping_the_engine_closes_a_recording(tmp_path):
    """A WAV left without its final header length is unplayable, which loses
    the whole recording rather than its last block."""
    engine = Engine()
    engine.recording_dir = tmp_path
    engine.start_recording(iq=True)
    _drain(engine)
    engine.stop()
    assert not engine.recording.active
    written = list(tmp_path.glob("*.wav"))[0]
    assert written.stat().st_size >= 44  # a complete WAV header


def test_a_recorder_that_stops_itself_is_noticed(tmp_path):
    from bettersdr.audio.record import RecordingLimits

    engine = Engine()
    engine.recording_dir = tmp_path
    engine.recording_limits = RecordingLimits(max_bytes=1_024, min_free_bytes=0)
    engine.start_recording(iq=True)
    _drain(engine)

    for _ in range(4):
        engine._service_recorders(
            np.full(2_048, 128, dtype=np.uint8), np.zeros(0, dtype=np.float32)
        )

    status = engine.recording
    assert not status.active
    assert "size limit" in (status.message or "")


def test_capture_declines_when_the_radio_is_not_running():
    assert Engine().capture(0.1, timeout=0.1) is None


def test_capture_collects_what_it_was_asked_for():
    """The DSP thread fills the request on its way past; nothing else may
    reach into the sample stream."""
    from bettersdr.core.engine import _Capture

    request = _Capture(1_000)
    request.feed(np.ones(400, dtype=np.complex64))
    assert not request.done.is_set()
    request.feed(np.ones(900, dtype=np.complex64))
    assert request.done.is_set()
    assert request.result().size == 1_000


# -- RDS ---------------------------------------------------------------------


def test_rds_is_attached_to_broadcast_fm_and_to_nothing_else():
    engine = Engine()
    assert engine._rds is not None
    assert engine._demod.mpx_sink is engine._rds

    engine._rebuild("nfm", None)
    assert engine._rds is None


def test_narrowing_the_channel_filter_past_the_subcarrier_detaches_rds():
    """RDS sits 57 kHz off centre, so a narrow filter removes it from the air.

    Leaving the receiver attached there would burn CPU decoding a subcarrier
    the channel filter had already thrown away, and report a station that had
    simply stopped being received rather than stopped transmitting.
    """
    engine = Engine()
    engine._rebuild("wfm", 80_000.0)
    assert engine._rds is None

    engine._rebuild("wfm", 200_000.0)
    assert engine._rds is not None


def test_rds_can_be_switched_off():
    engine = Engine()
    engine.set_rds(False)
    engine._apply_rds()
    assert engine._rds is None
    assert engine._demod.mpx_sink is None


# -- stereo ------------------------------------------------------------------


def test_stereo_is_attached_to_broadcast_fm_and_to_nothing_else():
    engine = Engine()
    assert engine._stereo is not None
    assert engine._demod.stereo is engine._stereo

    engine._rebuild("nfm", None)
    assert engine._stereo is None


def test_narrowing_the_channel_filter_past_the_subcarrier_detaches_stereo():
    """The difference channel reaches 53 kHz out, so a narrow filter loses it.

    A decoder left attached there would sit on a pilot the channel filter had
    already removed, and the badge would report a station that had stopped
    being received rather than one that had stopped transmitting.
    """
    engine = Engine()
    engine._rebuild("wfm", 80_000.0)
    assert engine._stereo is None

    engine._rebuild("wfm", 200_000.0)
    assert engine._stereo is not None


def test_stereo_can_be_switched_off():
    engine = Engine()
    engine.set_stereo(False)
    engine._apply_stereo()
    assert engine._stereo is None
    assert engine._demod.stereo is None


# -- scanning ----------------------------------------------------------------
#
# Two things belonged to whichever band the radio was *listening* to and
# leaked into the sweep of a different one, both reproducible from the GUI in
# the same six clicks: scan the AM band, listen to a station in it, go back to
# Discover, pick FM Radio, scan.


class FakeScanReader(FakeReader):
    """A reader that also remembers the window it was asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.sample_rates: list[int] = []
        self.tuned: list[int] = []

    def set_sample_rate(self, rate: int) -> None:
        self.sample_rates.append(int(rate))

    def tune(self, hz: int) -> None:
        self.tuned.append(int(hz))


class FakeGainDevice:
    """Just enough device for `choose_gain` to run against."""

    sample_rate = 2_400_000
    gains_db = (0.0, 20.0, 40.0)

    def __init__(self) -> None:
        self.center_freq = 0
        self.manual = False

    def set_manual_gain(self, enabled: bool) -> None:
        self.manual = bool(enabled)

    def reset_buffer(self) -> None:
        pass

    def read(self, n: int) -> np.ndarray:
        return np.full(n, 128, dtype=np.uint8)


def _scan_steps(engine: Engine) -> int:
    update = engine.scan_update()
    assert update is not None
    return update.progress.steps


def test_a_scan_does_not_inherit_the_window_it_was_listening_through():
    """The window belongs to the band being swept, not to the last station.

    Listening to an AM station leaves the engine at 240 kHz, which is right
    for listening and catastrophic for a sweep of FM broadcast: 141 steps
    instead of 12, through a window narrower than a single FM station, so the
    scan crawled across the dial and mismeasured everything it found.
    """
    engine = Engine()
    engine.reader = FakeScanReader()
    engine.sample_rate = 240_000  # left over from the AM band

    engine.start_scan(88e6, 108e6)
    assert _scan_steps(engine) == 12


def test_a_band_that_asks_for_a_narrow_window_still_gets_one():
    """The fix above must not undo the reason `sample_rate_hz` exists: the AM
    band's own preference is still what a sweep of it uses."""
    engine = Engine()
    engine.reader = FakeScanReader()

    engine.start_scan(530e3, 1.7e6, sample_rate_hz=240_000)
    assert _scan_steps(engine) == 9


def test_a_sweep_below_the_upconverter_narrows_even_unasked():
    """A 2.4 MHz window at 530 kHz contains 0 Hz, where the V4's oscillator
    leak sits 65 dB above the noise. `safe_sample_rate` still guards this."""
    engine = Engine()
    engine.reader = FakeScanReader()

    engine.start_scan(530e3, 1.7e6)
    _drain(engine)
    assert engine.sample_rate < 2_400_000


def test_a_scan_measures_the_gain_of_the_band_it_is_about_to_sweep():
    """Coming from an AM station the tuner sits near 34 dB, over 20 dB more
    than the FM band takes without clipping the 8-bit ADC - and a clipped
    front end manufactures spurs the detector reports as stations."""
    engine = Engine()
    engine.reader = FakeScanReader()
    engine.start_scan(88e6, 108e6)
    _drain(engine)

    assert engine.reader.commands, "no gain probe was queued for the sweep"
    device = FakeGainDevice()
    engine.reader.commands[-1](device)
    # The first tile centre, offset off-channel the way the sweep tunes.
    assert device.center_freq == 88_937_000
    assert engine.gain is not None


def test_a_failing_scan_gain_probe_does_not_abandon_the_sweep():
    """The sweep still measures at the old gain; a refused command is a
    diagnosable condition, not a reason to stop."""
    engine = Engine()
    engine.reader = FakeScanReader()
    engine.start_scan(88e6, 108e6)
    _drain(engine)

    class Broken(FakeGainDevice):
        def set_manual_gain(self, enabled: bool) -> None:
            raise RuntimeError("dongle went away")

    engine.reader.commands[-1](Broken())  # must not raise
    assert engine.last_error is not None
    assert engine._sweeper is not None


def test_ending_a_sweep_re_measures_the_gain_for_where_it_returns_to():
    """`_probe_scan_gain` set the front end up for the band that was swept,
    which is not the band being listened to."""
    engine = Engine()
    engine.reader = FakeScanReader()
    engine.center_hz = 710_000
    engine.start_scan(88e6, 108e6)
    _drain(engine)
    before = len(engine.reader.commands)

    engine._end_scan()
    assert len(engine.reader.commands) == before + 1
    assert engine.reader.tuned[-1] == 710_000


def test_a_second_end_of_scan_does_not_probe_again():
    """`stop_scan` queues `_end_scan` unconditionally and a sweep that ends on
    its own has already run it. A repeat probe is 340 ms of dead air."""
    engine = Engine()
    engine.reader = FakeScanReader()
    engine.start_scan(88e6, 108e6)
    _drain(engine)
    engine._end_scan()
    after_first = len(engine.reader.commands)

    engine._end_scan()
    assert len(engine.reader.commands) == after_first
