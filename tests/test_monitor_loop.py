"""The monitor as the engine actually runs it, against a synthetic band.

`test_monitor.py` drives the state machine and `test_voice.py` drives the
detector. Everything between them lives in `Engine` and is where the real
risks are: the tuner is parked off-channel and the block shifted back in
software, the sweep's samples have to be thrown away before the audition
starts, the demodulator is cached across blocks, and the audio is classified
before the audio chain and played after it.

None of that needs a dongle. The fake radio below answers reads from a scene
that responds to retuning, exactly as `tests/synth.Air` does for the sweeper -
with one addition it needs and `Air` does not: the samples are continuous
across reads, because an FM discriminator and a phase-continuous frequency
shifter both carry history between blocks and a scene that restarted its phase
every read would hide any fault in either.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.core.engine import Engine
from bettersdr.scan import monitor as mon
from bettersdr.scan import voice
from bettersdr.scan.sweeper import TUNE_OFFSET_HZ
from tests import synth_audio as sa

RATE = 240_000
# A slice of the 2 m amateur band. Somewhere the band plan has an opinion
# about, on purpose: the audition demodulates a channel with the mode the
# classifier chose, so a stretch of dial with no allocation would have the
# test measuring what shape alone makes of an FM signal rather than what the
# monitor does with a known one.
BAND_LOW = 145_000_000
BAND_HIGH = 145_200_000
STATION_HZ = 145_100_000


def _station(audio48, rate=RATE, deviation_hz=2_500.0, amplitude=0.45):
    """One NFM transmitter's complex baseband, as long as the audio given."""
    up = int(rate // sa.RATE)
    block = np.repeat(np.asarray(audio48, dtype=np.float64), up)
    smooth = np.hanning(up * 2 + 1)
    block = np.convolve(block, smooth / smooth.sum(), mode="same")
    phase = 2 * np.pi * deviation_hz * np.cumsum(block) / rate
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


class Ring:
    """Generates IQ on demand, at whatever the radio is currently tuned to.

    Continuous in time: `_t` only ever moves forward, so two consecutive reads
    join without a phase step. That is the whole reason this is not a list of
    pre-made blocks.
    """

    def __init__(self, radio: FakeRadio) -> None:
        self.radio = radio
        self.overruns = 0

    def clear(self) -> None:
        pass

    def read(self, n_bytes: int, timeout: float | None = None) -> np.ndarray:
        samples = n_bytes // 2
        iq = self.radio.sample(samples)
        raw = np.empty(samples * 2, dtype=np.uint8)
        raw[0::2] = np.clip(np.real(iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        raw[1::2] = np.clip(np.imag(iq) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        return raw


class FakeDevice:
    """Just enough of a device to be retuned and to have its gain measured.

    The gain probe is included rather than stubbed out because it is the
    first thing `_begin_monitor` does and it runs on the reader thread with
    no capture in flight - so a session that never reached it would be
    skipping the one part of the start-up that has bitten this app before.
    """

    def __init__(self, radio: FakeRadio) -> None:
        self.radio = radio
        self.sample_rate = RATE
        self.gains_db = [0.0, 15.0, 30.0]
        self.gain_db = 0.0

    @property
    def center_freq(self) -> int:
        return self.radio.center

    @center_freq.setter
    def center_freq(self, hz: int) -> None:
        self.radio.center = int(hz)
        # Where the sweep actually went. `Reader.tune` is only the way back
        # at the end of a session; every step of the sweep gets there by
        # submitting a command, which is the path this records.
        self.radio.visited.append(int(hz))

    def reset_buffer(self) -> None:
        pass

    def set_manual_gain(self, on: bool) -> None:
        pass

    def read(self, n_bytes: int) -> np.ndarray:
        return self.radio.ring.read(n_bytes)


class FakeRadio:
    """The reader's surface, over a band with one transmitter in it."""

    def __init__(self, baseband: np.ndarray, noise_rms: float = 0.012) -> None:
        self.baseband = baseband
        self.noise_rms = float(noise_rms)
        self.center = STATION_HZ
        self.ring = Ring(self)
        self.last_error: str | None = None
        self.device = FakeDevice(self)
        self._t = 0
        self._rng = np.random.default_rng(4)
        self.tuned_to: list[int] = []
        self.visited: list[int] = []

    # -- the reader's API --------------------------------------------------

    def submit(self, command) -> None:
        command(self.device)

    def tune(self, hz: int) -> None:
        self.center = int(hz)
        self.tuned_to.append(int(hz))

    def set_sample_rate(self, rate: int) -> None:
        pass

    def set_gain(self, db) -> None:
        self.device.gain_db = db

    def set_gapless(self, on: bool) -> None:
        pass

    # -- the air -----------------------------------------------------------

    def sample(self, count: int) -> np.ndarray:
        start = self._t % (self.baseband.size - count - 1)
        self._t += count
        offset = STATION_HZ - self.center
        block = self._rng.normal(0, self.noise_rms, count) + 1j * self._rng.normal(
            0, self.noise_rms, count
        )
        if abs(offset) < RATE / 2.0:
            time = (np.arange(count) + start) / RATE
            block = block + self.baseband[start : start + count] * np.exp(
                2j * np.pi * offset * time
            )
        return block.astype(np.complex64)


@pytest.fixture(scope="module")
def talking() -> np.ndarray:
    return _station(sa.speech(8.0, seed=1))


@pytest.fixture(scope="module")
def paging() -> np.ndarray:
    return _station(sa.data(8.0, baud=1_200, seed=1))


def _engine(radio: FakeRadio) -> Engine:
    """An engine with a radio and no threads, driven a turn at a time."""
    engine = Engine(sample_rate=RATE)
    engine.reader = radio
    engine.center_hz = 100_000_000
    return engine


def _run(engine: Engine, radio: FakeRadio, turns: int = 4000) -> mon.MonitorState:
    """Turn the DSP loop by hand until the monitor holds, or give up.

    Deliberately not a thread. Everything the loop does here is synchronous,
    so a test that started one would be testing the scheduler.
    """
    while engine._commands.qsize():
        engine._commands.get()()
    for _ in range(turns):
        while engine._commands.qsize():
            engine._commands.get()()
        if engine._monitor is None:
            break
        engine._monitor_step()
        if engine._monitor is not None and engine._monitor.phase == mon.HOLDING:
            break
    return engine.monitor_update()


# -- the sweep half ---------------------------------------------------------


def test_the_sweep_finds_the_station_and_counts_it(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(
        BAND_LOW, BAND_HIGH, band_name="Test", sample_rate_hz=RATE, listen=False
    )
    state = _run(engine, radio, turns=200)
    assert state is not None
    found = [c for c in state.channels if abs(c.frequency_hz - STATION_HZ) < 20_000]
    assert found, [c.frequency_hz for c in state.channels]
    assert found[0].sightings >= 2
    engine.stop_monitor()


def test_the_dial_the_user_is_on_never_moves(talking):
    """A borrowed radio changes nothing anybody can see.

    `center_hz` is what every view reads to decide what the radio is pointed
    at, and the aircraft screen's version of this bug demodulated an FM
    station as `raw` at 49.6 dB of gain.
    """
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE, listen=False)
    _run(engine, radio, turns=200)
    assert engine.center_hz == 100_000_000
    engine.stop_monitor()


def test_the_window_and_the_frequency_come_back(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.sample_rate = 2_400_000
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE, listen=False)
    _run(engine, radio, turns=100)
    assert engine.sample_rate == RATE
    engine.stop_monitor()
    while engine._commands.qsize():
        engine._commands.get()()
    assert engine.sample_rate == 2_400_000
    assert radio.center == engine.device_center_hz


# -- the listening half -----------------------------------------------------


def test_a_talking_channel_is_found_heard_and_held(talking):
    """The whole feature, end to end, with no dongle in it."""
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, band_name="Test", sample_rate_hz=RATE)
    state = _run(engine, radio)
    assert state is not None
    assert state.phase == mon.HOLDING, state.phase
    assert state.target_hz == pytest.approx(STATION_HZ, abs=20_000)
    held = [c for c in state.channels if c.heard_voice]
    assert held, [(c.frequency_hz, c.sound) for c in state.channels]
    assert held[0].verdict.kind == voice.VOICE
    engine.stop_monitor()


def test_the_tuner_is_parked_off_the_channel_it_is_listening_to(talking):
    """The dongle's own DC offset must not land inside a narrow channel.

    Same reasoning as the sweep's `TUNE_OFFSET_HZ`, and a stronger case: a
    sweep only has to measure through the notch, while this has to listen
    through it.
    """
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    state = _run(engine, radio)
    assert state.phase == mon.HOLDING
    assert radio.center == pytest.approx(state.target_hz + TUNE_OFFSET_HZ, abs=1)


def test_the_channel_still_demodulates_from_off_centre(talking):
    """The shift has to be the right way round, and nothing else says so.

    A sign error here parks the receiver two offsets away from the station
    and every channel in the band comes back as static - which looks exactly
    like a quiet band.
    """
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    state = _run(engine, radio)
    channel = next(c for c in state.channels if c.verdict is not None)
    assert channel.verdict.kind == voice.VOICE
    assert channel.verdict.features.level_dbfs > voice.SILENCE_DBFS


def test_a_pager_is_heard_and_the_sweep_carries_on(paging):
    """Data is worth knowing about and not worth stopping for."""
    radio = FakeRadio(paging)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    heard = None
    for _ in range(4000):
        while engine._commands.qsize():
            engine._commands.get()()
        engine._monitor_step()
        state = engine.monitor_update()
        found = [c for c in state.channels if c.verdict is not None]
        if found:
            heard = found[0]
            break
    assert heard is not None
    assert heard.verdict.kind == voice.DATA
    assert not heard.heard_voice
    assert engine._monitor.phase != mon.HOLDING
    engine.stop_monitor()


def test_the_sound_card_only_opens_once_the_answer_is_yes(talking):
    """Opening it for every audition puts a click on every channel glanced at."""
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    while engine._commands.qsize():
        engine._commands.get()()
    for _ in range(4000):
        engine._monitor_step()
        if engine._monitor.phase == mon.AUDITIONING:
            assert not engine._monitor_playing
        if engine._monitor.phase == mon.HOLDING:
            break
    assert engine._monitor_playing


def test_skipping_a_held_channel_puts_the_radio_back_to_sweeping(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    state = _run(engine, radio)
    assert state.phase == mon.HOLDING
    engine.monitor_skip(state.target_hz)
    while engine._commands.qsize():
        engine._commands.get()()
    assert engine._monitor.phase == mon.SWEEPING
    assert not engine._monitor_playing
    assert engine.monitor_update().channels[0].skipped


def test_stopping_mid_hold_leaves_nothing_behind(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    _run(engine, radio)
    engine.stop_monitor()
    while engine._commands.qsize():
        engine._commands.get()()
    assert engine._monitor is None
    assert engine._monitor_sweeper is None
    assert not engine._monitor_playing
    assert not engine.monitoring


def test_a_monitor_session_refuses_to_start_twice(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    assert engine._commands.qsize() == 1


def test_a_scan_cannot_start_while_monitoring(talking):
    """One thing borrows the radio at a time, or neither gives it back."""
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    engine.start_scan(BAND_LOW, BAND_HIGH, sample_rate_hz=RATE)
    engine.start_adsb()
    assert not engine.scanning
    assert not engine.receiving_adsb


# -- watching more than one stretch of dial ----------------------------------
#
# The Expert discovery screen lets several ranges be watched together. The
# session then crosses a boundary several times a cycle, and everything the
# loop does per range - plan the steps, set the window, set the gain, pick
# which range a channel it wants to audition belongs to - has to survive it.

OTHER_LOW = 146_000_000
OTHER_HIGH = 146_200_000


def test_a_session_sweeps_every_range_it_was_given(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(
        band_name="2 ranges",
        ranges=[
            (BAND_LOW, BAND_HIGH, RATE),
            (OTHER_LOW, OTHER_HIGH, RATE),
        ],
        listen=False,
    )
    _run(engine, radio, turns=400)
    # Tile centres, not band edges: the last step of a range deliberately
    # reaches past it so there is no hole at the top.
    tiles = {hz - TUNE_OFFSET_HZ for hz in radio.visited}
    for low, high in ((BAND_LOW, BAND_HIGH), (OTHER_LOW, OTHER_HIGH)):
        middle = (low + high) / 2
        assert any(abs(hz - middle) < RATE for hz in tiles), (
            f"never looked at {middle / 1e6} MHz: {sorted(tiles)}"
        )
    engine.stop_monitor()


def test_a_station_is_still_found_and_held_across_two_ranges(talking):
    """The empty half must not cost the busy half its channel.

    A sweep of several ranges is a longer cycle, and the ledger's persistence
    gate and the revisit timer are both expressed in passes and seconds -
    so a second range stretching the cycle is exactly the sort of change that
    quietly stops a scanner ever stopping on anything.
    """
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(
        band_name="2 ranges",
        ranges=[
            (BAND_LOW, BAND_HIGH, RATE),
            (OTHER_LOW, OTHER_HIGH, RATE),
        ],
    )
    state = _run(engine, radio)
    assert state is not None
    assert state.phase == mon.HOLDING
    assert state.target_hz is not None
    assert abs(state.target_hz - STATION_HZ) < 20_000
    held = [c for c in state.channels if abs(c.frequency_hz - STATION_HZ) < 20_000]
    assert held and held[0].verdict is not None
    assert held[0].verdict.kind == voice.VOICE
    engine.stop_monitor()


def test_a_channel_is_auditioned_through_the_range_it_belongs_to(talking):
    radio = FakeRadio(talking)
    engine = _engine(radio)
    engine.start_monitor(
        ranges=[(BAND_LOW, BAND_HIGH, RATE), (OTHER_LOW, OTHER_HIGH, RATE)]
    )
    while engine._commands.qsize():
        engine._commands.get()()
    assert engine._range_containing(STATION_HZ) == 0
    assert engine._range_containing(OTHER_LOW + 50_000) == 1
    # A frequency in neither is not forced into one of them.
    assert engine._range_containing(400_000_000) is None
    engine.stop_monitor()
