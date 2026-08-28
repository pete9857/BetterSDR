"""Tests for the demodulators.

Each test plants a signal carrying a known audio tone and asserts the
demodulator recovers that exact tone. This is the strongest claim available
without hardware: if the tone comes back at the right frequency with the right
amplitude and nothing else nearby, the whole chain - channel filter,
demodulation, de-emphasis, audio decimation - is working.
"""

from __future__ import annotations

import numpy as np
import pytest

from bettersdr.dsp import demod
from bettersdr.dsp.demod import AUDIO_RATE

from . import synth

RATE = 2_400_000
# 0.2 s of baseband, which gives 9600 audio samples and 5 Hz FFT bins.
SAMPLES = 480_000


def dominant_tone(audio: np.ndarray, rate: int = AUDIO_RATE) -> float:
    """The loudest audio frequency above 100 Hz."""
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / rate)
    usable = freqs > 100.0
    return float(freqs[usable][np.argmax(spectrum[usable])])


def tone_purity_db(audio: np.ndarray, tone_hz: float, rate: int = AUDIO_RATE) -> float:
    """How far the tone stands above everything else in the audio."""
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / rate)
    near_tone = np.abs(freqs - tone_hz) < 50.0
    elsewhere = (~near_tone) & (freqs > 100.0)
    peak = float(np.max(spectrum[near_tone]))
    floor = float(np.sqrt(np.mean(spectrum[elsewhere] ** 2)))
    return 20.0 * np.log10(peak / max(floor, 1e-12))


# -- Chain planning --------------------------------------------------------


def test_wfm_decimates_to_a_sensible_intermediate_rate():
    wfm = demod.WfmDemodulator(RATE)
    # 240 kHz comfortably carries a 200 kHz channel and divides evenly to 48 k.
    assert wfm.if_rate == 240_000
    assert wfm.block_multiple == 50


def test_rejects_sample_rate_that_is_not_a_multiple_of_audio_rate():
    with pytest.raises(ValueError, match="whole multiple"):
        demod.WfmDemodulator(2_048_000)


def test_create_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        demod.create("fsk", RATE)


def test_mode_table_describes_every_mode():
    table = demod.mode_table()
    assert {info.mode for info in table} == set(demod.MODES)
    assert all(info.label and info.description for info in table)


# -- Tone recovery ---------------------------------------------------------


def test_wfm_recovers_broadcast_tone():
    signal = synth.fm(SAMPLES, 0.0, tone_hz=1_000.0, deviation_hz=75_000.0)
    audio = demod.WfmDemodulator(RATE).process(signal)

    assert audio.size == SAMPLES // 50
    assert dominant_tone(audio) == pytest.approx(1_000.0, abs=10.0)
    assert tone_purity_db(audio, 1_000.0) > 30.0


def test_nfm_recovers_narrowband_tone():
    signal = synth.fm(SAMPLES, 0.0, tone_hz=1_000.0, deviation_hz=2_500.0)
    audio = demod.NfmDemodulator(RATE).process(signal)

    assert dominant_tone(audio) == pytest.approx(1_000.0, abs=10.0)
    assert tone_purity_db(audio, 1_000.0) > 30.0


def test_am_recovers_envelope_tone():
    signal = synth.am(SAMPLES, 0.0, tone_hz=1_000.0, depth=0.8)
    audio = demod.AmDemodulator(RATE).process(signal)

    assert dominant_tone(audio) == pytest.approx(1_000.0, abs=10.0)
    assert tone_purity_db(audio, 1_000.0) > 30.0


def test_usb_recovers_tone_from_upper_sideband():
    # A bare complex exponential at +1 kHz *is* a USB signal carrying 1 kHz.
    signal = synth.carrier(SAMPLES, 1_000.0, amplitude=0.5)
    audio = demod.UsbDemodulator(RATE).process(signal)

    assert dominant_tone(audio) == pytest.approx(1_000.0, abs=10.0)


def test_lsb_recovers_tone_from_lower_sideband():
    signal = synth.carrier(SAMPLES, -1_000.0, amplitude=0.5)
    audio = demod.LsbDemodulator(RATE).process(signal)

    assert dominant_tone(audio) == pytest.approx(1_000.0, abs=10.0)


def test_usb_rejects_the_opposite_sideband():
    wanted = demod.UsbDemodulator(RATE).process(
        synth.carrier(SAMPLES, 1_000.0, amplitude=0.5)
    )
    unwanted = demod.UsbDemodulator(RATE).process(
        synth.carrier(SAMPLES, -1_000.0, amplitude=0.5)
    )
    quiet = float(np.std(unwanted))
    loud = float(np.std(wanted))
    assert 20 * np.log10(quiet / loud) < -35.0


def test_cw_shifts_a_silent_carrier_to_an_audible_tone():
    # A CW carrier tuned 700 Hz off centre should come out as a 700 Hz beep.
    signal = synth.carrier(SAMPLES, 700.0, amplitude=0.5)
    audio = demod.CwDemodulator(RATE, tone_hz=700.0).process(signal)

    assert dominant_tone(audio) == pytest.approx(700.0, abs=15.0)


# -- Streaming behaviour ---------------------------------------------------


def test_block_size_does_not_change_the_result():
    """Ragged blocks must give the same audio as one contiguous call.

    USB reads arrive in multiples of 512 bytes, which does not divide evenly
    into the decimation chain, so the demodulator has to buffer the remainder.
    """
    signal = synth.fm(SAMPLES, 0.0, tone_hz=1_000.0)

    one_shot = demod.WfmDemodulator(RATE).process(signal)

    streamed = demod.WfmDemodulator(RATE)
    # 8192 complex samples is the real read size, and 8192 % 50 != 0.
    chunks = [
        streamed.process(signal[i : i + 8_192]) for i in range(0, signal.size, 8_192)
    ]
    joined = np.concatenate(chunks)

    assert joined.size == one_shot.size
    np.testing.assert_allclose(joined, one_shot, atol=1e-5)


def test_block_size_does_not_change_sideband_output():
    """USB runs the FFT filter path, which joins blocks differently from WFM."""
    signal = synth.carrier(SAMPLES, 1_000.0, amplitude=0.5)

    one_shot = demod.UsbDemodulator(RATE).process(signal)

    streamed = demod.UsbDemodulator(RATE)
    joined = np.concatenate(
        [streamed.process(signal[i : i + 8_192]) for i in range(0, signal.size, 8_192)]
    )

    assert joined.size == one_shot.size
    np.testing.assert_allclose(joined, one_shot, atol=1e-5)


def test_partial_block_produces_no_audio_yet():
    wfm = demod.WfmDemodulator(RATE)
    assert wfm.process(np.zeros(10, dtype=np.complex64)).size == 0


def test_channel_power_tracks_signal_level():
    strong = synth.fm(96_000, 0.0, amplitude=0.5)
    weak = synth.fm(96_000, 0.0, amplitude=0.05)

    loud = demod.WfmDemodulator(RATE)
    loud.process(strong)
    quiet = demod.WfmDemodulator(RATE)
    quiet.process(weak)

    assert loud.channel_power_dbfs == pytest.approx(-6.0, abs=1.5)
    assert quiet.channel_power_dbfs == pytest.approx(-26.0, abs=1.5)


def test_squelch_silences_noise_but_passes_a_station():
    noise = synth.noise(SAMPLES, rms=0.02)
    station = synth.fm(SAMPLES, 0.0, tone_hz=1_000.0, amplitude=0.5)

    gated = demod.WfmDemodulator(RATE, squelch_dbfs=-30.0)
    assert float(np.max(np.abs(gated.process(noise)))) == 0.0

    gated.reset()
    audio = gated.process(station)
    assert float(np.max(np.abs(audio))) > 0.05


def test_output_never_exceeds_full_scale():
    # Deliberately over-driven: deviation far beyond what the mode expects.
    signal = synth.fm(96_000, 0.0, tone_hz=500.0, deviation_hz=200_000.0)
    audio = demod.WfmDemodulator(RATE, volume=4.0).process(signal)
    assert float(np.max(np.abs(audio))) <= 1.0


def test_a_sharper_filter_rejects_a_neighbour_a_softer_one_lets_through():
    """SDR#'s "filter order": more taps per branch, a steeper skirt.

    Planted just outside the channel, where the difference between a 12-tap
    and a 96-tap branch is the whole point of exposing the control.
    """
    rate = 2_400_000.0
    n = 240_000
    t = np.arange(n) / rate
    # 9 kHz off centre with a 12.5 kHz channel: right on the filter's skirt.
    neighbour = (0.5 * np.exp(2j * np.pi * 9_000.0 * t)).astype(np.complex64)

    leaked = {}
    for taps in (12, 96):
        stage = demod.NfmDemodulator(rate, bandwidth_hz=12_500.0, filter_taps=taps)
        stage.process(neighbour)
        leaked[taps] = stage.channel_power_dbfs

    assert leaked[96] < leaked[12] - 6.0


def test_filter_taps_default_to_the_shared_constant():
    from bettersdr.dsp.filters import DEFAULT_TAPS_PER_PHASE

    assert demod.create("wfm", 2_400_000.0).filter_taps == DEFAULT_TAPS_PER_PHASE
