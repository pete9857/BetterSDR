# BetterSDR

A beginner-friendly SDR receiver for the **RTL-SDR Blog V4** on Windows.

## The product idea, in one paragraph

Every mainstream SDR app (SDR#, SDR++, HDSDR) opens on a blank spectrum and expects you to already know what modulation to pick, what bandwidth to set, and which frequencies are worth visiting. BetterSDR inverts that: it sweeps the air around you and shows **a list of what it found in plain English**, the way a Wi-Fi picker shows networks. The mental model is *scan and browse*, not *tune and configure*.

Two principles drive most design decisions here:

1. **Beginner-friendly must not mean capability-capped.** If a user outgrows BetterSDR they leave for SDR#, which defeats the point. Everything SDR# does ships — revealed progressively through Simple / Standard / Expert levels, never removed. The DSP engine is identical at every level; only visible controls change.
2. **The app explains itself.** Signal classification is rule-based rather than ML *specifically so* the UI can say "constant power, 150 kHz wide, sits in the 88–108 MHz broadcast band → FM radio station." Explainability is the product, not a debugging aid.

Full roadmap and phase breakdown: **[docs/PLAN.md](docs/PLAN.md)**.

## Current status

| Phase | State |
|---|---|
| **0 — Device layer + Driver Doctor** | **Complete and verified on hardware** |
| **1 — Listen** (demod, audio, spectrum/waterfall) | **Complete and verified on hardware** |
| **2 — Discovery** (sweep, detect, classify) | **Complete and verified on hardware** |
| **3 — SDR# parity** | **Complete and verified on hardware** (small gaps and untested paths listed in docs/PLAN.md) |
| **4 — Decoders** (RDS, HD Radio, ADS-B, POCSAG) | **RDS and FM stereo complete and verified on hardware**; the rest not started |
| **5 — Packaging** | Not started |

### Driver state on this machine

WinUSB is bound to **interface 0** (done manually with Zadig on 2026-08-27), and the
full path is verified: tuner reads `R828D`, capture runs at 2.4 MS/s, FM demodulates
to clean audio. Interface 1 correctly remains driverless — that is the expected and
desired state, not a fault.

Check current state any time with:

```
.venv/Scripts/python.exe -m bettersdr.core.device --info
```

Exit code 0 means ready; 1 means it prints the specific remedy.

**Still to build: `core/installer.py`** — drives a bundled `wdi-simple.exe` (from
libwdi, the library Zadig is built on) to bind WinUSB from inside the app, so a
first-time user never has to find Zadig. See **Amendment 1** in docs/PLAN.md. The
Zadig walkthrough in `doctor.py` stays as the fallback for a declined UAC prompt or
a machine where policy blocks driver installation.

Two things to know before working on it:

- **`wdi-simple.exe` must be compiled from libwdi source.** Upstream ships only
  `Zadig.exe`; the third-party prebuilt on GitHub is a 2023 binary that installs a
  self-signed certificate, which is not a supply-chain risk worth taking. MSVC
  14.51 (VS Build Tools 2026) is installed on this machine and can build it.
- **Testing it requires reverting the driver**, which the user has asked us not to
  do — they want a working dongle. Device Manager alone is not enough either: the
  package stays in the DriverStore and rebinds on replug, so a real revert needs
  `pnputil /delete-driver oemNN.inf /uninstall /force`. Treat the install path as
  testable only on a second machine or with explicit permission.

## Running things

```bash
.venv/Scripts/python.exe -m bettersdr.core.device --info          # driver + hardware check
.venv/Scripts/python.exe -m bettersdr.core.device --info --freq 98.5
.venv/Scripts/python.exe -m pytest -q                             # no hardware needed
.venv/Scripts/python.exe -m ruff check .

# Tune and play. Auto-picks gain; --mode is any of wfm/nfm/am/usb/lsb/cw/dsb/raw.
.venv/Scripts/python.exe -m bettersdr.listen --freq 94.9
.venv/Scripts/python.exe -m bettersdr.listen --freq 162.55 --mode nfm --squelch -35
.venv/Scripts/python.exe -m bettersdr.listen --list-audio
```

```bash
# The GUI. --level is simple/standard/expert; the DSP engine is the same in all
# three. Every flag defaults to what was remembered last time, so the app opens
# where it was left; --no-settings ignores and does not write that file.
.venv/Scripts/python.exe -m bettersdr.app
.venv/Scripts/python.exe -m bettersdr.app --freq 162.55 --level expert
.venv/Scripts/python.exe -m bettersdr.app --no-settings
```

Settings and bookmarks live in `%APPDATA%/BetterSDR/`; recordings go to
`~/BetterSDR Recordings` as WAV, audio as 16-bit mono at 48 kHz and IQ as the
two-channel 8-bit file SDR#, HDSDR and GNU Radio all read back.

`listen.py` is the Phase 1 acceptance test in runnable form and has no Qt in it, so
audio faults and UI faults stay distinguishable. Its closing summary reports capture
percentage, audio underruns and ring overruns — if any of those are non-zero, the
threading is wrong, not the DSP.

Environment is a plain venv at `.venv`, installed with `pip install -e ".[dev]"`. Verified working on **Python 3.14.5** with numpy 2.5.2, scipy 1.18.1, PySide6 6.11.2, pyqtgraph 0.14.0, sounddevice 0.5.6.

## Hardware and driver facts

These were established by inspecting the actual DLL and the live device. **Do not re-derive them; several contradict what the documentation and forums say.**

- **`rtlsdr_set_dithering` is NOT exported** by the RTL-SDR Blog Windows release V1.4.0, despite existing in the fork's source. It cannot be used to detect the Blog fork. Use **`rtlsdr_check_dongle_model`** instead — that is the marker `native.py` relies on.
- **The bundled DLL set is only three files.** `rtlsdr.dll`, `pthreadVC2.dll`, `msvcr100.dll`. There is no `libusb-1.0.dll` — libusb is statically linked into this build.
- **Load the DLL by absolute path.** The single most common V4 failure is picking up the stock Osmocom `rtlsdr.dll` from somewhere else on the system: it mis-detects the R828D tuner and produces garbage that looks like a hardware fault. `native.load()` preloads the two dependencies by absolute path first, so the loader never searches. This is also why we do **not** use `pyrtlsdr` — it resolves the DLL through the system search path.
- **The dongle is a composite USB device**: a `usbccgp` parent plus `&MI_00` and `&MI_01` interfaces. The SDR driver must bind to **interface 0**. Interface 1 will not work and is a classic user error worth guarding against explicitly.
- **V4 identification is the tuner type**: `R828D` (enum 6) means V4. `R820T` (enum 5) means V3 or a clone.
- **HF needs no special handling.** The V4 has a built-in SA612 upconverter, not a direct-sampling hack. Tune anywhere in 500 kHz–28.8 MHz and the Blog driver handles the offset transparently — no mode switch, no Nyquist folding, full gain control. `Device.uses_upconverter` is informational only.
- **`rtlsdr_read_sync` requires a multiple of 512 bytes.** 16384 is the practical block size; smaller costs throughput.
- **`rtlsdr_set_freq_correction` returns -2 when set to the value already in effect.** `Device.freq_correction_ppm` guards against this — it is not an error.
- Usable range is 500 kHz – 1.766 GHz. 8-bit ADC. **2.4 MS/s** is the highest rate the RTL2832U sustains without dropping samples.

## DSP facts

Measured on this machine, not estimated. Same rule as the hardware section: don't
re-derive them.

- **Demodulating on the reader thread costs ~11% of the sample stream.** Read-only
  sustains 99.5% of real time; interleaving WFM demodulation into the same loop
  drops it to 89%, because while DSP runs no USB transfer is in flight and the
  dongle's data is simply lost. This is *the* reason `core/reader.py` exists, and it
  is invisible in any test that does not compare captured seconds to wall seconds.
  The symptom is a steady trickle of audio underruns, not distortion.
- **USB read size drives capture percentage.** Measured over 10 s each: 16 KB →
  99.49%, 64 KB → 99.72%, 256 KB → 100%. Every call has a fixed cost with no
  transfer in flight. `Reader` therefore defaults to **128 KB**, not the 16 KB that
  `rtlsdr_read_sync` nominally prefers. Anything under 100% silently drains the
  audio buffer while reporting zero ring overruns — a 4-minute soak at 99.8% gave
  305 audio underruns with no other symptom.
- **Capture below 100% must be compensated, not just minimised.** The dongle and
  the sound card run off different crystals, so buffering alone only postpones
  starvation. `audio.output.ClockSync` uses buffer depth as a control signal and
  resamples each block by at most 0.5% (under a tenth of a semitone, inaudible).
  Without it, a 0.2% shortfall empties a 150 ms buffer in about 75 seconds.
- **Every demodulator runs at under 10% of one core** at 2.4 MS/s (WFM/NFM/AM ~7%,
  USB/CW ~9%), leaving ample headroom for FFT and waterfall work.
- **Long FIR filters need FFT convolution.** `FirDecimator` switches to
  `oaconvolve` above 64 taps per output sample. The narrow SSB and CW filters run
  hundreds of taps at a decimation factor of 1, where it is worth ~6.5x — CW went
  from 603 ms to 93 ms per second of radio. The wideband decimating stages stay on
  the direct polyphase path, which is faster for them.
- **Every streaming stage must be tested block-by-block against one-shot.** A stage
  that forgets its history at a block boundary ticks once per block, which is
  inaudible in a single-buffer test and obvious on air.
- **The FM band overloads the front end at max gain.** A test capture at 100 MHz
  showed 1.64% of samples clipping. `listen.choose_gain()` steps down from maximum
  and takes the first setting below -12 dBFS with essentially no clipping; this
  belongs in a shared front-end module once the scanner needs it too.
- **Sample-rate constraint:** `dsp/demod.py` requires the sample rate to be a whole
  multiple of the 48 kHz audio rate. 2.4 MS/s gives exactly 50. Other rates raise
  rather than silently resampling.

## HD Radio facts

Measured off air on this machine on 2026-08-27, same rule as the other fact
sections: don't re-derive them. Full reasoning in **Amendment 3** of docs/PLAN.md.

- **Seven of the eight strongest local FM stations carry hybrid IBOC.** This is
  not a fringe feature locally; there is 20 dB of sideband margin on 94.9.
- **The signature is a flat plateau 129-198 kHz either side of the analog
  carrier**, 12-18 dB below it, dropping to the noise floor past 220 kHz.
  Flatness measured 0.4-1.3 dB of sub-band spread. That flatness is what
  separates OFDM from the sloping skirt of an over-deviating analog station.
- **Measure flatness on sub-bands, never on raw bins.** A single periodogram
  bin fluctuates ~5.6 dB whatever the signal, so a per-bin standard deviation
  mostly reports how many frames the caller averaged.
- **Symmetry is the guard that matters.** A neighbour on the adjacent channel
  lands exactly where a sideband would; requiring both sides within 10 dB of
  each other is what rejects it. 106.1 MHz is the local example.
- **`occupied_bandwidth_hz` will not find HD.** The digital part is ~15 dB
  down, so 99%-of-power sees only the analog core and still reports ~160 kHz.
  HD detection has to look at the shoulders specifically.
- **Decoding needs 1,488,375 S/s**, which is not a whole multiple of 48 kHz
  (31.0078), so it cannot use the `demod.py` skeleton. The decision is to pipe
  IQ to **nrsc5 as a child process** rather than reimplement NRSC-5 - the HDC
  codec is proprietary with no public spec, and the subprocess boundary keeps
  its GPL lineage from dictating terms for the whole app.

## HF and sample-rate facts

Measured off air on this machine on 2026-08-27, prompted by "no audio on the
AM band". Same rule as the other fact sections: don't re-derive them.

- **A 2.4 MHz window on the AM band is the bug, not the antenna.** The V4's
  SA612 upconverter leaks its local oscillator at 0 Hz on the dial. Tuned to
  710 kHz the window spans -490 to +1910 kHz, so the leak is *inside* it at
  **-15 dBFS, 65 dB above the noise floor**. `choose_gain` correctly drops to
  1.4 dB to keep the 8-bit ADC out of the rails, and the wanted station goes
  down with it. Max gain is worse, not better: the leak clips and KIRO's SNR
  falls from 8.6 to 3.3 dB.
- **The cure is a narrower window, and it is worth ~35 dB of audio.** At
  240 kS/s the leak is outside the window entirely, gain rises to 28-37 dB,
  and demodulated audio went from -64.6 to -28.7 dBFS on 710 kHz and -56.0 to
  -24.1 dBFS on 1000 kHz. `frontend.safe_sample_rate` is the guard; it only
  ever narrows, and the band plan's `sample_rate_hz` is where a band states a
  preference. Only rates that are both legal for the RTL2832U and a whole
  multiple of 48 kHz exist - see `SUPPORTED_SAMPLE_RATES`.
- **Block sizes must be defined in time, not bytes.** 128 KB is 27 ms at
  2.4 MS/s but **273 ms at 240 kS/s**, longer than the whole 150 ms jitter
  buffer, so every read arrived after the sink had run dry. `read_bytes_for`
  and `dsp_block_bytes_for` both derive from a duration and reproduce the
  measured 128 KB / 64 KB exactly at 2.4 MS/s.
- **A block can be smaller than one FFT frame.** At 240 kS/s a DSP block is
  3072 samples against a 4096-point FFT, so `Spectrum.process` returned empty
  and *the spectrum silently stopped updating* while audio carried on
  perfectly. The engine carries the remainder between blocks.
- **A rate change costs the audio buffer.** The ring is emptied, so the sink
  starves for as long as it takes to refill - 25 underruns for one hop from FM
  to AM. The sink is stopped and restarted across the change for the same
  reason `_begin_scan` does it: an underrun count full of expected underruns
  reports nothing.
- **The tune offset eats into the trusted window, and it matters at low
  rates.** The tuner sits 37 kHz off each tile centre, so a tile's lower edge
  is 37 kHz further out than its width suggests. At 2.4 MS/s that is 1.5% of
  the window and never binds; at 240 kS/s it is 15%, and the bottom 19 kHz of
  every tile was measured and then discarded for being outside `EDGE_GUARD`.
  **KIRO on 710 kHz landed exactly in that sliver and was missing from a scan
  of its own band.** `sweeper.usable_span` now takes the smaller of the two
  limits; FM broadcast is still 12 steps.
- **A steady carrier means opposite things in different bands.** An AM
  broadcaster radiates its carrier continuously and only the sidebands follow
  the programme, so a 50 ms dwell between words measures the carrier alone -
  which had the scanner reporting every Seattle AM station as "Unmodulated
  carrier". On the airband a channel is silent unless someone is speaking, so
  a steady carrier there really is interference. Hence `Band.continuous`:
  same measurement, opposite meaning, and only the band can tell them apart.
- **Timings:** AM broadcast 530-1700 kHz at 240 kS/s is **9 steps**, three
  passes in **4.9 s**, and finds ~23 stations indoors.

## Gain and tuning-range facts

Measured off air on this machine on 2026-08-27, prompted by "I cranked the RF
gain way up and now I can hear AM, but it's static-y" and "tuning below
0.5 MHz crashes the whole program". Same rule as the other fact sections:
don't re-derive them.

- **Gain was chosen once, in `Engine.start`, for whichever band the app
  opened on.** That is the FM band, the loudest thing the dongle ever sees,
  so the probe stepped down to **8-12 dB** - and tuning to the AM band kept
  it. Measured at 710 kHz, 240 kS/s: audio came out at **-59.8 dBFS** on the
  FM-band gain, **-48.3 dBFS** at the 20 dB a user reaches for by hand, and
  **-34.9 dBFS** at the 33.8 dB the probe picks for that band. Nothing was
  clipping at any of the three; it was simply 25 dB of signal left on the
  table. `Engine.auto_gain()` now re-measures on a band change, on a window
  change, and on arriving from a scan.
- **A gain probe is the one thing on the reader thread long enough to
  starve the audio.** It reads twice per gain setting, and on a strong band
  it walks most of the R828D's 29-entry table: **~340 ms with no capture, or
  twice the jitter buffer.** `_run` parks the sink whenever `_gain_pending`
  is set, the same reasoning as `_begin_scan` - expected underruns in the
  count are what stop it reporting real faults. With this, a scan, a card
  click, a hop to 710 kHz and a hop back measured **0 underruns end to end**;
  without it, 79.
- **The probe size is the 2.4 MS/s latent bug again.** 32 KB is 6.8 ms at
  2.4 MS/s and 137 ms at 240 kS/s, which across 29 settings is a four-second
  freeze. `frontend.probe_bytes_for` derives it from a duration and
  reproduces the measured 32 KB exactly at the full rate.
- **De-duplicate the probe.** A band change asks for one and the window
  change it triggers asks again; two back-to-back probes cost 23 underruns
  on a single hop to AM. `Engine._gain_pending` collapses them, and must
  clear in a `finally` or one failed probe wedges every later one.
- **Click-to-tune could ask for a frequency the dongle cannot reach.** At the
  bottom of the AM band the window legitimately extends below 500 kHz, so a
  click on the left of the plot sent 400 kHz to the radio. The digit tuner
  clamps; `_tune_from_display` was passing its own unclamped value straight
  past it. `frontend.safe_center_hz` is the guard, applied in `Engine.tune`
  because that is the one choke point every tuning path goes through.
- **A device command that raised anything but `RtlSdrError` killed the reader
  thread, silently.** `Device.center_freq` raises `ValueError` out of range,
  `_drain_commands` did not catch it, and the thread unwound leaving
  `errors` at 0 and `last_error` at None - so the app went deaf with a frozen
  display and nothing upstream could tell why. It now catches every
  exception: a rejected command is a diagnosable condition, never a reason to
  stop pumping samples.

## Scanning facts

Measured off air on this machine on 2026-08-27, with the antenna **indoors,
about three feet from an exterior wall**. That environment matters for
interpreting several of these. Same rule as the other fact sections: don't
re-derive them.

- **Per-frame DC removal must subtract the *window-weighted* mean.** Taking
  the plain mean nulls the unwindowed sum, not bin 0, and a strong signal off
  centre leaks into that mean - so removing it writes the leakage back as a
  real three-bin spike at DC, measured **25 dB above the noise floor** with one
  FM station 37 kHz away. The scanner duly reported it as a signal. This was a
  Phase 1 defect in `dsp/psd.py` that only showed up once something was
  searching the spectrum rather than looking at it.
- **The sweep parks the tuner 37 kHz off each tile centre.** DC removal blanks
  the centre bin, and the 25% step overlap does *not* rescue whatever is there:
  a signal at one step's centre is 1.8 MHz from its neighbours, outside their
  2.4 MHz windows entirely. The offset keeps the dead bin off every raster in
  the band plan (5, 10, 12.5, 25, 200 kHz).
- **The persistence gate's match tolerance has to be a few kHz, not tens.**
  At 586 Hz per bin, 25 kHz is an 85-bin window: in a quiet band most noise
  peaks find a partner near where some other one landed last pass, and the gate
  passes nearly everything. Tightening it to 4 kHz took an FM scan from 43
  entries to 36 with no real station lost. It is a *different* number from the
  merge tolerance used across overlapping steps, which stays at 25 kHz.
- **Spectral flatness cannot tell analog FM from digital.** Real FM carrying
  music measures 0.7-0.9, indistinguishable from OFDM. Only *synthetic*
  tone-modulated FM measures 0.45, which is what made 0.6 look like a safe
  threshold and had the app calling most of the local dial digital. Flatness is
  now used only where nothing is allocated. What actually identifies a digital
  signal in a known band is *where* the flat energy sits - the HD sideband test.
- **A 50 ms dwell measures instantaneous width, not occupied bandwidth.** An FM
  station caught between phrases collapses to a bare carrier: 98.1 MHz measured
  **11 kHz wide at 51 dB SNR**. Occupied bandwidth is bounded from below by
  modulation but not from above, so the sweeper keeps the *widest* view of each
  channel across passes. That also separates a real station from a spur, since
  a station widens when somebody talks and a spur never does.
- **Where a band's channels start is per-band data.** Deriving it as "half a
  raster in from the band edge" is right for FM broadcast and wrong for most of
  the rest: it put the NOAA station on 162.550 MHz on screen as 162.537. NOAA
  starts at the band edge, US FM at 88.1, US AM at 540 kHz with the band edge
  10 kHz below it. Hence `raster_base_hz`.
- **The best-view cache is keyed on the channel alone, never on channel plus
  label.** One transmitter classifies differently from pass to pass, and
  keying on both put 90.3 MHz on screen twice - once correctly, once as
  interference.
- **`Engine.scanning` reports what was *asked for*, not what the DSP thread has
  got round to.** Starting a scan queues a command, and a view polling at 20 Hz
  sees "not scanning" in the gap and concludes the sweep already finished.
- **An indoor aerial fills the airband with stable narrow carriers.** 83 of
  them, 2 kHz wide, 18-31 dB SNR, surviving every persistence pass - switching
  supplies and the dongle's own clock, not aircraft. They are real RF and the
  app reports them, but as "Unmodulated carrier", not as the band's traffic.
  A bare carrier far narrower than its channel is classified by shape even when
  the allocation is known.
- **Timings:** FM broadcast 88-108 MHz is **12 steps**, three passes in
  **5.1 s**. Weather Radio (150 kHz) is one step, 0.7 s. Nothing is reported
  until pass two completes, so a list appears about two thirds of the way in.
- **A sweep must take nothing from the band it was listening to.** Two things
  did. `start_scan` fell back to the *current* window when a band stated no
  preference, so scanning AM (240 kHz) and then listening to a station in it
  left FM broadcast planned at **141 steps instead of 12** - through a window
  narrower than one 200 kHz FM station, so every width and shape it measured
  was wrong as well as twelve times slower. The band's `sample_rate_hz`, or
  the 2.4 MS/s default, is the only input; `safe_sample_rate` still narrows it
  where the window would reach 0 Hz. Reproduced from the GUI in six clicks:
  scan AM, listen, back to Discover, FM Radio, Scan.
- **Gain was the second half of the same leak, and there is no probe anywhere
  in the sweep path.** A scan retunes 20 MHz away and keeps whatever the last
  station needed: arriving from AM that is ~34 dB into a band `choose_gain`
  measures at 8-12 dB, and a clipped 8-bit front end manufactures spurs the
  detector reports as stations. `_probe_scan_gain` measures at the *first
  step's* frequency - not through `auto_gain`, which probes wherever the tuner
  is parked and de-duplicates against `_gain_pending` - and `_end_scan`
  re-measures once for the frequency it returns to. The sizes above are from
  the existing gain measurements in this file; the sweep-time effect has not
  been re-measured off air.

## SDR# parity facts

Measured on this machine on 2026-08-28, while building Phase 3. Same rule as
the other fact sections: don't re-derive them.

- **The volume control and the limiter had to move out of the demodulator.**
  An AGC placed behind a fixed attenuator spends its range undoing it, and one
  behind a limiter cannot recover what the limiter already flattened. So the
  engine builds every demodulator at `volume=1.0` with `clip=False` and
  `dsp/chain.AudioChain` owns the tail of the path: noise reduction, the audio
  band-pass, the AGC, then volume, then the clip. `listen.py` still uses the
  demodulator's own volume, which is why the flag exists rather than the
  behaviour simply being removed.
- **IF noise reduction belongs after the channel filter, not on the raw
  stream.** Measured at **33% of one core** on the full 2.4 MS/s window against
  **~3%** at a 240 kHz IF, and the cost barely moves with FFT size because it
  is per-sample work, not per-transform. It is also the right answer
  acoustically: noise outside the channel is about to be discarded anyway.
  `Demodulator._front` is the hook, and it carries the remainder - an
  overlap-add stage returns whole hops, and the audio decimator after it
  insists on a multiple of its own factor.
- **Every noise-reduction time constant has to be stated in real time.** The
  frame rate is 187 per second at 48 kHz and 9,375 at 2.4 MS/s, so a "per
  frame" rise rate is fifty times faster on the wide window and the noise
  tracker simply follows the signal, reducing nothing. This is the
  2.4 MS/s-sizing bug again in a new place.
- **Spectral subtraction removes a steady tone, because a steady tone is
  indistinguishable from a noise floor.** On gated speech-like audio it cuts
  hiss by 4-6 dB and costs the speech 0.15 dB; on a permanently steady tone it
  removes the tone. That makes it wrong for CW and it is why the calibration
  assistant does not use it.
- **A noise blanker must detect and suppress at different levels.** Clipping to
  the same threshold that detected leaves the impulse at several times the
  signal around it, and makes the blanking window pointless - by definition no
  other sample in it crossed the threshold. Detect above `threshold x average`,
  suppress down to the average.
- **The blanker's running average must be seeded from the first block.**
  Starting it at zero made every sample of the first millisecond look like an
  impulse: **77 samples of real signal blanked** before the filter had climbed
  to the signal.
- **CPU, measured per second of radio:** noise blanker 19 ms (1.9% of a core)
  at 2.4 MS/s, frequency shifter 59 ms, DC removal plus IQ balance 64 ms,
  audio noise reduction 4.7 ms per second of audio. All are opt-in and all
  short-circuit to a single boolean test when off.
- **The V4 barely needs IQ correction.** Measured off air: **+0.07 degrees and
  +0.02 dB** of quadrature imbalance, and a DC offset of 0.004. The correction
  is worth shipping because a recording or another dongle may need it, not
  because this one does.
- **Raw IQ is 4.8 MB per second** at 2.4 MS/s - 17 GB an hour - so a recording
  size cap and a disk-space guard are not theoretical. It is written from the
  ring buffer, ahead of every correction, because a capture is meant to be
  replayable through software that does not exist yet.
- **`rtlsdr_set_dithering` is still not exported**, so the dithering control
  on the SDR# parity list cannot ship on this driver. The rest of that list
  ships; the few small gaps and the paths that could not be tested against
  real air are itemised under Phase 3 in docs/PLAN.md.
- **Parking the audio sink used to cost 150 ms of latency every time.**
  `AudioSink.stop` left the queue in place and `start` primed a fresh target
  buffer on top of it, so one gain probe took the buffer from **190 ms to
  369 ms** and it never came back; a few probes later it sat against the
  400 ms cap discarding blocks. `stop` now flushes, unconditionally - including
  when no stream is open, so a caller cannot end up with a primed buffer it
  believes it discarded. Latency now stays at 190-220 ms across probes, rate
  changes and sweeps, with no dropped blocks.

## PPM calibration facts

Measured off air on 2026-08-28. These decide whether the feature is useful or
actively misleading, and none of them were obvious beforehand.

- **A wideband FM broadcast station is a useless calibration reference.** Its
  energy is spread over 150 kHz by the modulation and the strongest bin wanders
  with the programme: six consecutive readings at 94.9 MHz had a standard
  deviation of **1814 Hz**, which is 19 ppm of random number. The same six on
  NOAA weather radio at 162.55 MHz spread by **11.5 Hz** - 0.07 ppm. So
  `calibrate()` measures the capture in four segments, reports the median, and
  **refuses to answer at all** when they disagree by more than 1 ppm of the
  carrier. The dialog asks for a weather-radio or AM broadcast carrier and
  says explicitly that an FM music station will not work.
- **The sign was checked on hardware, not reasoned about**, because it depends
  on librtlsdr's internals. Forcing +50 ppm at 162.55 MHz moved the measured
  offset by **+8246 Hz** against the +8128 Hz that +50 ppm of that frequency
  comes to, so `offset(ppm) = offset(0) + ppm * carrier * 1e-6`: raising the
  correction moves the carrier *up* the window. Planting +20 ppm and -15 ppm
  and running the assistant recovered to within 35 Hz in one step from both
  directions.
- **The correction only takes effect on the next retune**, so `set_ppm`
  reprograms the centre frequency straight afterwards. Without that the
  calibration appears to do nothing until the user moves the dial.
- **This dongle is already accurate to 0.24 ppm**, which is the V4's TCXO
  doing its job. The assistant correctly reports 0 ppm on it. Expect the
  feature to matter for V3s and clones rather than here.
- **Peak strength is measured in the transform, not the time domain.** There is
  about 39 dB of processing gain in a 16,384-point transform, so a carrier
  30 dB *below* the noise still measures exactly. The trust threshold is
  therefore set against bare noise, which peaks 4.0-4.8 dB above its own
  median, rather than against any signal-to-noise ratio.

## RDS facts

Measured off air on this machine on 2026-08-28, same rule as the other fact
sections: don't re-derive them. Full reasoning in **Amendment 7** of
docs/PLAN.md.

- **The multiplex arrives 98 ppm off nominal, not 0.24 ppm.** The dongle's
  TCXO is that good, but what reaches `decode/rds.py` is the *station's*
  baseband resampled by the *dongle's* clock, and the ratio measured 98 ppm
  on three local stations. That is 5.6 Hz on the 57 kHz subcarrier - 28
  degrees of constellation rotation across one DSP block - and a whole
  symbol of timing slip every eight seconds. Tracking phase alone left 21%
  of blocks failing their checkword and, tellingly, got *worse* the larger
  the DSP block; tracking a rate as well took the same three stations to
  **0.97, 0.94 and 0.71**. Assuming a clock is accurate because the datasheet
  says so is the mistake here.
- **An early-late timing detector does not work on biphase.** A biphase
  symbol correlates almost as strongly against its own inverse half a symbol
  away, so the loop has two places to sit and only one is right. Reading the
  matched filter at sixteen positions across the symbol and taking the
  largest has one maximum by construction; the peak stands out by a factor of
  four to seven even over the sixteen symbols one block holds.
- **A station's eight-character name is not necessarily its name.** American
  stations scroll song titles through it. 94.9 read " on KUOW" and "NPR's He"
  in alternate frames, so confirming each character separately - which sounds
  equivalent - produced "ren KUow". Frames are accepted whole, in order, and
  the name is only trusted as a name when two arrive identical; otherwise the
  callsign is the better answer.
- **The PI code needs confirming too.** A corrupt block passes its checkword
  about once in a thousand, which over a minute of a weak station is a near
  certainty - one of them labelled 102.5 MHz with a Los Angeles callsign.
- **The callsign arithmetic is right, and a station can still contradict it.**
  KUOW (0x4652) and KING (0x2678) both decode correctly. 102.5 MHz transmits
  0x137A, which is KBIG, exactly 0x4000 away from the code its own callsign
  implies. That is the station's data, not our arithmetic; codes outside the
  two arithmetic ranges get no callsign at all rather than a guess.
- **The whole decoder costs 2.4% of a core** and only runs on broadcast FM
  with a channel filter wide enough to still contain the subcarrier, so it is
  on by default.
- **The multiplex only exists between the discriminator and the de-emphasis.**
  `_FmBase.mpx_sink` is the tap. Anywhere later, the audio filter has already
  removed everything above 15 kHz, subcarriers included.

## FM stereo facts

Measured on this machine on 2026-08-28. Same rule as the other fact sections:
don't re-derive them. Full reasoning in **Amendment 8** of docs/PLAN.md.

- **The 38 kHz subcarrier is suppressed, so a phase error deletes the
  difference channel rather than distorting it.** What comes back is scaled
  by the cosine of the error, so 90 degrees out is a clean mono broadcast
  with nothing to notice. `dsp/stereo.py` squares an *analytic* pilot - a
  complex bandpass keeps only positive frequencies, so squaring doubles the
  frequency and carries the phase with it. No oscillator, no loop to tune.
- **The sign of that square is not a detail.** A sine's analytic form is
  `-j.e^{jwt}`, so squaring gives `-e^{2jwt}` and the imaginary part is
  `-sin(2wt)`. Left in, every broadcast plays with its channels swapped -
  and that measures as *perfect* separation on any test that only asks how
  different the two channels are.
- **The pilot filter's group delay is a phase error, not a delay.** 289 taps
  at 240 kHz is 144 samples, which is eleven cycles at 19 kHz. The multiplex
  is held back by exactly that much, which is why `StereoDecoder.process`
  returns the **sum as well as** the difference: a caller using its own
  undelayed sum would put the two ears 0.6 ms apart. That is the structural
  difference from the RDS receiver, which is a passive tap.
- **A mono station is not silent at 19 kHz, it is noisy there.** Detection is
  the ratio of the pilot band to a guard band at 16.8 kHz - above the audio,
  below the pilot, allocated on no station. Five local stations measured
  13.8-27.6 dB; a mono broadcast measures about 0. The threshold is not
  delicate.
- **Whether the difference channel is programme or noise is a question of
  shape, not level.** 88.5 MHz reads L-R only 0.2 dB below L+R with the
  channels uncorrelated, which is either very wide stereo or noise. The
  difference channel's roll-off between 0.3-3 kHz and 8-15 kHz settles it:
  27.5 dB against the sum's 24.8 dB means programme. The weakest station
  tried, 96.5, reads 18.6 against 24.3 - noise starting to show, and the
  station a stereo blend would eventually be for.
- **Audio is now mono `(frames,)` or stereo `(frames, 2)` everywhere past
  the demodulator.** `lfilter` defaults to the trailing axis, which on a
  stereo block filters the left channel against the right; and two
  independent AGCs are a stereo image that wanders. Every stage takes a
  frame axis and shares one control signal across channels.
- **`AudioSink` opens two channels whatever the radio is doing** and conforms
  each block on the way in. The pilot comes and goes several times a minute
  on a marginal station, and reopening the stream at each transition would
  put a gap in the audio every time.
- **Spectral noise reduction is the one stage that mixes down**, because two
  independent noise estimates pull the image apart. `AudioChain.keeps_stereo`
  reports it, and the badge is lit from what reached the sound card rather
  than from the pilot - so switching it on visibly turns stereo off instead
  of leaving a badge that quietly stopped being true.
- **The whole decoder costs 2.0% of a core** (20 ms per second of radio at
  2.4 MS/s, against the demodulator's 63 ms), and is on by default.

## Architecture

```
bettersdr/
  app.py          GUI entry point
  listen.py       headless tune-and-play CLI; Phase 1 acceptance test
  core/
    native.py     ctypes bindings; absolute-path DLL load; fork detection
    device.py     Device class + `--info` CLI
    doctor.py     driver diagnosis via cfgmgr32 (no UI, no PowerShell)
    ringbuffer.py preallocated SPSC byte ring; drops oldest, never blocks
    reader.py     the reader thread + its device-command queue
    frontend.py   gain selection, safe_sample_rate, safe_center_hz
    engine.py     the DSP thread, device ownership, single-slot mailbox
    settings.py   persisted preferences, atomic JSON, defaults on any fault
    bookmarks.py  named frequencies with groups and CSV exchange (no Qt)
    calibrate.py  ppm measurement against a known carrier
  dsp/
    convert.py    uint8 -> complex64 LUT
    filters.py    streaming FIR decimation, de-emphasis, discriminator,
                  squelch, audio band-pass
    demod.py      wfm/nfm/am/usb/lsb/cw/dsb/raw, one shared 3-stage skeleton
    psd.py        Welch PSD in dBFS, peak hold, noise floor, occupied bandwidth
    features.py   classifier features; HD Radio sideband detection
    agc.py        audio AGC: threshold, slope, hang, separate ramps
    denoise.py    noise blanker (impulses) + spectral subtraction (hiss)
    correct.py    DC removal, IQ imbalance, swap I/Q, offset tuning
    stereo.py     the 19 kHz pilot and the 38 kHz L-R subcarrier
    chain.py      FrontEnd and AudioChain: the optional stages either side
                  of the demodulator
  scan/
    bandplan/     us.yaml + loader; feeds the ribbon, the classifier and the
                  Discover band chips (`scan: true`)
    detector.py   shaped noise floor, thresholding, grouping, persistence gate
    classifier.py band plan + shape features -> a labelled, explained Signal
    sweeper.py    step planning, the scan state machine, stitching
  decode/
    rds.py        the 57 kHz subcarrier: BPSK, block sync, station name,
                  radio text, PI code and the US callsign it encodes
                  (adsb, pocsag still to come)
  audio/
    output.py     jitter buffer + clock drift sync
    record.py     audio WAV and byte-exact baseband IQ WAV, with limits
  ui/
    app shell     main_window.py, levels.py, listen_view.py, discover_view.py
    freq_manager.py  the bookmark window
    widgets/      spectrum.py, waterfall.py, frequency.py, meter.py,
                  colormaps.py, axes.py, signalcard.py, icons.py,
                  panel.py (the sectioned, level-gated control column)
drivers/win-x64/  bundled RTL-SDR Blog driver V1.4.0 (committed on purpose)
tests/
  synth.py        synthetic IQ generator — most tests need no hardware
```

### Threading model

Three threads, one direction of data flow, no locks on the hot path:

1. **Reader thread** — `rtlsdr_read_sync()` in a loop into a preallocated ring buffer. We use `read_sync` rather than `read_async` because it releases the GIL inside the C call and lets the scanner retune synchronously between reads, so one mechanism serves both listening and scanning.
2. **DSP thread** — consumes blocks; writes audio into the sounddevice jitter buffer and display frames into a single-slot mailbox. **Scanning runs on this same thread**, taking turns with listening rather than adding a consumer: two things draining one ring would each get half the samples. Audio stops for the length of a sweep, so scan-time starvation cannot pollute the underrun count that reports real faults.
3. **Qt GUI thread** — a 30 Hz timer reads the latest mailbox value. Never blocks, never touches the device, drops frames rather than queueing them.

Device control calls are serialised through a command queue consumed by the reader thread between reads. **Never call into `Device` from the GUI thread.**

## Conventions

- **`Device` is not thread-safe by design.** The reader thread owns it. Do not add locks; use `Reader.submit()` / `.tune()` / `.set_gain()`, which run the call between reads. `tests/test_reader.py` asserts that only the `sdr-reader` thread ever touches the device.
- **DSP stages are stateful and block-oriented.** Anything with history carries it across calls and exposes `reset()`. Demodulators buffer whatever does not divide evenly into their decimation chain, so USB read sizes stay independent of DSP arithmetic.
- **No Python loops over samples.** Everything in `dsp/` is vectorised NumPy. The conversion path runs on every sample the app ever sees.
- **Diagnostics are pure logic.** `doctor.py` has no UI imports so both the CLI and the wizard share it. Keep it that way.
- **User-facing strings avoid jargon.** `doctor.py` remedies are written for someone who has never heard of a USB driver. No dB values, no "demodulator", no acronyms in Simple mode.
- **Tests use synthetic IQ.** Plant a signal with known parameters, assert the pipeline recovers them. Hardware tests are a separate manual checklist in docs/PLAN.md — never in the automated suite.
- **PSD levels are calibrated dBFS.** A full-scale tone reads 0 dB whatever the FFT size or window, so a detection threshold means the same thing at every display setting. `tests/test_psd.py` asserts this across all of both. The scanner and the display share `dsp/psd.py` deliberately: if the detector measured the band differently from the picture, "the app found a signal I can't see" becomes possible, and explainability is the product.
- **The GUI owns no threads and never touches `Device`.** `ui/` reads
  `Engine.latest()` on a 30 Hz timer and nothing else. Anything that needs the
  radio goes through `Engine`, which serialises it onto the right thread.
- **The mailbox drops frames; it does not queue them.** A GUI that stalls
  should resume with the current picture, not work through a stale backlog.
- **The band plan is data.** `scan/bandplan/us.yaml` drives the ribbon, the
  auto mode/bandwidth on tuning, and later the classifier. A second region is a
  second file, not a second code path.
- **Tuning into a new band adopts that band's mode and bandwidth**, but only on
  a change of band, so a deliberate choice survives retuning within one. In
  Simple mode there is no mode control at all, so without this an AM airband
  transmission would be demodulated as wideband FM and the app would look
  broken.
- **The three stacked panes share `AXIS_WIDTH`.** A waterfall offset from the
  spectrum above it puts every frequency wrong by a few hundred kHz.
- **Widget logic that can be pure, is.** Colour maps, the digit arithmetic and
  the waterfall ring are plain functions tested without a window.
- **Discover is the landing screen at every level.** Opening on a spectrum is
  what every other SDR application does; being shown a list of what is actually
  out there is the argument this one is making.
- **The classifier says why, and says when it is unsure.** Every `Signal`
  carries `reasons`; a card with `certain` false is badged BEST GUESS. "Unknown
  signal" and "Unmodulated carrier" are valid, non-embarrassing answers, and a
  confident wrong one costs trust in every other line on the screen.
- **Never set a stylesheet on a `QScrollArea` viewport.** It drags every
  descendant through the stylesheet style, and `QLabel` is a `QFrame`, so each
  one starts painting a frame it never asked for. Use a palette, or name the
  content widget.
- **Anything sized for 2.4 MS/s is a latent bug.** Buffer sizes, block sizes
  and step plans get written against the default rate and quietly break when
  the window narrows. Express them as a duration or a fraction of the rate,
  and check the answer at 240 kS/s as well as at 2.4 MS/s.
- **Gain belongs to the band, not to the session.** How loud a band is has
  nothing to do with how loud the last one was, and the difference measures
  30 dB between FM broadcast and AM. Any moment the front end is pointed at
  something materially different - a new band, a new window, a card clicked
  in Discover - re-measures. `Engine.auto_gain` de-duplicates, so callers may
  ask freely.
- **Anything that runs on the reader thread stops the radio.** A gain probe
  is 340 ms of it. Park the audio sink around such work rather than let the
  underrun count absorb it, and size the work in time so it does not grow
  tenfold when the window narrows.
- **The window width is a correctness setting, not a display preference.**
  `frontend.safe_sample_rate` guards every path that picks a frequency, and it
  only ever narrows - so a deliberate choice survives unless it would put the
  upconverter's oscillator on screen instead of a station.
- **Audio may have two channels.** Anything downstream of a demodulator
  takes `(frames,)` or `(frames, channels)` and works along the frame axis.
  A stage that derives a control signal - a gain, a gate, a noise estimate -
  derives **one** from all the channels and applies it to all of them, or
  else mixes down and says so. Independent per-channel control is a stereo
  image that wanders, which is a stranger fault than any level error.
- **The audio chain owns the end of the path, not the demodulator.** Volume
  and the final limiter come after the AGC, so a gain rider has the full range
  to work with and the volume slider still does something when it is on. Every
  demodulator the engine builds gets `volume=1.0` and `clip=False`.
- **An optional stage costs one boolean test when it is off.** `FrontEnd` and
  `AudioChain` return the block they were handed - the same object - when
  nothing is enabled, so the whole Phase 3 feature set is free on a default
  install. Anything added there must keep that property.
- **Nothing user-facing may leave the radio somewhere a beginner cannot get
  back from.** The bias tee asks first and is never restored from settings; the
  window width is clamped by `safe_sample_rate` whatever is remembered; a
  corrupt settings file is replaced by defaults rather than reported.
- **A measurement that cannot be trusted is not reported.** The calibration
  assistant refuses rather than returning a confident number derived from a
  wandering reference - same principle as the classifier's "Unknown signal".
- **Recording is written from the ring, not from the audio path.** The clock
  drift correction resamples by up to 0.5%, which is right for listening and
  wrong for anything anybody might later measure.
- Line length 90, ruff with `E,F,I,UP,B,SIM`. Keep `ruff check .` clean.

## Git

`git init` has been run. **Nothing has been committed yet** — the user has not asked for a commit. Ask before committing.

The DLLs in `drivers/win-x64/` are committed deliberately; bundling them is the whole point of the absolute-path load strategy.
