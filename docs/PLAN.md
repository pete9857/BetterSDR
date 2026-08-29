# BetterSDR — Beginner-Friendly SDR for the RTL-SDR Blog V4

## Context

The RTL-SDR Blog V4 is a capable receiver, but every mainstream app for it (SDR#, SDR++, HDSDR) is built for people who already know RF. They open on a blank spectrum at some arbitrary frequency and expect you to already know what modulation to pick, what bandwidth to set, what gain means, and which frequencies are worth visiting. Auto-scanning — arguably the most obvious thing a newcomer wants — requires third-party plugins.

BetterSDR inverts the default. It opens, sweeps the air around you, and shows a **list of what it found in plain English**, the way a Wi-Fi picker shows networks. The mental model is "scan and browse", not "tune and configure".

Critically, **beginner-friendly must not mean capability-capped**. If a user outgrows BetterSDR they'll leave for SDR#, and the project fails at its real goal — bringing people *into* the hobby. So the app ships full SDR#-class capability, revealed progressively rather than removed. Nothing is missing; things are just quiet until you ask for them.

**Target outcome:** someone who has never heard the word "demodulator" plugs in the dongle, clicks Scan, sees `🎵 FM Radio — 98.5 MHz — Strong`, clicks Listen, and hears music. Under two minutes, no manual. Six months later, the same person is dragging filter edges and tweaking noise reduction in the same app.

## Decisions locked

| | |
|---|---|
| **Stack** | Python 3.14 + PySide6 6.11.2 + NumPy 2.5 / SciPy 1.18 + pyqtgraph 0.14 + sounddevice |
| **Region** | United States band plan (data-driven, so other regions are a data file later) |
| **Distribution** | PyInstaller one-folder portable build with the RTL-SDR Blog driver bundled |
| **Device access** | Our own thin `ctypes` wrapper — **not** `pyrtlsdr` |

**Why our own ctypes wrapper.** The single most common V4 failure is loading the wrong `rtlsdr.dll` — the stock Osmocom build mis-detects the R828D tuner and produces garbage. `pyrtlsdr` finds the DLL via the system search path, so we can't guarantee which one it loads. We bundle the RTL-SDR Blog fork's x64 DLL (release V1.4.0) and load it by **absolute path**, eliminating that entire class of bug. It also gets us `rtlsdr_set_bias_tee` and `rtlsdr_set_dithering`, which are blog-fork-only and not exposed by `pyrtlsdr`. The librtlsdr C API is ~25 functions; the wrapper is ~250 lines.

## Progressive disclosure — the organising principle

One app, three levels. The level control is a single segmented button in the toolbar; the DSP engine is identical at every level, only the visible controls change.

| Level | Who it's for | Shows |
|---|---|---|
| **Simple** *(default)* | Never used an SDR | Scan button, signal cards, Listen, volume. Mode/bandwidth/gain fully automatic. |
| **Standard** | Comfortable after a few sessions | Spectrum + waterfall, click-to-tune, mode selector, bandwidth, squelch, gain, step size, bookmarks, recording. |
| **Expert** | SDR# refugee | Everything below — FFT internals, filter design, NR/NB, AGC parameters, IQ correction, offset tuning, raw IQ capture. |

Level is remembered per user. Promoting a control from hidden to visible is a data change in the view definition, not a rewrite.

## Hardware facts this design depends on

- **HF is free.** The V4 has a built-in SA612 upconverter, not a direct-sampling hack. Tune anywhere from 500 kHz–28.8 MHz and the blog driver handles the offset transparently — no mode switch, no Nyquist folding, full gain control. AM broadcast and shortwave are just bands in the scan list.
- **Coverage:** 500 kHz – 1.766 GHz continuous. 8-bit ADC. 2.4 MS/s is the reliable ceiling (3.2 drops samples).
- **Tuner is R828D** — that's how we positively identify a V4 vs a V3 (R820T2) at runtime.
- **Dongle is currently unconfigured on this machine.** It should be plugged in but left on the stock Windows DVB-T driver: that "wrong driver" state is the Driver Doctor's most important code path and is awkward to recreate once Zadig has been run.

## Architecture

```
BetterSDR/
├── pyproject.toml
├── run.bat                       # venv bootstrap + launch
├── drivers/win-x64/              # bundled blog-fork rtlsdr.dll, libusb-1.0.dll, pthreadVC2.dll
├── bettersdr/
│   ├── app.py                    # entry point
│   ├── core/
│   │   ├── native.py             # ctypes librtlsdr bindings, absolute-path DLL load
│   │   ├── device.py             # Device: open/tune/gain/bias-tee, tuner identification
│   │   ├── reader.py             # reader thread -> RingBuffer (rtlsdr_read_sync)
│   │   ├── ringbuffer.py         # preallocated lock-free-ish IQ ring
│   │   └── settings.py           # persisted config, UI level, bookmarks
│   ├── dsp/
│   │   ├── convert.py            # uint8 -> complex64 via 256-entry LUT
│   │   ├── psd.py                # Welch PSD, DC-spike removal, sweep stitching
│   │   ├── demod.py              # WFM / NFM / AM / USB / LSB / CW / DSB / RAW
│   │   ├── filters.py            # decimation chains, de-emphasis, squelch, channel filters
│   │   ├── agc.py                # audio AGC: threshold, decay, slope, hang
│   │   ├── denoise.py            # noise blanker + IF/audio noise reduction
│   │   ├── correct.py            # DC removal, IQ imbalance, offset tuning
│   │   └── features.py           # envelope variance, spectral flatness, carrier ratio
│   ├── scan/
│   │   ├── sweeper.py            # band stepping state machine
│   │   ├── detector.py           # noise floor + thresholding + peak grouping
│   │   ├── classifier.py         # band plan + feature fusion -> labelled Signal
│   │   └── bandplan/us.yaml      # allocations, plain-English descriptions, icons
│   ├── decode/
│   │   ├── rds.py                # FM station names
│   │   ├── adsb.py               # 1090 MHz aircraft
│   │   └── pocsag.py             # pager text
│   ├── audio/
│   │   ├── output.py             # sounddevice callback + jitter buffer
│   │   └── record.py             # WAV audio + raw IQ capture
│   └── ui/
│       ├── main_window.py        # shell, view switching, level selector
│       ├── discover_view.py      # signal card list  <- the landing screen
│       ├── listen_view.py        # tuned playback + spectrum/waterfall
│       ├── explore_view.py       # wideband spectrum with band-plan overlay
│       ├── setup_wizard.py       # Driver Doctor + first-run
│       ├── freq_manager.py       # bookmarks / memory channels
│       └── widgets/              # signal card, spectrum, waterfall, digit tuner, meters
└── tests/
    ├── synth.py                  # synthetic IQ generator (no hardware needed)
    └── test_*.py
```

### Threading model

Three threads, one direction of data flow, no locks on the hot path:

1. **Reader thread** — `rtlsdr_read_sync()` in a loop, writing `uint8` IQ into a preallocated ring buffer. `read_sync` releases the GIL inside the C call and, unlike `read_async`, lets the sweeper retune synchronously between reads. One mechanism serves both listening and scanning.
2. **DSP thread** — consumes blocks; produces audio frames into the sounddevice ring and display frames (PSD row, level, demod state) into a single-slot mailbox.
3. **Qt GUI thread** — a 30 Hz timer reads the latest mailbox value. Never blocks, never touches the device. Drops frames rather than queueing them.

Device control calls are serialised through a command queue consumed by the reader thread between reads — never called from the GUI thread.

## Spectrum and waterfall

The centrepiece of Standard mode, and where SDR# parity matters most.

**Spectrum analyser**
- FFT size 512–32768 (Expert), default 4096. Window selectable: Hann, Blackman-Harris, Hamming, flat-top.
- Display smoothing (exponential averaging), adjustable.
- **Peak hold** trace with decay, and a max-hold reset.
- Draggable **passband overlay** — a shaded region showing the current filter bandwidth. Dragging its edges changes bandwidth directly, the iconic SDR# interaction.
- Frequency markers and a **band-plan ribbon** above the axis: coloured, labelled allocation blocks. This is simultaneously SDR# parity and the single best passive teaching device in the app.

**Waterfall**
- pyqtgraph `ImageItem` over a preallocated `(rows × bins) float32` ring. Scroll via a rolling write index, not `np.roll` — no per-frame copy of the whole history.
- Colour maps: SDR#-style classic, viridis, turbo, grayscale, inferno.
- Contrast and brightness as an explicit dB display range (min/max), plus a one-click **auto-range** that fits to the current noise floor and peak.
- Waterfall speed decoupled from FFT rate by row averaging, so a slow scroll doesn't mean a slow display.
- Adjustable spectrum/waterfall split ratio.

**Shared interactions:** click-to-tune, drag-to-pan, wheel-to-zoom, centre lock, snap-to-step. Zoom and pan stay linked between the two panes.

## SDR# feature parity checklist

Everything here ships. The Level column is where each control becomes visible.

| Area | Features | Level |
|---|---|---|
| **Demodulators** | WFM (+ stereo), NFM, AM, USB, LSB, CW, DSB, RAW | Standard |
| **Tuning** | Digit-wise scroll frequency display, click-to-tune, snap-to-step, configurable step size, VFO offset | Standard |
| **Filtering** | Adjustable bandwidth, draggable passband edges, filter type and order, filter audio | Standard / Expert |
| **Squelch** | Threshold with hysteresis and attack/release ramps | Standard |
| **Audio** | Output device select, volume, mute, sample rate, unity gain | Standard |
| **AGC** | Audio AGC with threshold, decay, slope, use-hang | Expert |
| **De-emphasis** | 50 µs / 75 µs / off | Expert |
| **Noise handling** | Noise blanker, IF noise reduction, audio noise reduction | Expert |
| **Radio front end** | RF gain (manual + auto), tuner AGC, sample rate select, PPM correction, bias tee, dithering | Standard / Expert |
| **IQ correction** | DC removal, IQ imbalance correction, swap I&Q, offset tuning | Expert |
| **Recording** | Audio WAV, baseband IQ WAV, with size/duration limits | Standard |
| **Frequency manager** | Named bookmarks, groups, import/export, auto-populated from scan results | Standard |
| **Display** | FFT size, window function, smoothing, peak hold, colour maps, contrast/range, split ratio | Standard / Expert |
| **Scanning** | Band sweep, signal detection, classification, auto-demod — **built in, not a plugin** | Simple |

The last row is the differentiator: in SDR# that capability requires assembling plugins. Here it's the default screen.

## Build phases

### Phase 0 — Device layer and Driver Doctor *(riskiest work first)*

Everything else is worthless if we can't reliably get samples, so this ships first and standalone.

- `core/native.py`: ctypes signatures for open/close, `set_center_freq`, `set_sample_rate`, `set_tuner_gain_mode`, `set_tuner_gain`, `get_tuner_gains`, `set_agc_mode`, `reset_buffer`, `read_sync`, `set_bias_tee`, `set_dithering`, `set_freq_correction`, `get_tuner_type`. Load the bundled DLL by absolute path via `ctypes.WinDLL` with the driver dir added to the DLL search path.
- **Driver Doctor** (`ui/setup_wizard.py`) — the piece that makes or breaks first-run:
  1. Enumerate USB for `VID_0BDA` / `PID_2838|2832`. Not found → "Is it plugged in?" with a re-check button.
  2. Found but device won't open → it's still bound to the Windows DVB-T driver. Show a **step-by-step Zadig walkthrough** naming the exact entry to select (`Bulk-In, Interface (Interface 0)`) and warning against Interface 1, with a link to Zadig (linked, not bundled — AV false positives) and a Re-check button.
  3. Opens → read tuner type. `R828D` ⇒ V4 confirmed. Verify the loaded DLL exports `rtlsdr_set_dithering`; if it doesn't, we've picked up a non-blog DLL — warn loudly with the fix.
  4. **Smoke test:** sweep 88–108, tune the strongest station, play two seconds. *"You should hear music."* Yes/No buttons. This converts "did I set it up right?" from a mystery into an answer.
- CLI harness `python -m bettersdr.core.device --info` for debugging without the GUI.

**Done when:** starting from the stock DVB-T driver state, the wizard walks through Zadig, identifies the dongle as a V4, and a scripted 2-second FM capture demodulates to audible audio.

### Phase 1 — Listen

- `dsp/convert.py`: LUT-based `uint8 → complex64` (256-entry table, fancy-index, `.view(complex64)`).
- `dsp/demod.py`:
  - **WFM** — polar discriminator `np.angle(x[1:] * np.conj(x[:-1]))`, decimate 2.4M→240k, 75 µs de-emphasis (one-pole IIR), →48k via `resample_poly`.
  - **NFM** — same discriminator, 12.5 kHz channel filter, no de-emphasis.
  - **AM** — `np.abs(x)` with a DC blocker.
  - **USB/LSB/CW/DSB** — frequency shift + Hilbert, sideband select; CW adds a BFO tone offset.
- Power squelch with hysteresis and attack/release ramps (a hard gate sounds broken; ramps sound intentional).
- `audio/output.py`: sounddevice callback pulling from a jitter buffer; underruns emit silence, never garbage.
- `ui/listen_view.py` with the spectrum + waterfall described above, large digit-wise frequency readout, and a signal-strength meter.

**Done when:** you can tune an FM station by hand and it sounds clean for 10 minutes with no dropouts.

### Phase 2 — Discovery *(the headline feature)*

- **`scan/sweeper.py`** — step the tuner in `0.75 × sample_rate` increments (25% overlap discarded to dodge filter roll-off and the RTL2832U DC spike). ~5 ms settling after each retune, discard the first block, then ~50 ms dwell. FM band = ~11 steps ≈ 0.6 s. Full 24–1766 MHz ≈ 970 steps ≈ 50 s.
- **`scan/detector.py`**
  - Noise floor via a wide-kernel median/percentile filter across frequency — it tracks the band's shape instead of assuming flat.
  - Threshold = floor + K dB. K is exposed as a three-position **Sensitivity** control in Simple mode, as dB in Expert.
  - Group contiguous over-threshold bins, merge sub-bin gaps, require ≥2 bins wide. Emit `Detection(center_hz, bw_hz, peak_dbfs, snr_db)`.
  - **Persistence gate:** a signal must appear in 2 of 3 consecutive sweeps before it's shown. Without this the list flickers with noise and the app feels untrustworthy.
- **`scan/bandplan/us.yaml`** — allocations as *data*: frequency range, friendly name, plain-English description, expected modulation, expected channel bandwidth, channel raster, icon. Ships with FM broadcast, AM broadcast, shortwave, airband, marine VHF, NOAA weather radio, 2 m / 70 cm ham, FRS/GMRS, ISM 433/915, POCSAG pagers, ADS-B 1090, NOAA APT 137. Also feeds the spectrum band ribbon.
- **`scan/classifier.py`** — rules, not ML. Four cheap features from `dsp/features.py`:
  - carrier ratio (peak bin / mean) → AM and CW have a strong carrier
  - envelope variance of `|x|` → FM is constant-envelope, so low
  - spectral flatness → digital modes look noise-like
  - 99% occupied bandwidth
  Fused with the band-plan match into a confidence score. **Rules are chosen deliberately over ML so the app can explain itself:** the UI shows *"Constant power, 150 kHz wide, sits in the 88–108 MHz broadcast band → FM radio station."* Explainability is the product here, not a debugging aid.
- **`ui/discover_view.py`** — the landing screen. Signal cards with icon, friendly name, frequency, strength bars, and a Listen button. Band filter chips across the top. A "What is this?" expander on each card with a plain-English paragraph from the band plan.
- Classification drives demod automatically: mode, bandwidth, and squelch come from the matched allocation, so Listen just works. Scan hits can be saved to the frequency manager in one click.

**Done when:** a cold scan of the FM band lists real stations at correct frequencies with no phantom entries, and clicking any card produces audio.

### Phase 3 — Full radio (SDR# parity)

Implements the rest of the parity checklist and wires up the Standard/Expert levels.

- `dsp/agc.py` — audio AGC with threshold, decay, slope, hang.
- `dsp/denoise.py` — noise blanker (impulse clipping on the IF) and spectral-subtraction noise reduction for IF and audio.
- `dsp/correct.py` — DC removal, IQ imbalance estimation, swap I&Q, offset tuning to move the DC spike out of the passband.
- `audio/record.py` — audio WAV and baseband IQ capture with size caps and a disk-space guard.
- `ui/freq_manager.py` — named bookmarks with groups, import/export, and one-click population from scan results.
- Full display controls: FFT size, window, smoothing, peak hold, colour maps, dB range, split ratio.
- Bias-tee toggle, guarded behind a confirmation (it puts 4.5 V on the antenna port and can damage equipment that isn't expecting it).
- PPM calibration assistant using a known-strong broadcast carrier.

**Done when:** an SDR# user can do everything they normally do without reaching for another app.

**Status: complete and verified on hardware, 2026-08-28.** Everything above
ships, plus `core/settings.py` and `core/bookmarks.py`, which the original
architecture listed but no earlier phase had needed. Two additions to the
plan as written: `dsp/chain.py` holds the optional stages either side of the
demodulator rather than leaving a dozen switches loose in the engine, and
`ui/widgets/panel.py` holds the control column, which went from eight controls
to about seventy-five at Expert.

Findings worth carrying forward are in **Amendment 6**.

#### Still open at the end of Phase 3

Small gaps against the parity checklist, none of them blocking:

| Item | State |
|---|---|
| **Dithering** | **Cannot ship.** `rtlsdr_set_dithering` is not exported by the RTL-SDR Blog Windows release — known since Phase 0. |
| **FM stereo** | Shipped in Phase 4. See the Phase 4 status block and Amendment 8. |
| **VFO / transverter offset** | Not done. This is the *display* offset for a transverter — show 144 MHz while tuned to 28 MHz — and is a different feature from the offset tuning that ships. A few lines in `FrequencyDisplay` plus one setting. |
| **Filter type** | Only windowed-sinc FIR. Filter *order* ships as "filter edge"; the type does not. Low value on 8-bit data. |
| **Squelch shape** | Hysteresis and the attack/release ramps are implemented but not exposed as controls, only the threshold. |
| **Audio sample rate** | Deliberately fixed at 48 kHz. `dsp/demod.py` requires the SDR rate to be a whole multiple of it, and making it adjustable would put a resampler in every chain to no audible benefit. Not a gap to close. |

#### Built but not exercised on hardware

Everything in the list below is covered by synthetic tests and was not
verifiable against real air in this session. Worth putting on the hardware
checklist rather than assuming.

- **The bias tee.** Deliberately not switched on: it puts 4.5 V down the
  antenna cable, and the user has equipment connected. The confirmation dialog
  and the `Device.set_bias_tee` path are both untested end to end.
- **Swap I/Q**, which needs a source with the opposite convention to show
  anything.
- **Choosing a different sound card** at run time.
- **The noise blanker against real impulse noise.** On the FM band it correctly
  blanked nothing, because there were no impulses to find.
- **The recording size and duration caps, and the disk-space guard**, on a
  recording long enough to reach them.
- **Import and export in the frequency manager**, which go through file
  dialogs.

### Phase 4 — Decoders and polish

- **RDS decoding** — 57 kHz subcarrier, BPSK 1187.5 bps, differential decode, block sync on offset words, group 0A/0B for the station name. This is the single biggest "wow": the discovery list stops saying `FM Radio — 98.5 MHz` and starts saying `KQED 88.5`. Worth the effort.

  **Status: complete and verified on hardware, 2026-08-28.** `decode/rds.py`
  ships the whole chain — subcarrier, symbol timing, carrier tracking, block
  sync, and groups 0A/0B and 2A/2B — feeding a station name, radio text, PTY,
  traffic flags, the PI code and the US callsign it encodes. It hangs off
  `_FmBase.mpx_sink`, is on by default, costs 2.4% of a core, and detaches
  itself on any mode or bandwidth that cannot carry the subcarrier. Measured
  block quality on three local stations: **0.97, 0.94, 0.71**. Findings are in
  **Amendment 7**.

  Two things it deliberately does not do. It does not use the checkword's
  error-correcting capability, because a mis-corrected block puts plausible
  wrong text on screen. And it does not run during a sweep — a 50 ms dwell is
  a twentieth of one group — so the Discover list still names stations from
  the band plan, and only the listen screen reads them off the air.
- **FM stereo** — 38 kHz pilot-locked L−R subcarrier.

  **Status: complete and verified on hardware, 2026-08-28.** `dsp/stereo.py`
  recovers the subcarrier by squaring an analytic 19 kHz pilot, holds the
  multiplex back by the pilot filter's own group delay so the reference and
  the signal refer to the same instant, and hands `_FmBase` a matched sum and
  difference to matrix into L and R. Detection is a pilot-to-guard-band ratio
  rather than a level, so a mono station is not mistaken for a weak stereo
  one. It costs 2.0% of a core, is on by default, and detaches on any mode or
  bandwidth that cannot carry the difference channel. Measured **61 dB** of
  separation on a synthetic broadcast and locked on every local station
  tried, at 14–27 dB of pilot margin. Findings are in **Amendment 8**.

  The audio path became mono-or-stereo to carry it: `AudioSink` now opens two
  channels and conforms whatever it is handed, `ClockSync` stretches both
  channels onto one grid, `Squelch`, `BiquadState` and `Agc` take a frame axis
  and share one control signal across channels, and `AudioRecorder` writes
  either. Audio noise reduction is the one stage that mixes down, and it says
  so — see the amendment.
- **ADS-B (1090 MHz)** → aircraft callsign list.

  **Status: complete and verified on hardware, 2026-08-28.**
  `decode/adsb.py` takes complex IQ at 2.4 MS/s and produces a list of
  aircraft with callsign, altitude, position, ground speed, track and climb
  rate: preamble search on a half-microsecond energy grid, pulse-position
  slicing from a running sum, CRC-24, DF17/18 extended squitters and DF11
  all-call replies, and global plus local CPR position decoding. It costs
  5.6% of a core.

  `Engine.start_adsb` takes the radio to 1090 MHz at the full window, probes
  the gain there, parks the audio and feeds the receiver from the same DSP
  block the demodulator would have had - a second consumer of the block, not
  a second owner of the device - and gives everything back when it stops.
  `ui/aircraft_view.py` is the third screen, a list that fills itself in with
  no tuning, no mode and no bandwidth to get wrong.

  Off air, indoors, on the stock aerial: **six aircraft in 70 seconds at
  ~800 messages a minute**, 248 positions, callsigns and altitudes and tracks
  all plausible for the Seattle approach, and **0 audio underruns and 0 ring
  overruns** across the excursion and back to a stereo FM station with its
  RDS name intact. Findings are in the ADS-B facts section of CLAUDE.md and
  in **Amendment 9**.

  Two things it deliberately does not do, both for the same reason as RDS. It
  does not use the checkword's error-correcting capability, and it does not
  decode altitudes above 50,000 ft, where the encoding changes to one there is
  no way to check against real air.
- **HD Radio (NRSC-5)** → the digital programme, and the extra stations
  beside it.

  **Status: complete and verified on hardware, 2026-08-28.**
  `decode/hdradio.py` runs the bundled `vendor/nrsc5/win-x64/nrsc5.exe` as a
  child process - `cu8` down its stdin untouched, 44.1 kHz audio up its
  stdout resampled to 48, station metadata off its stderr - and
  `Engine.set_hd` is a standing wish that starts a session on every station
  that carries one and hands the station back to the analog receiver after 12
  seconds on every station that does not. The switch and the subchannel list
  live in the Audio section at Simple.

  It cost two things the plan did not anticipate. A session runs at a window
  **no demodulator can be built for**, so the analog path is not merely
  bypassed but absent. And `rtlsdr_read_sync` cannot carry OFDM at all: the
  gap between one read and the next took MER from +10 dB to -13 and left the
  decoder with no audio, so `Reader` grew a gapless mode used only while HD
  holds the radio. Both are in **Amendment 10**, with the measurements.

  Off air: **MER +8 to +12.5 dB and 92 kbps** on local stations, seven of
  eight carrying HD and five of them carrying an HD2, and **0 audio underruns
  and 0 ring overruns** across a sweep, an aircraft excursion and a programme
  change.

  One thing it deliberately does not do: change programme without restarting
  the decoder. nrsc5 takes that only as a console keypress, which a pipe
  cannot deliver, so HD1 to HD2 costs the acquisition again.
- POCSAG → pager text. A strong "look what's out there" demo.
- Favourites, recently played, session history.

### Phase 5 — Packaging

- PyInstaller **one-folder** (`--onedir`, not `--onefile`: faster startup, far fewer AV false positives).
- Bundle blog-fork V1.4.0 x64 `rtlsdr.dll`, `libusb-1.0.dll`, `pthreadVC2.dll`, `msvcr100.dll` into `_internal/drivers/`.
- Ship a README noting the unsigned-binary SmartScreen warning and the "More info → Run anyway" path.

## Verification

**Without hardware** — `tests/synth.py` generates known IQ, which makes almost the whole DSP and scan stack testable:
- Synthesise WFM/NFM/AM/SSB at a known offset with a known audio tone → assert the demodulator recovers the tone frequency within 1 Hz and hits a target SNR.
- Plant N signals at known frequencies in synthetic wideband noise → assert the detector finds exactly N at the right centres and bandwidths, and that a pure-noise input yields zero detections.
- Feed synthetic signals with known characteristics through the classifier → assert correct labels.
- AGC, squelch, and noise blanker: assert bounded output level and no ringing on step inputs.
- Ring buffer: assert no data loss and correct wraparound under simulated overrun.

**With hardware** — a manual checklist script:
1. `python -m bettersdr.core.device --info` prints `Tuner: R828D`.
2. Scan 88–108; compare the result list against a known local station list.
3. Listen to the strongest station for 10 minutes; assert zero audio underruns logged.
4. Scan 118–137 with an antenna outdoors; confirm airband detections appear only when aircraft are transmitting.
5. Tune 1.0 MHz (HF, exercises the upconverter path) and confirm AM broadcast audio.
6. Full 24–1766 MHz sweep completes without a device timeout.
7. Waterfall runs at 30 fps at 4096-point FFT with no GUI stutter while audio plays.
8. Open the Aircraft screen, press Listen for aircraft, and confirm the list
   fills in with callsigns and altitudes; then leave the screen and confirm
   the station that was playing comes back at its own gain, in stereo, with
   its RDS name - the hand-back is the half that breaks.

## Risks

| Risk | Mitigation |
|---|---|
| Wrong `rtlsdr.dll` gets loaded | Bundle the blog fork, load by absolute path, verify `set_dithering` exists at startup |
| Audio dropouts from GIL contention | Reader thread does only `read_sync` + memcpy; jitter buffer absorbs scheduling jitter; all DSP vectorised, no Python loops per sample |
| Waterfall rendering cost at high FFT sizes | Rolling-index ring instead of array shifts; row averaging decouples display rate from FFT rate; drop frames rather than queue |
| Classifier mislabels and erodes trust | Persistence gate, confidence shown honestly, `Unknown signal` is a valid and non-embarrassing answer |
| Full-spectrum sweep feels slow (~50 s) | Band-targeted scans are sub-second and are the default; full sweep is opt-in with a progress bar |
| Feature parity balloons scope | Parity is Phase 3, after the differentiator ships; the checklist above is the fixed definition of done |
| Python 3.14 wheel gaps | Verified available: PySide6 6.11.2, NumPy 2.5.2, SciPy 1.18.1, pyqtgraph 0.14.0, sounddevice 0.5.6 |

## First steps on approval

1. Plug in the dongle, leave it on the stock Windows driver, and confirm what Windows binds it to — that's the Driver Doctor's starting-state fixture.
2. `git init`, scaffold the tree, `pyproject.toml`, venv + deps.
3. Download blog driver release V1.4.0 x64 into `drivers/win-x64/`.
4. Build `core/native.py` + `core/device.py` and get `--info` reporting `R828D`.

---

# Amendment 1 — Built-in driver installation (2026-08-27)

## Why

The original plan sent the user to Zadig with a step-by-step walkthrough. That
is still a big improvement on the status quo, but it leaves a third-party tool,
a dropdown with a wrong option that silently fails (Interface 1), and a manual
step between "plugged in" and "working". For an app whose entire premise is
removing setup friction, that is the wrong place to stop.

## Decision

Bundle **`wdi-simple.exe`** from libwdi and drive the driver installation from
inside BetterSDR. libwdi is the library Zadig itself is built on, and
`wdi-simple` is its reference command-line front end, explicitly intended for
embedding in application installers.

The invocation we need:

```
wdi-simple.exe --vid 0x0bda --pid 0x2838 --iid 0 --type 0 \
               --name "RTL-SDR (BetterSDR)" --progressbar
```

- `--type 0` selects WinUSB, the in-box Microsoft driver.
- `--iid 0` targets interface 0 of the composite device. This is the option
  users get wrong by hand, and hard-coding it removes that failure entirely.
- libwdi generates the INF, builds and self-signs a catalog, installs the
  certificate, and calls the driver-install API — the same work Zadig does.

## What this actually achieves

**The UAC prompt cannot be removed.** Installing a device driver is a
privileged operation on Windows; any approach that claims otherwise is either
wrong or doing something we should not ship. So the honest target is:

> plug in → app detects "no driver" → one button → UAC prompt → ~10 seconds → done

That is not literally zero-click, but it removes: finding and downloading
Zadig, knowing to enable *List All Devices*, choosing between two
identical-looking interfaces, and knowing which target driver to pick. In
practice it turns the single largest source of beginner failure into one
consent dialog.

## Design

New module `bettersdr/core/installer.py`, pure logic like `doctor.py`:

- `is_available()` — is the bundled helper present in this build?
- `install(interface=0)` — launch elevated via `ShellExecuteExW` with the
  `runas` verb, wait for exit, map the exit code to a result enum.
- Result states: `SUCCESS`, `DECLINED` (user dismissed UAC), `FAILED`,
  `UNAVAILABLE`.
- Never runs implicitly. The wizard asks first and explains what will happen.

Wizard flow becomes:

1. `doctor.diagnose()` reports `NO_DRIVER` / `DVB_T_DRIVER`.
2. Offer **"Set up my dongle"**, with a plain-English note that Windows will
   ask for permission and that this stops the dongle working as a TV tuner.
3. Run the installer, show progress, re-run `diagnose()` to confirm.
4. On `DECLINED` or `FAILED`, fall back to the existing manual Zadig
   walkthrough in `doctor.py`. **That walkthrough is not wasted work — it
   becomes the escape hatch**, and it is what a user with locked-down
   machine policy will need.

`drivers/win-x64/` gains `wdi-simple.exe` alongside the three DLLs.

## Risks

| Risk | Assessment |
|---|---|
| **Antivirus false positives** | An unsigned executable that installs a self-signed certificate and rebinds a USB driver matches malware heuristics closely. This is the most serious risk and it applies to the whole portable-exe distribution, not just this feature. It strengthens the case for eventually buying a code-signing certificate. |
| **Build dependency** | No reliably maintained prebuilt x64 `wdi-simple.exe` was found; we likely have to compile libwdi once (MSVC). Mitigate by committing the built binary to `drivers/win-x64/` exactly as we did with `rtlsdr.dll`, so the toolchain is not a per-developer requirement. |
| **Composite-device quirks** | libwdi issue #206 reports `wdi-simple` failing to bind WinUSB on composite devices in some configurations. Our dongle *is* composite. Needs real-hardware verification before it can be relied on, with the Zadig path staying available. |
| **Licensing** | libwdi is LGPL v3. We invoke it as a separate process, so there is no linking obligation. Ship the license text and a source offer. |
| **Corporate/locked-down machines** | Driver installation may be blocked by policy regardless. The manual fallback and a clear error message cover this. |

## Sequencing

This lands in **Phase 0**, since it is part of first-run and the diagnosis code
it builds on already exists. The manual Zadig walkthrough stays in place
throughout as the fallback path.

---

# Amendment 2 — Phase 1 audio findings (2026-08-27)

Phase 0 is complete: WinUSB is bound to interface 0, the tuner reports `R828D`,
and FM demodulates to clean audio. The audio half of Phase 1 is built and
verified on air. Three findings changed the design and are worth recording,
because none of them are visible in a test that only checks DSP correctness.

## Reading and demodulating on one thread loses 11% of the samples

Measured on a V4: a bare read loop sustains 99.5% of real time, and
interleaving WFM demodulation into that same loop drops it to 89%. While DSP
runs there is no USB transfer in flight, and the dongle's output is simply
lost — there is no back-pressure on a radio.

This validates the planned three-thread split, but it also means the split is
not an optimisation to defer. `core/reader.py` and `core/ringbuffer.py` were
built as part of the audio work rather than alongside the GUI.

## Read block size matters far more than expected

Capture percentage against block size, 10 s each:

| Block | Captured |
|---|---|
| 16 KB | 99.49% |
| 64 KB | 99.72% |
| 256 KB | 100% |

The original note that "16384 is the practical block size" holds for latency,
not for throughput. `Reader` now defaults to **128 KB**. The failure mode is
nasty: a 99.8% capture rate reports zero ring overruns and zero device errors,
and shows up only as audio underruns accumulating over minutes.

## Buffering cannot fix a clock mismatch

The dongle and the sound card are timed by separate crystals, so any residual
rate difference empties the jitter buffer eventually — a 0.2% shortfall drains
150 ms in about 75 seconds. `audio/output.py` therefore uses buffer depth as a
control signal and resamples each block by at most 0.5%, which is under a tenth
of a semitone and inaudible.

This is worth keeping in mind for Phase 3 recording: a WAV written from this
stream is very slightly rate-corrected. Raw IQ capture must be taken from the
ring buffer, ahead of the audio path, not from it.

## Consequence for Phase 2

`dsp/psd.py` was written to serve both the display and the scanner, in
calibrated dBFS that does not move with FFT size or window. The detector and
the waterfall must keep sharing it, so that a detection threshold always means
the same thing as the picture the user is looking at.

---

# Amendment 3 — HD Radio (NRSC-5) (2026-08-27)

## Why this is in scope

Standalone HD Radio receivers are scarce and expensive outside of cars, which
makes a $30 dongle plus software a genuine replacement for a $200 box. More
importantly it fits the product thesis exactly: HD stations carry **HD2/HD3/HD4
subchannels**, entire programs that exist only digitally. "There is a whole
second station hiding inside the one you are listening to" is the discovery
pitch delivered literally.

The metadata is also richer than RDS — station name, song title, artist, and
album art via the Station Information Service.

## What the air actually looks like here

Measured on this machine, 88–108 MHz, the eight strongest stations. Six show
unambiguous hybrid IBOC; two (93.3, 106.1) show one clear sideband and one
marginal, so the real count is probably seven or eight.

Spectrum of 94.9 MHz relative to its own peak:

```
     -180 kHz   -19.1 dB   <- IBOC        +140 kHz   -17.2 dB   <- IBOC
     -160 kHz   -19.4 dB   <- IBOC        +160 kHz   -16.8 dB   <- IBOC
     -140 kHz   -20.4 dB   <- IBOC        +180 kHz   -16.9 dB   <- IBOC
     -120 kHz   -31.7 dB                  +200 kHz   -21.2 dB
     -220 kHz   -39.0 dB  (noise floor)   +220 kHz   -38.3 dB  (noise floor)
```

Flat plateaus roughly 130–200 kHz either side of the analog carrier, about
14 dB below it, standing **20 dB clear of the noise floor**, dropping back to
noise beyond 220 kHz. Flatness across the plateau measured 0.7 dB standard
deviation — that is OFDM, not an FM skirt. There is ample signal margin here;
this is not a fringe-reception proposition on this antenna.

## The job splits in two, and the halves are not comparable

### Detection — cheap, ships in Phase 2

Measuring the shoulders needs nothing that `dsp/psd.py` does not already
provide: mean power in the 129–198 kHz sidebands, compared against the noise
floor beyond 240 kHz, requiring both sides present and flat. It lands in
`dsp/features.py` as one more classifier feature.

It is also *explainable*, which is the whole reason the classifier is
rule-based: "flat digital shoulders 130–200 kHz out, 14 dB below the analog
carrier → this station also broadcasts HD Radio."

**Detection is implemented.** See `dsp/features.py:detect_hd_radio`.

### Decoding — Phase 4, and one piece cannot be written from spec

| Stage | Difficulty |
|---|---|
| Retune to 1,488,375 S/s | Trivial |
| OFDM sync + FFT (4096-point, 363.4 Hz spacing) | Moderate |
| Differential QPSK demod | Moderate |
| Deinterleave (long block — why HD takes seconds to lock) | Moderate |
| Viterbi FEC decode | Moderate, and slow in NumPy |
| Layer 2 framing / PDU extraction | Tedious |
| **HDC audio codec** | **Not writable from spec** |

HDC is Xperi's proprietary codec. It is HE-AAC-derived but has no public
specification, so there is no clean-room path to it. The open-source answer is
**nrsc5**, which carries a decoder built on a modified faad2.

## Decision: wrap nrsc5 as a separate process

Do not reimplement NRSC-5. Pipe IQ to `nrsc5` running as a child process.

That one choice buys three things: it sidesteps the codec problem entirely, it
keeps a C decoder crash out of the Python process, and it keeps a clean
licensing boundary — nrsc5 inherits faad2's GPL lineage, which would otherwise
dictate terms for the whole bundled app. **Verify the licence before
committing to any tighter coupling.**

It fits the existing architecture well. The reader thread already owns the
device and writes `uint8` IQ into a ring buffer; HD mode adds a second consumer
of that ring alongside the demodulator, rather than a second device owner.

> **Amendment 10 corrects the last clause.** It is not a second consumer
> alongside the demodulator: at 1,488,375 S/s there *is* no demodulator, so an
> HD session replaces the analog path rather than running beside it. The
> reader also turned out to need a second reading mode - `read_sync` does not
> deliver a stream an OFDM receiver can decode.

## Two constraints to design around

- **1,488,375 S/s is mandatory**, and `1488375 / 48000 = 31.0078` — not a whole
  multiple, so HD cannot go through the `demod.py` skeleton, which requires an
  integer ratio by design. It needs its own path. Not a problem, but not a
  reuse either.
- HD mode therefore **retunes the device**, so sweeping and decoding HD cannot
  happen at once. Listening still gets ±744 kHz of spectrum, which is plenty to
  show the station and both its sidebands.

## Sequencing

- **Phase 2** — detection and labelling in the discovery list. Done.
- **Phase 4** — decoding, next to RDS. Note that HD supersedes RDS *for HD
  stations only*; RDS still matters for the majority that do not carry it.

---

# Amendment 4 — Phase 1 UI findings (2026-08-27)

## The GUI needed a third component the plan did not name

The plan describes three threads but gives the DSP thread no home. It ended up
in **`core/engine.py`**, deliberately outside `ui/`, so that `listen.py` still
proves the audio path with no Qt loaded at all. `ui/listen_view.py` is a view
onto the engine: it owns no threads, touches no device, and reads a single-slot
mailbox on a timer.

`choose_gain` also moved out of `listen.py` into **`core/frontend.py`**. Three
callers now need the same answer, and a scanner measuring the band at a
different gain from the display would make "the app found a signal I cannot
see" possible - the exact failure `dsp/psd.py` is shared to prevent.

## Stacked panes must share an axis width

The spectrum reserves ~52 px on the left for its dBFS labels; the waterfall and
ribbon originally did not, so all three were misaligned by that much - every
frequency read a few hundred kHz wrong. Fixed with a shared `AXIS_WIDTH` and a
`BlankAxis` that reserves the space without drawing. Worth stating plainly
because it is invisible in code review and obvious in a screenshot.

## The band plan had to arrive early

The plan puts `scan/bandplan/us.yaml` in Phase 2, but Phase 1's spectrum
specifies a band-plan ribbon, so the data file came forward. It immediately
earned a second job: **tuning into a new band now adopts that band's mode and
bandwidth**.

That is not a nicety. Tuned to the airband, the app was demodulating an AM
transmission as wideband FM - silence - and in Simple mode there is no mode
control to fix it with. The plan's line "classification drives demod
automatically" turns out to be load-bearing well before the classifier exists.
Applied only on a *change* of band, so a deliberate choice survives retuning.

## HD Radio detection is visible in the ordinary display

No extra work was needed to make the Amendment 3 detector user-facing: an "HD"
badge next to the band name, checked once a second. The sidebands are also
plainly visible in the waterfall as flat bars flanking each analog carrier, so
the picture and the label agree - which is the whole reason the detector reads
the same PSD the display draws.

## Verified on hardware

FM broadcast, NOAA weather radio and airband, at all three levels: zero audio
underruns, zero ring overruns, 4096-bin spectrum at 30 Hz with the waterfall
scrolling, auto gain, auto mode, and the HD badge correctly lit on 94.9 and
dark on 162.55.


---

# Amendment 5 — Phase 2 findings (2026-08-27)

Phase 2 is built and verified against live air. The shape the plan described —
sweep, detect, classify, list — survived contact with a real band unchanged.
What did not survive was several of the *numbers*, and one of the features.
Every correction below came from pointing the finished scanner at the sky and
disbelieving the answer.

The antenna for all of this was **indoors, about three feet from an exterior
wall**. That is worth stating because two findings only make sense in that
light, and because it is closer to what a beginner's first setup will look
like than a roof-mounted dipole would be.

## The detector found a bug in Phase 1's spectrum

The first synthetic sweep reported a signal at exactly the tuned frequency of
every step. It was not a detector fault: `psd.Spectrum` removed DC by
subtracting each frame's plain mean, which nulls the *unwindowed* sum rather
than bin 0. A strong signal off centre leaks into that mean, so subtracting it
writes the leakage back as a genuine three-bin spike at DC — measured 25 dB
above the noise floor with one FM station 37 kHz away.

Subtracting the window-weighted mean instead nulls the bin exactly and leaves
its neighbours alone. This had been shipping in the spectrum display since
Phase 1 and was invisible there: on real hardware the dongle's own DC offset is
larger than the artefact, so removal still looked like a clear improvement.

**It took something that reads the spectrum numerically to notice.** A display
is judged by eye, and the eye forgives a small spike in the middle where
everyone knows there is a DC spike anyway.

## The overlap does not cover the DC blind spot

Removing DC blanks the centre bin, which means every step is blind at its own
centre. The 25% overlap does not rescue it: with 1.8 MHz of usable tile per
step, a signal at one step's centre is 1.8 MHz from its neighbours, outside
their 2.4 MHz windows entirely. Only one step ever sees it, and that step
deletes it.

The fix is to keep the tile boundaries where they are and park the *tuner* 37
kHz away from each tile centre. The dead bin then lands on a frequency that is
not a channel on any raster in the band plan — 5, 10, 12.5, 25 or 200 kHz —
while the tile stays comfortably inside the window. A transmitter on a
non-standard frequency can still fall in the notch, which is why the offset is
a named constant with an argument attached rather than an accident.

## Rule-based classification works, but two of the rules were wrong

The plan's four features were carrier ratio, envelope variance, spectral
flatness and occupied bandwidth. Three survive. Envelope variance was dropped
before it was written: it needs time-domain IQ per detection, and the sweep has
only a PSD per step. The band plan turned out to carry enough prior knowledge
that it was not missed.

**Spectral flatness cannot separate analog FM from digital.** Synthetic
tone-modulated FM measures about 0.45 and OFDM about 0.94, which made a
threshold of 0.6 look generous. Real FM carrying music measures 0.7 to 0.9 —
programme material spreads energy as smoothly as noise does — so the app
confidently labelled most of the local dial digital. The synthetic signal was
misleading precisely because a single tone gives FM a line spectrum that real
audio never has.

Flatness now only decides where nothing is allocated, and the shape clause says
only what a power spectrum can support: whether there is a carrier, and whether
the power is spread out. What actually identifies a digital signal in a known
band is *where* the flat energy sits, which is what the HD sideband test
already does properly.

**This is the general lesson of the phase.** A synthetic signal tests the
arithmetic. It does not test the assumption, because it was built from the same
assumption.

## Occupied bandwidth is not measurable in one dwell

A 50 ms dwell measures a signal's *instantaneous* width. For anything carrying
speech or music that is not its bandwidth: an FM station caught between phrases
collapses to a bare carrier, and 98.1 MHz measured 11 kHz wide at 51 dB SNR —
strong, narrow, and duly reported as an anomaly rather than as the local
station it is.

Occupied bandwidth is bounded from below by modulation but not from above, so
the sweeper keeps the *widest* view of each channel across the passes it is
already making. That fixes the reporting and, as a side effect, supplies the
one thing that distinguishes a real station from a spur on a single spectrum:
**a station widens when somebody talks, and a spur never does.**

## The airband is full of things that are not aircraft

With the aerial indoors, a scan of 118–137 MHz found 83 signals: stable, 2 kHz
wide, 18–31 dB above the floor, surviving every persistence pass. They are real
RF — switching supplies, LED drivers, the dongle's own clock — and they sit on
25 kHz channels because the raster covers the whole band.

Tightening the persistence gate barely touched them, which was the useful
result: it falsified the assumption that they were noise. A gate that requires
two sightings in three passes rejects things that move, and these do not move.

So the classifier no longer lets a band match override an obvious shape. A bare
carrier a fraction of its channel wide, with all its power in one bin, is
labelled **"Unmodulated carrier"** and described as probable interference,
whatever is allocated there. The band still supplies the mode and bandwidth, so
Listen does something sensible. Reporting 83 aeroplanes over an empty sky would
have been the single most trust-destroying thing the app could do.

The persistence tolerance was wrong too, separately: at 586 Hz per bin, 25 kHz
is an 85-bin window, wide enough that noise peaks pair up with *each other*
across passes. Four kilohertz is the right number for matching a signal to
itself; 25 kHz remains right for merging one signal seen by two overlapping
steps. They had been the same constant, and were two different questions.

## Channel rasters are per-band data

`Band.snap` derived the first channel as half a raster in from the band edge.
That is right for US FM broadcast and wrong for most of the rest, and it put
the NOAA weather station on 162.550 MHz on screen as 162.537 MHz — a wrong
number in the one place the app is meant to be authoritative. NOAA channels
start at the band edge, US AM starts at 540 kHz with the band edge 10 kHz
below, marine VHF counts from 156.025.

`raster_base_hz` is now explicit data, and every raster in the plan is tested
against a frequency that genuinely exists.

## What the scanner ended up needing from the UI

Two things the plan did not name.

**One channel is one entry, decided after snapping.** Two detections either
side of a station's dip are 40 kHz apart — outside the merge tolerance — and
both are 100.1 MHz. Deduplicating on the snapped frequency is what makes them
one line. Keying on the frequency *and* the label is a trap: one transmitter
classifies differently from pass to pass, so 90.3 MHz appeared twice, once
correctly and once as interference.

**Strongest first, like a Wi-Fi picker.** Frequency order reads naturally on a
clean band and buries the real stations in a dirty one.

## Verified on hardware

Antenna indoors, three feet from an exterior wall.

- **FM broadcast 88–108 MHz** — 12 steps, three passes, **5.1 s**. 38–42
  stations, every one of them on a real odd-tenth channel, measured centres
  within a few kHz of the snapped frequency. HD correctly flagged on 94.9,
  88.5, 96.5, 90.3 and 101.5.
- **Weather Radio 162.4–162.55 MHz** — one step, 0.7 s. Both nearby NOAA
  transmitters found on 162.550 and 162.475; the third, audible only as a
  carrier, honestly reported as one.
- **Airband 118–137 MHz** — 83 carriers, none claiming to be aircraft.
- **A quiet synthetic band produces zero entries**, which is the other half of
  the acceptance criterion and the easier half to lose.
- **Scan, then Listen** — clicking a card switches to the listening view tuned
  to 94.9 MHz in WFM at 200 kHz, HD badge lit, **zero audio underruns and zero
  ring overruns**.

Phase 2's stated done-when was "a cold scan of the FM band lists real stations
at correct frequencies with no phantom entries, and clicking any card produces
audio". That is met.

---

# Amendment 6 — Phase 3 findings (2026-08-28)

Four decisions changed shape once the parity work met real hardware. None of
them are visible in a test that only checks that a control does something.

## The demodulator had to give up the end of the audio path

An AGC placed behind a fixed attenuator spends its range undoing it, and one
placed behind a limiter cannot recover anything the limiter has already
flattened. Both were true of `Demodulator.process`, which applied volume and
then clipped.

So the engine now builds every demodulator at unity with the limiter off, and
`dsp/chain.AudioChain` owns the tail: noise reduction, the audio band-pass,
the AGC, then volume, then the clip. Verified off air - at volume 0.5 the
audio measured -6.8 dBFS and at 0.1 it measured -21.9, exactly the 14 dB
apart the setting implies, with the AGC in circuit.

`listen.py` still uses the demodulator's own volume, which is why this is a
flag rather than a removal: the Phase 1 acceptance test has to keep working
with no engine at all.

## IF noise reduction belongs at the IF, and that has a price

Spectral subtraction on the raw 2.4 MS/s stream costs **33% of one core**.
After the channel filter, at a 240 kHz IF, the same stage costs about **3%**.
The cost barely moves with FFT size, because it is per-sample work rather than
per-transform, so there is no way to buy it back by making the transform
bigger.

That forced a hook inside the demodulator - `Demodulator._front` - and the
hook has to carry a remainder, because an overlap-add stage returns whole hops
and the audio decimator after it insists on a multiple of its own factor. It
also turned out to be the acoustically correct place: noise outside the
channel is about to be thrown away regardless.

The general lesson is the one from Phase 2 in a new place: **a time constant
stated per frame means something different at every rate**. 187 frames a
second at 48 kHz against 9,375 at 2.4 MS/s, so a noise tracker allowed to
climb "0.4 dB per frame" climbs fifty times faster on the wide window and
simply follows the signal.

## The calibration assistant had to learn to refuse

The obvious design - tune to a strong broadcast station, measure where its
carrier lands - produces a confident number from noise. A wideband FM station
has no carrier to speak of: its energy is spread over 150 kHz and the
strongest bin wanders with the programme. Six consecutive readings at
94.9 MHz had a standard deviation of **1814 Hz**, which is 19 ppm.

The same six readings against NOAA weather radio at 162.55 MHz spread by
**11.5 Hz**, or 0.07 ppm. So the assistant measures the capture in four
segments, reports the median, and refuses to answer when the segments disagree
by more than 1 ppm of the carrier — and the dialog asks for a weather-radio or
AM carrier and says in as many words that an FM music station will not work.

The sign was settled by experiment rather than by argument, because it depends
on librtlsdr's internals: forcing +50 ppm moved the measured offset by
+8246 Hz at 162.55 MHz, against the +8128 Hz that +50 ppm of that frequency
comes to. Planting +20 ppm and −15 ppm and running the assistant recovered to
within 35 Hz in one step from both directions.

Worth knowing for expectations: **this dongle is already accurate to
0.24 ppm**, because the V4 has a TCXO. The assistant correctly reports 0 ppm
on it. The feature earns its place on V3s and clones.

## The control column outgrew a form layout

Phase 3 takes the listening screen from eight controls to about seventy-five
at Expert, which is more than fits on a laptop. `ui/widgets/panel.py` groups
them under headings in a scroll area, and a section hides itself when every
row under it belongs to a higher level — otherwise Simple mode is a screen of
headings with nothing beneath them, which reads as an app that has broken
rather than one being quiet. Measured at 3 rows under 1 heading in Simple, 27
under 5 in Standard, 75 under 8 in Expert.

The `QScrollArea` viewport rule from Phase 2 applies here too and is repeated
in that module: a stylesheet on the viewport drags every descendant through
the stylesheet style, and `QLabel` is a `QFrame`, so every label starts
painting a border it never asked for.

## Verified on hardware, 2026-08-28

One session, FM broadcast to the AM band and back, then a full band scan:

| | |
|---|---|
| Volume 0.5 → 0.1 | −6.8 → −21.9 dBFS (14 dB, as asked) |
| Mute | exact silence, not a small number |
| FFT size changed live | 4096 → 1024 bins, no gap in the audio |
| Hop to 710 kHz | gain re-measured 3.7 → 29.7 dB, window 2.4 MS/s → 240 kS/s |
| Back to 94.9 MHz | gain 29.7 → 3.7 dB |
| Scan of 88–108 MHz | 39 signals in 5.0 s |
| Recording | 3.0 s audio and IQ, 4.8 MB/s for IQ exactly as documented |
| **Underruns and ring overruns, whole session** | **0 and 0** |

---

# Amendment 7 — RDS, and a clock that was not where it said it was (2026-08-28)

## The premise the design started from was wrong

The plan for the subcarrier was to lock 57 kHz to the third harmonic of the
19 kHz stereo pilot, the way the standard says a transmitter builds it. That
was dropped early for a good reason — it would tie RDS to stereo and lose it
on mono stations that carry it perfectly well — and replaced with a
free-running oscillator, on the argument that the multiplex comes out of an FM
discriminator and so is timed by the dongle's TCXO, which is good to 0.24 ppm.
0.24 ppm of 57 kHz is 0.014 Hz. Nothing to track.

That argument is wrong, and the measurement says so: the ratio between the
station's baseband and ours is **98 ppm**. Not the dongle's crystal — the
whole path, sample rate included. On the subcarrier that is 5.6 Hz, which
rotates the constellation 28 degrees across a single DSP block, and on the
symbol clock it is a whole symbol every eight seconds.

The symptom was diagnostic in hindsight. Block quality sat at 0.78 and got
*worse* as the DSP block grew — 0.59 at 16 KB — which is the signature of a
per-block estimate being asked to stand in for something that moves within the
block. Tracking a rate alongside the phase, in both the timing loop and the
carrier loop, took the same recording from 0.78 to 0.97 and made block size
stop mattering at all. This is the "anything sized for 2.4 MS/s is a latent
bug" rule wearing a different hat: a constant that is only right at one block
size is the same defect whichever end of the chain it sits at.

## An early-late timing detector cannot lock biphase

It was the first thing tried and it produced 19% bit errors while the same
signal decoded perfectly at a fixed sampling phase — which is how the timing
loop, rather than the demodulator, was identified as the fault. The reason is
structural: a biphase symbol correlates almost as strongly against its own
inverse half a symbol away, so the detector has two stable points and only one
of them is the symbol.

What replaced it measures the criterion directly — the matched filter read at
sixteen positions across the symbol, take the largest, refine with a parabola
through its neighbours. One maximum by construction, and the peak stands out
by four to seven times over the sixteen symbols one block holds.

## What a station calls itself is not what it sends

Two separate lessons, both from the same field:

- 94.9 MHz alternates `" on KUOW"` and `"NPR's He"` — it scrolls a programme
  title through the eight characters meant for a name. Confirming each
  character position separately, which sounds equivalent to confirming the
  frame, mixes neighbouring frames and yields `"ren KUow"`. Frames are
  accepted whole and in order; a name is only treated as a name once two
  identical ones arrive, and otherwise the callsign wins. That also decides
  what a saved bookmark is called — `KUOW`, not `BBC News`.
- One corrupt block in a thousand passes its checkword, which over a minute of
  a weak station is a certainty. One of them put a Los Angeles callsign on
  102.5 MHz. The identifier now has to arrive twice running.

## The callsign is arithmetic, and stations disagree with it anyway

RBDS encodes most American callsigns as a base-26 number. KUOW (0x4652) and
KING (0x2678) both come out right. 102.5 MHz sends 0x137A, which decodes to
KBIG — exactly 0x4000 from the code its own callsign implies. That is the
station's data and not our arithmetic, and it is worth knowing that a small
minority of stations will be labelled with somebody else's call. Codes outside
the two arithmetic ranges get no callsign rather than a guess.

## Verified on hardware, 2026-08-28

Three local stations, indoor aerial, 12 s each:

| Station | PI | Callsign | Radio text | Block quality |
|---|---|---|---|---|
| 94.9 | 0x4652 | KUOW | "NPR's Here & Now on KUOW" | 0.97 |
| 98.1 | 0x2678 | KING | "Edward German, Theme and Six Diversions…" | 0.94 |
| 102.5 | 0x137A | KBIG | "Seattle's Rock" | 0.71 |

0 audio underruns and 0 ring overruns throughout. The offscreen GUI showed
`KUOW   News` in the header, `BBC Newshour on KUOW` beneath it, and saved a
bookmark named `KUOW`.

## Still open in Phase 4

- **Stereo blend on a weak signal.** The difference channel sits at 23–53 kHz
  where FM noise rises as f², so a weak station is 20 dB noisier in stereo
  than in mono. Real receivers blend towards mono as the signal drops. The
  measurements below say the local stations do not need it, but a fringe one
  will.
- **Group 4A clock time, and alternate frequencies (0A blocks C/D).** Both are
  small additions to `RdsDecoder.update`.
- **POCSAG**, and favourites/recently-played, all untouched. HD Radio
  shipped; see **Amendment 10**.
- **A map for the aircraft screen.** Positions are decoded and shown as
  numbers; a map is the obvious next step and is a self-contained piece of
  work. Nothing else needs it.
- **Aircraft heard while doing something else.** Tracking takes the radio
  over completely, which is honest but exclusive - a second dongle, or
  time-slicing against listening, is the only way round it and neither is
  worth doing yet.
- **RDS in the Discover list.** Not possible from a sweep — one group takes
  87.6 ms and a dwell is 50 ms — so it would need a second pass that parks on
  each candidate for a second or two. Worth doing, and a different feature
  from the sweep.

---

## Amendment 8 — FM stereo, and three ways to get a suppressed carrier wrong (2026-08-28)

Stereo is the oldest subcarrier on the FM dial and by far the easiest to
believe you have working when you have not. Every failure below produces
audio. Most of them produce *good* audio.

### The three that matter

**A suppressed carrier has to be inferred, and the error is multiplicative.**
The transmitter sends L−R on a 38 kHz subcarrier it then removes, leaving only
a 19 kHz pilot to say where it was. Recover that phase 90° out and the
difference channel is multiplied by cos(90°) — it does not distort, it
vanishes, and what comes out is a clean mono broadcast. There is no artefact
to notice. `StereoDecoder` gets the phase by squaring an *analytic* pilot:
a complex bandpass keeps only positive frequencies, so squaring doubles the
frequency and carries the phase with it, with no oscillator to lock and no
loop to tune.

The sign of that square is not a detail. A sine's analytic form is −j·e^{jωt},
so squaring gives −e^{2jωt} and the imaginary part comes back as −sin(2ωt).
Left alone, every broadcast plays with its channels swapped — which measures
as *perfect* separation on any test that only asks how different the two
channels are. `test_the_channels_are_not_swapped` exists because the headline
test cannot see this at all.

**A filter's group delay on the pilot is a phase error, not a delay.** The
bandpass isolating the pilot has to reject audio at 15 kHz and the difference
channel at 23 kHz, which at a 240 kHz multiplex rate is 289 taps and 144
samples of delay. That is 0.6 ms — eleven cycles at 19 kHz, so the reference
would be at an essentially random phase, and the separation would wander with
the tuning. The fix is to hold the multiplex back by exactly the same 144
samples, which is why `process` returns the **sum as well as the difference**:
a caller that took only the difference and used its own undelayed sum would
put the two ears 0.6 ms apart. Making that impossible is worth more than the
convenience of a passive tap like the RDS receiver, and it is the one
structural difference between the two.

**A mono station is not silent at 19 kHz, it is noisy there.** Detecting a
pilot by asking whether anything is present sets a threshold against the noise
floor, which moves with the signal, the gain and the band. So the decision is
a *ratio*: pilot power against a guard band at 16.8 kHz, which is above the
audio and below the pilot and allocated to nothing on any station. Measured on
air, the gap is not delicate — five local stations read 13.8 to 27.6 dB and
the same measurement on a mono broadcast reads about 0.

### Measured

Synthetic broadcast, one channel carrying a 1 kHz tone and the other silent,
through the whole demodulator at 2.4 MS/s: **61 dB of separation**, and 25 dB
still under noise at 0.02 amplitude. Off air, indoors:

| Station | Signal | Pilot vs guard | L−R vs L+R | L/R correlation |
|---|---|---|---|---|
| 94.9 (KUOW, talk) | −1.4 dBFS | 27.6 dB | −27.9 dB | 0.997 |
| 98.1 (KING, classical) | −16.8 dBFS | 26.4 dB | −4.7 dB | 0.495 |
| 102.5 | — | 18.5 dB | −13.1 dB | 0.907 |
| 88.5 | −15.4 dBFS | 26.8 dB | −0.2 dB | 0.021 |
| 96.5 | −18.7 dBFS | 21.2 dB | −7.6 dB | 0.706 |

The correlation column is the interesting one, and it needed a second
measurement to interpret. 88.5 reads 0.021 — two channels sharing almost
nothing — which is either an unusually wide broadcast or the difference
channel being noise, and those look identical in a level measurement. The
separator is *shape*: programme material rolls off towards 15 kHz and noise
does not. Measured as the fall between the 0.3–3 kHz and 8–15 kHz bands, the
difference channel tilts 27.5 dB against the sum's 24.8 dB on 88.5 — steeper,
so it is programme. On 96.5, the weakest of the five, it is 18.6 against 24.3:
5.7 dB shallower, which is noise starting to show. That is the station a
stereo blend would eventually be for.

### What the audio path had to become

Stereo is the first thing in this app that is not one number per instant, and
the path from the demodulator to the sound card assumed it was throughout.
Rather than a parallel stereo path, every stage now takes a **frame axis**:

- `AudioSink` opens **two** channels whatever the radio is doing, and conforms
  each block on the way in — mono duplicated, stereo averaged if the device
  will only take one. The alternative, reopening the stream when the pilot
  comes and goes, would put a gap in the audio several times a minute on a
  marginal station. A device that refuses two channels falls back to one
  rather than failing to play at all.
- `ClockSync` stretches every channel onto the *same* grid. Resampling them
  independently — even to a length that rounds differently — is an image that
  wanders.
- `Squelch`, `BiquadState` and `Agc` count frames rather than samples and
  share one control signal across channels. `lfilter` defaults to the trailing
  axis, which on a `(frames, 2)` block filters the left channel against the
  right; and two independent gain riders are a stereo image that moves about,
  which is a far stranger fault than either channel being a decibel off.
- `AudioRecorder` takes a channel count, fixed when the header is written, and
  conforms every block to it. A station that drops its pilot mid-recording
  keeps the file it started rather than producing a WAV that changes channel
  count halfway — which is not a recoverable file, it is a burst of noise at
  double speed.

**Spectral noise reduction is the exception, and it says so.** It builds one
noise estimate and one gain mask per channel; run twice, the two estimates
diverge and pull the image apart. It is a tool for a weak, hissy, mono signal,
so `AudioChain` mixes down when it is on and `keeps_stereo` reports it. The
badge is lit from what actually reached the sound card rather than from the
pilot, so switching noise reduction on visibly turns stereo off instead of
leaving a badge that quietly stopped being true.

### Cost and verification

**20 ms per second of radio — 2.0% of a core** at 2.4 MS/s, on top of the
demodulator's 6.3%. On by default for the same reason as RDS: the difference
channel is right there in the multiplex, and a broadcast has been in stereo
since 1961.

Verified through the GUI against live air: the badge lights on 98.1 (with RDS
reading `KING` in the same header), goes out when the decoder is switched off
and comes back, stays out on NFM at 162.55 MHz, and lights again on 94.9. A
two-second recording wrote a genuine two-channel WAV. **1 audio underrun and
0 ring overruns** across the whole sequence, which included two mode changes,
a sample-rate change and a recording. The headless CLI still runs mono at
100.7% capture with 0 underruns.

---

## Amendment 9 — ADS-B in the app, and what "borrowing the radio" costs (2026-08-28)

The decoder was finished and tested against synthetic bursts a day before any
of this; wiring it in was expected to be plumbing. It was not, and the reason
is worth stating in one line: **aircraft tracking is the first feature in the
app that borrows the radio**, and the app had one existing borrower - the
sweeper - whose rules had never been written down.

### The shape of it

1090 MHz has nothing to listen to. Mode S is a 1 Mbit/s data burst, so there
is no demodulator, no audio, and no reason to keep the audio path running -
`_run` parks the sink for the whole session through the same expression that
parks it for a gain probe, so parking can never be started by one mechanism
and ended by the other. The receiver is fed the same block the demodulator
would have had, which is what the DSP thread already exists to hand out; the
spectrum keeps updating from it, so the screen shows the band being watched
rather than freezing.

That leaves the borrowing. `_begin_adsb` remembers the window, widens it,
parks the tuner at 1090 MHz and measures the gain there; `_end_adsb` puts the
window back, retunes, and measures again at the frequency being returned to.
Both are modelled on `_begin_scan` and `_end_scan`, deliberately, down to the
order of the steps.

### Three ways to get the giving back wrong

All three were found in one offscreen GUI session against real hardware, and
all three produced an app that looked like it was working.

**`center_hz` is not where the tuner is. It is where the user is.** Every view
reads it to configure itself, and `_begin_adsb` originally moved it with
`tune()`. Leaving the aircraft screen then took the listening screen through
the band plan's *aircraft* entry - `raw` mode, a 2 MHz channel - and asked for
a gain measurement, which arrived from a 1090 MHz probe at **49.6 dB**. That
was applied to 94.9 MHz: 50 dB into overload, audio pinned at 0.9 dBFS, no
RDS, no stereo. The sweeper never had this bug because it borrows the tuner
without touching `center_hz`, and the display frame carries the *borrowed*
frequency so the spectrum still says where its samples came from. That is now
a rule in CLAUDE.md rather than a property of one function.

**The page arriving configures itself from a radio the page leaving has not
handed back yet.** `_show_page` started the incoming view before stopping the
outgoing one, so the listening screen read the radio's state mid-excursion.
Stopping first is not tidiness; it is the ordering the whole hand-back depends
on.

**A de-duplicated gain probe can be the wrong probe.** `auto_gain` suppresses a
measurement while one is already queued, which is right when two callers ask
about the same band and wrong when the queued one was measured 995 MHz away.
The listening screen asks for one the instant it is shown, so `_end_adsb`
submits its own directly, behind the retune, and is the last word.
`_probe_scan_gain` exists for precisely the mirror image of this, and the
comment there says so - which is not the same as the code having been reused.

### Measured, first sky

Indoors, stock aerial, 70 seconds: six aircraft, ~800 messages a minute, 248
positions, 0 audio underruns, 0 ring overruns. Callsigns (SKW3857, ASA503,
ASA1897, N68767, RFS703), altitudes from 1,350 to 6,800 ft and positions
within a few tenths of a degree of the Seattle approach - all consistent with
each other frame to frame, which is the check that matters when there is no
reference to compare against.

Two numbers surprised. **Bad frames outnumber good ones two to one** (2,013
against 913), which is the candidate gate working as designed - it passes
anything vaguely preamble-shaped and lets the checkword reject it - and not
aircraft being missed. It is shown only at Expert, next to the message rate,
because next to a healthy list it reads as a fault to anyone else. And
**messages arrive between -3 and -24 dBFS**, so the four strength bars had to
be respread: the first thresholds, set from first principles against full
scale, gave every aircraft in view four bars.

### The one decoder bug a real sky found

Nothing synthetic had produced it, and it is the kind that does not announce
itself: an aircraft near Boeing Field appeared at **57 degrees east**, with a
latitude still correct to four decimal places, sitting in a list where every
other row was right.

CPR sends position in two halves, and a *surface* half divides the globe into
90 degrees of latitude where an *airborne* half uses 360. `_position` passed
the arriving frame's kind into the decode and applied it to both halves, so an
aircraft that sent one of each - which is precisely what an aircraft on final
approach does, within a second or two - had the wrong span applied to half the
arithmetic. Reproduced synthetically at 11.9 N, 59.6 E from a Seattle
position: not off the globe, not impossible, just wrong, and there is nothing
downstream that could ever have noticed. Each stored half now carries the kind
of frame it came from, and only matching halves pair.

The general lesson is the one the checkword rule already states: this decoder
would rather report nothing than report something plausible and wrong. The
guard costs a landing aircraft one pairing cycle, which `_cpr_local` covers
from the position it already had.

### The screen

The card is *updated*, never rebuilt - an aircraft reports twice a second and
a rebuilt widget would flicker, fight the scrollbar and slam shut anything
being read - and the list is only reordered when the set of aircraft changes,
so rows do not swap places under the cursor. Everything the card says in words
(`31,000 ft, climbing`, `heading west (272°)`, `Heard 3 s ago`) is a pure
function tested without a window, and an aircraft that has not said something
gets no line for it rather than a zero. A vertical rate under 200 ft/min is
not called a climb, because otherwise every aircraft on the screen is
permanently climbing and the word stops meaning anything.

## Amendment 10 — HD Radio in the app, and the one thing `read_sync` cannot carry (2026-08-28)

`decode/hdradio.py` was finished and passing its tests against nrsc5's own
sample recording before any of this. Wiring it in was expected to be a third
copy of the ADS-B excursion. It was not, and the reason is a single sentence
that invalidates a line in Amendment 3: **reading the dongle one block at a
time does not deliver a signal an OFDM receiver can decode.**

### The switch is a standing wish, not a command

"Play the digital version" is not a per-station decision the way a mode is.
Somebody who wants HD wants it on every station that has one, so
`Engine.hd_enabled` is a wish and the engine starts and stops sessions to
match it: on a retune, on a mode change, on the way back from a sweep or the
aircraft screen. `_apply_hd` recomputes whether a session is possible rather
than remembering, because every input to that question can change underneath
it.

The corollary is the part that makes it usable. A station that turns out not
to carry HD, or carries it too weakly to decode indoors, gets **12 seconds**
and is then handed back to the analog receiver with a line saying so — the
switch stays on for the next station. Acquisition measured 3–5 seconds on a
strong local signal and 5.5 on the recording, so 12 is comfortably clear of a
station that is going to work. Without this, leaving the switch on and tuning
to 100.7 would be a radio that is simply silent, which is exactly the kind of
dead end this app exists to not have.

### The third borrower, and the first that does not go anywhere

A sweep and the aircraft screen both take the tuner somewhere else. HD rides
on the same carrier as the analog broadcast, so it borrows the **window** and
nothing else: `center_hz` never moves, and the display frame is labelled with
the frequency the listener is actually on. What it borrows is stranger than a
frequency, though. 1,488,375 S/s is fixed by the standard and is not a whole
multiple of 48 kHz, so for the length of a session **there is no demodulator
at all** — `demod.create` refuses that rate by design, and rightly, so
`_rebuild` has to not ask. A mode or bandwidth chosen during a session is
remembered on `_wanted_mode` and `_wanted_bandwidth_hz` and applied when the
window comes back, which is also the moment the analog path exists again.

Amendment 3 said HD mode "adds a second consumer of that ring alongside the
demodulator". It does not. It **replaces** the demodulator, because at that
window there is nothing to run.

Two smaller things had to be put away for the duration. Offset tuning, because
the decoder is fed the raw bytes ahead of the front-end chain and a shift
applied in software never reaches it — nrsc5 would be handed a station a few
hundred kHz off the middle of its window. And the window control itself: there
is exactly one rate nrsc5 accepts, so a request during a session is read as a
statement of what to come back to. That last one is not hypothetical —
`_guard_window` re-asserts the current rate on every retune, and taken
literally it would have narrowed a listener's 2.4 MS/s to 1.44 the moment they
moved the dial.

### The finding: `rtlsdr_read_sync` leaves a gap, and OFDM cannot cross it

The first session on air synchronised, read the station name off the air, and
produced **no audio at all**, at a modulation error ratio of −13 to −17 dB.
nrsc5 driving the same dongle itself, minutes apart, gave **+10.8 dB and
91.8 kbps**.

Gain was the obvious suspect and was wrong. A sweep of the whole tuner table
against a real decode — eight settings from 2.7 to 40.2 dB — never produced a
usable MER at any of them. Neither did capturing to a file and decoding it
offline, which is what located the fault: the capture itself was bad, at
99.84% of real time with **zero ring overruns**. Nothing was being dropped by
us. The samples were missing at the source.

Between two `rtlsdr_read_sync` calls there is no USB transfer in flight. That
gap is the reason `Reader` uses a large read size at all, and until now the
whole cost of it was a fraction of a percent of samples — inaudible on any
analog mode, invisible to the scanner, irrelevant to ADS-B, all of which work
a block at a time and start each one fresh. An OFDM receiver tracks a frame
*across* blocks, so what matters is not the fraction lost but **how often the
stream is discontinuous**. Raising the read size to 1 MB (352 ms) took MER
from −17 dB to +6.3 and still lost sync five times in fifteen seconds; the
capture percentage was 100.06% at both sizes and said nothing about either.

`rtlsdr_read_async` keeps several transfers queued, so there is never a moment
with nothing in flight. Same station, same gain, same rate, nothing else
changed: **+9 to +10 dB, 91.8 kbps, no loss of sync in fifteen seconds.**

So `Reader` grew a second mode. It is used only while HD Radio holds the
radio, because the block-at-a-time loop is what gives the scanner a
synchronous point to retune at and that is worth keeping. The callback
librtlsdr delivers transfers to runs on the reader thread, so `Device` still
has exactly one owner and `tests/test_reader.py` still holds.

### The trap inside that: libusb is not reentrant

The first version drained the command queue from inside the read callback,
which looks like the natural place — it is the reader thread, and it is
between transfers. It is not. A control transfer issued from within libusb's
own event handling comes back `LIBUSB_ERROR_BUSY`, which librtlsdr reports as
`r82xx_set_freq: failed=-6` and then carries on regardless. The result is a
radio that **did not retune** and a decoder cheerfully reporting the station
it was already on. Found by surveying eight local frequencies and reading the
same call letters off all eight.

A pending command therefore ends the stream, `run` handles it the ordinary way
outside `read_async`, and streaming is entered again afterwards. That costs a
discontinuity per command, which is affordable because nothing queues commands
during a settled session — and the one thing that does, a retune, restarts the
decoder anyway.

### The acquisition gap, and one underrun count that lied

There is nothing to play for the first few seconds. A real HD receiver covers
it by playing the analog signal and blending, and we cannot: at 1,488,375 S/s
there is no analog path to blend from. So the sink is parked until the first
digital audio arrives, once, in one direction. Parking it again on every
drop-out was rejected — reopening the sound card at every flutter puts a gap
in the audio in exactly the conditions that least need one, and a mid-session
drop-out is the signal genuinely going away, which is worth seeing.

Wiring that up exposed an older fault with a wider blast radius than HD.
`_apply_sample_rate` stops and restarts the sink, and said nothing about it to
the parking bookkeeping — so a rate change *inside* a parked stretch left
`_run` believing the audio was still parked, and it therefore never parked it
again. Coming back from the aircraft screen into an HD session does exactly
that, and the sink played into the gain probe and the whole acquisition:
**162 underruns**, against 0 with the flag corrected at its source. The same
latent bug was reachable without HD at all, on the way back from aircraft
tracking to an AM station, where the two windows also differ.

### The screen

The switch lives in **Audio**, not in Radio, and at Simple. From where the
listener sits that is what it is — a choice about what comes out of the
speakers — and everything it does to the receiver is the engine's business.
It is offered only where it could work: a build with the decoder bundled, on a
band the plan says is broadcast FM, in `wfm`. A control that can only ever
fail is worse than no control.

Underneath it, the subchannel list, which is most of the reason to want the
feature: `HD2 — Jazz`, `HD3 — Talk`. It appears only when a station announces
more than one, it is rebuilt only when the labels actually change rather than
thirty times a second under the cursor, and the last list is kept across a
restart because changing subchannel empties it for a few seconds — precisely
when somebody is reaching for it. A restricted programme is labelled
`subscription only` rather than hidden. The subchannel does not survive a
retune: HD2 on one station has nothing to do with HD2 on the next, and asking
for one that is not there costs twelve seconds of silence to discover.

The digital signal wins over RDS while a session runs — it is the same station
saying the same things over a channel with a checkword on it, and the analog
receiver is not running to contradict it. The badge names the programme
(`HD2`) instead of just lighting, which is the only place on screen that says
which of a station's broadcasts is playing, and the passband marker widens to
the full 396 kHz of the hybrid signal, which is the clearest explanation the
display can give of where the extra sound is coming from.

### Verified on hardware, 2026-08-28

Through the app, on KUOW 94.9: first digital audio at **5.0 s**, MER **+8.0
dB**, **92.1 kbps** against the analog-equivalent 64, title `BBC World
Service`, artist `Seattle's NPR News Station`, no loss of sync in 21 seconds,
**0 dropped IQ blocks, 0 audio underruns, 0 ring overruns** including both
transitions.

Eight local FM stations surveyed at 15 s each, all decoding: **KNKX 88.5**
(HD1 Public, HD2 Jazz), **JACK 96.5** (HD1 Adult Hits, HD2 Rock, HD3 Talk),
**KING-FM 98.1** (HD1, HD2 and HD3 Classical), **KZOK 102.5** (HD1 Classic
Rock, HD2 Top 40), **KNDD 107.7** (HD1, HD2), **KHTP 103.7** and **KUOW 94.9**
(HD1 only). 100.7 carries no HD and fell back to the analog broadcast on its
own, which is the intended answer rather than a failure.

Switching programmes on KNKX 88.5 at MER +12.5 dB: HD1 playing `E.S.T. — Did
They Ever Tell Cousteau`, HD2 playing `Bill Evans — Waltz For Debby`, and back
again, at 3.0–4.0 s of acquisition each way and **0 underruns**.

And against the other two borrowers, from an HD session: a full 88–108 MHz
sweep (45 signals) and an aircraft excursion, both returning to HD playing at
MER +12.2, then off to a stereo FM station with its RDS name intact — **0
underruns and 0 ring overruns end to end**.

---
