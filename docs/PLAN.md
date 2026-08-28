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

### Phase 4 — Decoders and polish

- **RDS decoding** — 57 kHz subcarrier, BPSK 1187.5 bps, differential decode, block sync on offset words, group 0A/0B for the station name. This is the single biggest "wow": the discovery list stops saying `FM Radio — 98.5 MHz` and starts saying `KQED 88.5`. Worth the effort.
- FM stereo (38 kHz pilot-locked L−R subcarrier).
- ADS-B (1090 MHz) → aircraft callsign list; POCSAG → pager text. Both are strong "look what's out there" demos.
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
