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

  A weak station is now faded towards mono rather than left in noisy stereo:
  the difference channel is weighted between full and nothing across 20 dB
  to 11 dB of pilot margin, and a station blended all the way down is
  reported as mono rather than as stereo with nothing in it. Added
  2026-08-29; **Amendment 13** has the measurements and the one bug worth
  knowing about.

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
  no tuning, no mode and no bandwidth to get wrong - and, since 2026-08-29, a
  map above it, framed on whatever has been heard, with trails, a graticule,
  a scale bar and the coastline of the United States behind it.
  **Amendment 14** has that.

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
- **POCSAG** → pager text. A strong "look what's out there" demo.

  **Status: complete, tested against synthetic transmissions; not yet heard
  off air.** `decode/pocsag.py` takes the FM discriminator output and produces
  pages with a capcode and their text: a DC-tracked slicer with an
  interpolating bit clock running at all three baud rates at once, stateless
  frame sync on the 32-bit sync codeword, BCH(31,21) with one bit of
  correction, and both the numeric and the 7-bit alphanumeric readings of the
  message. It costs **1.3% of a core**, hangs off a tap of its own on
  `_FmBase` beside the RDS one, and attaches only on a narrow FM channel —
  which is what keeps it off broadcast FM, where it would find nothing.

  `ui/widgets/pagerlog.py` is a panel under the waterfall that stays hidden
  until a sync codeword has actually been seen, so the moment it appears is
  itself the finding. `scan/bandplan/us.yaml` gains the 929–932 MHz paging
  band as a scan target.

  Two things it deliberately does not do, both the RDS rule again. It corrects
  one bit rather than the two BCH(31,21) can manage, because a mis-corrected
  address codeword is somebody's message shown against somebody else's pager
  number. And it reports a page with no message as a beep rather than as an
  empty string, because that is a real thing pagers do.

  **What has not been tested: the whole thing against real air.** Nothing in
  it has met a transmitter. See **Amendment 11**.
- Favourites, recently played, session history.

  **Status: complete, 2026-08-29.** `core/history.py` records where the radio
  has been — a dwell gate so that tuning across a band is not mistaken for
  listening to it, a recent list persisted to `history.json`, and this
  session's trail behind a Back button. `Bookmark` gains a `favourite` flag,
  so a favourite is the saved entry marked rather than a second record of the
  same station. The listening screen gains a Recently played section at
  **Simple**, where nothing else is: a beginner who tunes away from something
  they were enjoying previously had no way back to it. Findings are in
  **Amendment 12**.

  This first shipped with a strip of favourite and recently-played chips above
  the band buttons on Discover as well (`ui/widgets/quicktune.py`). It was
  removed on 2026-08-29: Discover had grown band chips, a scan row, type chips
  and a status line above the list it exists to show, and the one thing on that
  screen a beginner does not need is a second route to a station they have
  already found. The route that stays is the one on the screen where somebody
  is actually listening.
- Stepping through what was found, from the listening screen.

  **Status: complete, 2026-08-29.** Buttons either side of the frequency
  readout walk the Discover list without going back to it — `results.neighbour`
  against the list *as displayed*, so the order the user chose is the order
  they step in and a kind they have hidden is a kind they skip. It wraps, the
  way a car radio's seek does, and a dial that is nowhere in the list enters it
  from whichever end was pressed. Available at **Simple**, the same argument as
  Recently played: Simple has no mode control and no bandwidth, so a beginner
  handed a list of eleven stations should not have to visit the other screen
  eleven times to hear them.

  Two things it deliberately does not do. It does not record a frequency the
  dial merely passed through, and it does not count time spent on another
  screen — a station left playing behind Discover accrues nothing, because
  "played" means somebody was listening.

### Phase 5 — Packaging

> **Amendment 17 replaces the third bullet and demotes the first.** The
> SmartScreen prompt this was written for has been superseded by Smart App
> Control, which blocks an unsigned build outright with no way past the
> dialog. The supported route is now to clone the repository and run
> `py tools/setup.py`. The PyInstaller build is built, checked as far as it
> can be without starting it, and unverified.

- PyInstaller **one-folder** (`--onedir`, not `--onefile`: faster startup, far fewer AV false positives).
- Bundle blog-fork V1.4.0 x64 `rtlsdr.dll`, `libusb-1.0.dll`, `pthreadVC2.dll`, `msvcr100.dll` into `_internal/drivers/`.
- Ship a README noting the unsigned-binary SmartScreen warning and the "More info → Run anyway" path.
- The README credits Natural Earth for the map data (public domain, so this
  is courtesy rather than obligation) and states nrsc5's GPL-3 licence and
  the process boundary that keeps it off the rest of the application.

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

- ~~**Stereo blend on a weak signal.**~~ Shipped on 2026-08-29. The penalty
  measured 15.4 dB rather than the 20 estimated here, and is the same at
  every signal level that decodes at all. `StereoDecoder` now fades the
  difference channel out between 20 dB and 11 dB of smoothed pilot margin and
  reports mono when it reaches the bottom. See **Amendment 13**.
- **Group 4A clock time, and alternate frequencies (0A blocks C/D).** Both are
  small additions to `RdsDecoder.update`.
- **POCSAG off air.** It shipped, but only against synthetic transmissions —
  see **Amendment 11** for the one-line hardware check it still needs.
  Favourites, recently played and session history shipped on 2026-08-29;
  what is left of them is listed in **Amendment 12**.
- ~~**A map for the aircraft screen.**~~ Shipped on 2026-08-29.
  `ui/widgets/planemap.py` draws the aircraft, their trails, a graticule, a
  scale bar and - since later the same day - the coastlines, lakes, state
  borders and cities of the United States, from 234 KB of public-domain
  Natural Earth vectors compiled into the package. Verified with real
  aircraft. What is left is a home position: the receiver does not know
  where it is, so there are no range rings and no bearings, and
  `_cpr_local` has no reference of its own either. See **Amendment 14**.
- **Aircraft heard while doing something else.** Tracking takes the radio
  over completely, which is honest but exclusive - a second dongle, or
  time-slicing against listening, is the only way round it and neither is
  worth doing yet.
- **RDS in the Discover list.** Not possible from a sweep — one group takes
  87.6 ms and a dwell is 50 ms — so it would need a second pass that parks on
  each candidate for a second or two. Worth doing, and a different feature
  from the sweep.

---

## Amendment 14 — A map with nothing under it (2026-08-29)

Positions were being decoded from the first day ADS-B shipped and shown as
two numbers on a card. Numbers are the wrong shape for a position: the whole
point of hearing six aircraft at once is the *arrangement* of them, which no
column of decimals will ever show.

### The thing this map does not have

Every map anybody has seen has a basemap under it, and the first version of
this one did not. A tile is a network request; a tile service is a dependency
that can go away, change its terms or want attribution, in an application
whose entire claim is that it works off a dongle and a laptop with nothing
else plugged in.

That argument turned out to be an argument against *tiles*, not against a
basemap - see the section at the end of this amendment, written the same day
after the size was actually measured. What it correctly rules out remains
ruled out: nothing is fetched while the app is running.

### The consequence, which is that there is no centre

The receiver does not know where it is. Nothing in ADS-B tells it, and asking
a beginner for their latitude before they can see anything is the
configuration screen this app exists to avoid. So the map has no fixed frame
of reference and frames itself on whatever it has heard.

That is honest and it works immediately, and it has two edges worth writing
down. A **single** aircraft has no extent at all, so fitting to the bounding
box divides by approximately nothing — hence `MIN_SPAN_NM`, a floor of six
miles across, which is about what a receiver hears when only one aircraft is
in range. And the fit has to stay **invertible**, so that a place on the
screen is a place on the earth: equirectangular about the centre of the view,
longitude squeezed by the cosine of the centre latitude. At Seattle's 47.6
degrees that squeeze is 0.674, and leaving it out draws a north-south airway
as a diagonal.

### Two smaller things, both the same shape

**A trail grows on a move, not on a frame.** An aircraft reports twice a
second and the screen refreshes five times a second, so appending on every
update is a list that grows without bound and draws a single point.
`update_trails` is a plain function tested without a window for the same
reason the colour maps are: what it gets wrong looks entirely normal for the
first minute and is wrong after ten.

**Longitude labels collide where latitude labels do not.** The graticule is
square in degrees, so its vertical lines are 1/cos(lat) closer together in
pixels than its horizontal ones — half as far again at this latitude — and a
spacing that reads comfortably down the side is a solid row of overlapping
text along the bottom. The lines are all drawn; only the labels are thinned.

### The basemap, added the same day

The objection above was to tiles. Asked what a *bundled* vector map would
cost, the honest answer needed a measurement rather than an intuition, so
the data was downloaded and encoded before anything was decided.

**The whole United States is 75,000 points.** Natural Earth at 1:10m,
clipped to the country, is 45,870 coastline points, 19,659 of lakes and
12,825 of state boundaries. Delta-encoded along each line as int16 steps on
an 11 metre grid and zlib'd, that is **234 KB including 482 cities with
their names and populations** - a fifth of the driver DLLs this repository
already commits on purpose, and 3.5% of the bundled nrsc5 binary. Against a
frozen build carrying PySide6, numpy and scipy it does not register. Size
was never the reason.

**Natural Earth rather than OpenStreetMap, and the reason is licensing.**
OSM data is ODbL: attribution on the map, and share-alike obligations on
anything derived from the database. Natural Earth is public domain, no
permission needed and no credit required - it is credited anyway. This
project already has one licensing boundary it has to be careful about, and
this is the version of that question that can simply be avoided.

**Steps rather than positions.** A coastline moves a few units at a time, so
storing steps makes the high byte of every int16 zero and that is what
compresses. Varints are 11 KB smaller (215 KB against 226) and were not
worth it: int16 decodes as `frombuffer` plus one `cumsum` per line, which is
**24 ms for the whole file**, once, at startup. The one thing it costs is
that a simplified border can be a single segment several degrees long, which
overflows an int16 - so the build puts points back into those.

**The obvious optimisation was the slow one.** The projection is affine, so
the tempting design is one path per layer built in degrees and a
`QTransform` per frame: build once, draw many. It measured **26 ms a frame
against 8**, because Qt then walks and clips the whole country's path every
time. Culling by bounding box in Python and rebuilding the path each frame
wins - and caching the *flattened* result against a window rounded outwards
to 0.05 degrees means the frame moving, which happens whenever an aircraft
does, costs nothing at all.

What is left of the cost is drawing: **4.3 ms at the widest view, 3 ms of it
antialiasing**, which at 5 Hz is 2% of a core and is worth paying. Building
the path from arrays is 0.17 ms; visiting the points from Python instead was
11.6 ms, which is what `pyqtgraph.functions.arrayToQPath` is there to avoid.

**A missing data file gives a plainer map, not an error.** The land is
decoration and must not be able to stop the aircraft being drawn - the same
rule as a corrupt settings file being replaced by defaults.

### What is left

**A home position.** It would buy range rings, a bearing to each aircraft, a
distance in the list, and a fixed frame that does not move when an aircraft
leaves — and it would give `_cpr_local` a reference of its own, which is
currently the aircraft's own last position and nothing else. It is a setting
and a marker, not a redesign. The basemap makes it less urgent than it was,
because a coastline answers "where am I" on its own.

**The map is only the United States.** A second region is a second `.bsm`
and a rebuild, exactly as a second band plan is a second YAML - but nothing
selects one yet, and an aircraft heard in Europe would be drawn over empty
space with a correct graticule around it.

**Panning and zooming.** The projection is invertible and the fit is one
function, so both are small; what stops it being obvious is that an
auto-framing map and a hand-driven one need different rules about when to
re-fit, and getting that wrong is a map that fights the mouse.

**Nothing is clickable.** Selecting an aircraft on the map and having its
card scroll into view is the natural next join between the two halves of the
screen.

---

## Amendment 13 — Blending to mono, and an asymmetry borrowed from the wrong place (2026-08-29)

FM stereo shipped in Phase 4 working and unqualified: lock on every local
station, 61 dB of separation on a synthetic broadcast. What it did not do was
notice that stereo is not always worth having. The difference channel sits at
23–53 kHz, where FM noise rises as the square of the frequency, so a fringe
station is *noisier* in stereo than in mono — and the receiver was choosing
stereo whenever a pilot was present, which on a distant station is exactly
when it should not.

### What the penalty actually is

Estimated at 20 dB in the Phase 4 plan; **measured at 15.4 dB**, by sweeping a
synthetic stereo broadcast through the real WFM demodulator with complex noise
added at the antenna. The number barely moves: 15.2 dB on a clean carrier,
15.5 dB at the FM threshold. It is a property of where the difference channel
sits, not of the station.

That has a consequence for how the blend can be decided. Comparing the noise
in the two channels always returns the same 15 dB, so it says nothing about
whether this particular station needs blending. What matters is the *absolute*
noise in the difference channel, and the pilot is the reference that measures
it: a 10% pilot is fixed by the standard, and the guard band immediately below
it is the same noise the difference channel is about to be built from. Over a
27 dB sweep the pilot-to-guard margin tracked the carrier-to-noise ratio to
within a couple of dB.

So the two thresholds are measured rather than chosen. At **20 dB** of margin
the difference channel is carrying about 12 dB of signal-to-noise and is
still worth its cost; by **11 dB** it is carrying none at all, the carrier
being at the FM threshold. Between them the weight is a straight line.

### The bug, which is the interesting part

The first version rode the margin down quickly and back slowly — the audio
AGC's attack and release, which is right there and wrong here. A single
block's margin swings several dB whatever the signal is doing, for the same
reason a single periodogram bin does, and an asymmetric smoother on a noisy
estimate does not average it: it parks at whichever extreme it reaches
faster. A **clean, noiseless** synthetic broadcast came out at **blend 0.70
and 12 dB of separation**, where it should have been 1.00 and 33.

The fix is to average the two powers symmetrically over half a second first,
and read the blend straight off the smoothed margin — 1.00 and 33.7 dB. The
fast path is not needed for the case it was there for: a signal that
genuinely collapses trips the lock's own hysteresis, which drops the
difference channel outright and is the honest answer to a station that has
gone away.

### Two smaller things

**The weight is a ramp across the block, not a number for it.** A gain that
steps between blocks is a click once per block — the same class of fault as a
filter that forgets its history — so each block ramps from the previous
weight to the new one. It short-circuits to a scalar 1.0 when both ends are
1.0, which is every block of a strong station, so blending costs nothing on
the signals that do not need it: 1.5 ms per second of radio when it is
working, none when it is not.

**Fully blended has to be reported as mono.** At weight 0 the two channels are
identical, and a lit STEREO badge over them is the receiver claiming something
it is not doing. The difference comes back as `None` — the same answer as no
pilot at all — and the audio returns to a single channel. The station is
still *locked*, which is a different question, and `pilot_db` still answers
it honestly.

### What is left

The blend is a whole-band weight. Real tuners also roll the *top* off the
difference channel first, because that is where its noise is worst, which
keeps some separation on a station this fades to mono. And it is a switch at
Expert rather than a slider: somebody who wants stereo on a fringe station can
have it, but cannot yet say where the fade should start.

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

---

## Amendment 11 — POCSAG, and three ways a bit clock can be right for the wrong reason (2026-08-29)

POCSAG is the simplest signal in the app — two frequencies, no carrier to
recover, no constellation to rotate — and that is exactly what makes it easy
to build something that decodes the test and nothing else. Every finding below
is about the gap between "the bits come out" and "the bits come out whatever
the air does".

### The design

The chain is short. The FM discriminator already produces the frequency
deviation; POCSAG *is* that deviation, so there is no subcarrier to mix down
and no filter in front of the slicer. What there is instead is a bit clock, a
frame finder and a checkword, and each of the three has a decision in it.

**The bit clock is the RDS timing loop again, and for the same reason.** There
is no whole number of samples per bit at any rate the radio produces — 512 bps
at a 96 kHz IF is 187.5 — so the bit instants are a floating-point ramp and
the value of each bit is the *integral across it*, read as the difference of a
running sum at two fractional positions with `np.interp`. That integral is the
matched filter for a rectangular symbol, which is why nothing filters ahead of
it; and the half-sample correction is the one ADS-B needed, for the same
reason. Timing is steered the way RDS steers its symbol clock: read the same
block at sixteen positions across the bit, take the one where the readings are
largest, and move a third of the way towards it. A clock half a bit out
averages every transition away, so that metric has one maximum by
construction.

**All three baud rates run at once.** The alternative is detecting the rate
from the preamble, which throws away every transmission whose preamble was
missed — and a transmission whose preamble was missed is exactly the one a
user tuning across the band arrives in the middle of. Three vectorised passes
cost 1.3% of a core between them, which is less than deciding cleverly would
be worth.

**Frame sync is stateless.** Every batch opens with the same 32-bit sync
codeword and a batch is 544 bits, so rather than acquire and hold a lock the
receiver looks for that codeword everywhere and decodes the 544 bits behind
each hit. Allowing two bit errors in the match, a false hit turns up once in
eight million bit positions — an hour of a 2400 bps channel — and the sixteen
codewords behind it still have to pass their own checkwords. Holding a lock
would buy nothing and would need a recovery path of its own.

### The three that matter

**A message that fills its last batch exactly has nothing to end it.** A page
is terminated by an idle codeword or by the next address codeword, and a
transmitter that stops on a batch boundary sends neither — so the message sits
half-assembled and never reaches the screen. Two of the first synthetic
transmissions written hit this, which is a fair sign of how common it is. The
fix is a staleness rule expressed in *bits*: a message is over once a whole
batch's worth of the stream has gone by with no batch in it. It cannot be
shorter than that, because a batch's worth of bits is exactly how long it
takes to know that a batch which *would* have been adjacent is not there. On
air that is half a second at 1200 bps, which is about as fast as a message
could honestly be declared over.

**One bit of correction, not two.** BCH(31,21) has a minimum distance of six
and can correct two errors, which is what most pager decoders do. It also
means three errors can land within two of the *wrong* codeword, and a
mis-corrected address codeword is somebody's message shown against somebody
else's pager number. Correcting one error needs five to go wrong the same way.
This is the RDS rule — "a mis-corrected block puts plausible wrong text on the
screen" — applied at the one setting where refusing to correct at all would
cost most of the traffic rather than a little of it.

**Two decoders, two taps.** RDS and POCSAG both want the same point in the FM
path: after the discriminator, before de-emphasis and before the audio filter.
It is tempting to give them one slot, because their attach conditions are
mutually exclusive — RDS needs a 200 kHz broadcast channel and POCSAG a
12.5 kHz two-way one, so the two are never wanted at once. But one slot means
whichever attached last silently detaches the other, and the failure that
produces is a feature that simply stopped working with nothing on screen to
say why. `_FmBase.data_sink` is separate from `mpx_sink` for that reason and
no other.

### Which way round the bits are

Three orderings in this protocol run against the obvious reading, and each of
them produces output rather than an error.

- **The deviation's polarity is not fixed.** Which way a transmitter deviates
  for a binary one, and which side of the tuner the channel landed on, both
  flip it. So the sync codeword is matched against *and* against its inverse,
  and the batch behind an inverted match is inverted before it is read.
- **Alphanumeric characters arrive least significant bit first**, seven bits
  at a time, packed across codeword boundaries with no alignment to them.
- **Numeric digits arrive least significant bit first too**, four bits at a
  time — so the character table is indexed by the nibble as transmitted and
  the reversal is baked into the table rather than done at every lookup.

And one that is not about bit order at all: **the bottom three bits of a
capcode are never transmitted.** They are which of the eight frames of the
batch the address codeword was placed in, which is how a pager can listen to
one eighth of the traffic and sleep through the rest. A decoder that ignores
that gets every capcode wrong by up to seven and looks almost right.

### Cost

Measured over three seconds of signal, per second of radio:

| IF rate | Traffic | Noise only |
|---|---|---|
| 96 kHz (2.4 MS/s window, NFM at 20 kHz) | 12.0 ms | 13.6 ms |
| 240 kHz (240 kS/s window) | 12.9 ms | 13.6 ms |

1.2–1.4% of one core either way, against the NFM demodulator feeding it and
RDS's 2.4%. It costs slightly *more* on noise than on traffic, which is the
right way round: the expensive part is the three timing loops, which run
whatever is there, and the cheap part is checking codewords, which only
happens behind a sync word.

### On screen

The panel lives under the waterfall rather than beside it, and it stays hidden
until a sync codeword has actually been seen. Two reasons. A panel that
appears on every narrow FM channel and never says anything teaches a beginner
to ignore that part of the screen; and the moment it does appear is itself the
finding — "there is pager traffic here" — which is the same argument the
Discover list makes. Rows are appended rather than rebuilt, because a message
does not change once it is decoded and rebuilding the list thirty times a
second would throw away the reader's scroll position before they finished a
sentence. `_forget_station` clears it: pager traffic belongs to a channel, and
carrying it to the next frequency would put somebody else's messages under a
heading that says they were heard here.

`scan/bandplan/us.yaml` gains **Pagers, 929–932 MHz**, as a scan target with a
25 kHz raster and a 20 kHz channel filter — wider than the 12.5 kHz two-way
default, because a pager deviates ±4.5 kHz and a filter sized for speech takes
the corners off the bits.

### What has been tested, and what has not

Twenty-eight tests against a synthetic transmitter written independently of
the decoder — its checkword is long division against the polynomial, not a
call to the receiver's own routine. They cover all three baud rates, six
arrival phases across a bit, inverted deviation, a quarter of full deviation
of tuning error, a transmitter clock 100 ppm out, noise, four DSP block sizes
from 256 to 32,768 samples, a message spanning three batches, eight capcodes
landing in all eight frames, and the whole path through the NFM demodulator at
2.4 MS/s. Noise alone and silence both decode to nothing.

**None of it has met a real transmitter.** That is the position ADS-B was in
before 2026-08-28, and the ADS-B experience is why it is worth saying plainly:
the surface-versus-airborne CPR fault was invisible to every synthetic test
and turned up in the first ninety seconds of real sky. The hardware check here
is one line — tune to a busy channel in 929–932 MHz and watch the panel — and
until it has been done, this feature is built rather than verified.

---

## Amendment 12 — Favourites, recently played, and what counts as listening (2026-08-29)

This is the smallest feature in Phase 4 and the one with the most ways to be
quietly wrong, because nothing about it fails loudly. A history that records
the wrong things still shows a list; a list of the wrong things is just a
worse app that nobody can point at.

### The one decision everything else follows from

**Tuning across a band is not listening to it.** The digit tuner emits a
frequency per keystroke and click-to-tune emits one per click, so a history
fed from "where is the radio" records the journey rather than the
destinations — walking up the FM dial would leave eleven entries and no
station. So a visit becomes an entry only after it has lasted, which is the
scanner's persistence gate in another costume: seen once is not seen.

That threshold is not a taste setting, because it has a floor and the floor
is a measurement from elsewhere in this document. RDS needs a few seconds to
confirm a name and HD Radio needs 5.5 to produce its first audio, so a dwell
shorter than either would promote every entry *before anything could name
it*, and the recent list would be a column of bare frequencies — which is
precisely the screen this project exists to replace. Ten seconds clears both
with room. It is also why a name arrives through `History.name()` afterwards
rather than as an argument to `tune()`: at the moment the radio moves, nothing
on air has said anything yet.

### Time is accrued, not read off the clock

`update()` is called from a view's own refresh timer, and `main_window`
stops a page when it stops showing. So a station left playing behind the
Discover screen accrues nothing while a sweep runs. That is deliberate and it
is the honest reading of "played": it counts the time somebody was actually
listening, not the time the tuner sat there. `MAX_TICK_SECONDS` covers the
other half — a window minimised for an hour must not come back and claim the
hour — and it is the same shape of guard as the one on a rate change.

Two consequences worth naming. Leaving the listening screen calls `update()`
and not `leave()`, so coming back two seconds later continues the visit
instead of starting a second one and demanding another ten seconds. And
retuning *inside* the bookmark match tolerance continues the visit too:
without that, a station corrected by 100 Hz every few minutes could be
listened to all evening and never recorded at all.

### A favourite is a bookmark marked, not a second list

`Bookmark` gained a `favourite` flag rather than there being a favourites
collection. A separate collection is two records of one station, and the
moment they disagree — a different mode, a frequency edited in one place —
the favourite recalls something the saved entry says is wrong. This is the
same argument `bookmarks.py` already makes for keeping the mode and bandwidth
on the bookmark. The flag round-trips through CSV as `yes`, because the column
is meant to be readable in a spreadsheet and typeable by hand.

The strip on Discover takes favourites before it applies its limit, so
somebody who stars nine stations gets nine chips. Quietly dropping the ninth
in favour of something they heard once would be the app overruling the only
explicit statement of preference it has. A station that is both a favourite
and recently played appears once, as the favourite.

### Where it is, and why there

The strip sits **above the band chips** on Discover, because on the second run
of the app it is a shorter route to hearing something than any sweep is — and
it hides itself entirely until there is something in it, so a first run sees
exactly the screen it saw before this existed. Same argument as the pager log:
the moment it appears is itself worth something.

The Recently played section on the listening screen is at **Simple**, which is
the level with no mode control, no bandwidth and no gain. A beginner who tunes
away from something they were enjoying previously had no way back to it at
all. It is the one panel section that is more use the less of the app somebody
understands.

### What is left

- **The trail is not persisted, only the recent list is.** Back is a
  within-session idea, and a Back button that jumped to where the radio was
  last Tuesday would be a different feature wearing the same label.
- **Nothing prunes by age.** The list is capped at sixty by last-heard, so a
  station heard once a year ago survives until sixty others displace it.
  Whether that is a fault depends on how the list reads after a few months of
  use, which is not knowable yet.
- **`most_played` is computed and not shown anywhere.** It is a different and
  probably better ordering for the strip than "most recent", and which one is
  right is a question about how somebody actually uses the app rather than one
  to be settled here.

---

## Amendment 14 — Channel names, and what to say where there is nothing to hear (2026-08-29)

Two complaints about the same screen. The listening header read **"Marine
VHF — Boats talking to each other, to harbours, and to the coastguard"** at
156.800 MHz, which is where everybody on the water says **Channel 16**; and
across roughly half of the tunable dial it read **"Nothing is normally
broadcast here"**, which for the 700 MHz mobile phone band is not modesty but
a false statement.

Both are answered with data rather than code: a `channels` list inside a band,
and a second top-level `allocations` list for the space between the bands.
159 channels and 41 allocations, about 1,300 lines of YAML, and no per-band
code path anywhere.

### A channel has two names, and which one is shown is a level decision

`Channel(name, frequency_hz, use, official)`. `name` is what somebody would
say out loud — "Channel 16", "WX1", "Guard". `official` is the designation in
the rule book — "International Distress, Safety and Calling" — which is the
phrase to search for and the one printed on a chart. `use` is the plain
English, held to the same standard as `Band.description`.

The friendly name is drawn on the ribbon at **every** level, because a
channel number is not an expert feature; a beginner tuning across the marine
band is exactly who needs it. The regulator's phrase is appended to the
header's prose from **Standard** upwards, and so is the licensed use of a
stretch of dial no band covers. This is the progressive-disclosure rule that
already governs the control panel, applied to words instead of widgets:
nothing is removed at a lower level, it is only quiet until asked for.

`band_headline(hz, level)` is a plain function in `ui/listen_view.py` and
returns the two strings the header needs. The interesting part is the wording
and the gating, not the labels it ends up in, so it is tested without a
window — same argument as the colour maps and the digit arithmetic.

### A channel claims half a raster, and no more

Marine channels sit shoulder to shoulder on a 25 kHz raster, so every dial
position inside the band belongs to one of them and the name has to change the
moment the dial crosses the halfway point. The airband is the opposite: eight
frequencies everybody knows out of a thousand ordinary tower and approach
channels, so 118.300 MHz has to come back with **nothing** rather than
borrowing the name of the nearest one. Half the raster where a band has one,
half the channel width where it does not, serves both.

### The lists are listed, not counted

Every one of these would be wrong if the channel numbers were derived from the
raster:

- **CB channel 23 sits above 24 and 25**, at 27.255 MHz, and has since 1977.
- **Marine channels carry an A** where the US uses the ship half of an
  international duplex pair as a simplex channel — 18A, 22A, 79A — and the
  shore halves are 4.6 MHz up, in the same band, under the same numbers.
- **NOAA numbers its seven weather channels** WX1 at 162.550 down to WX7 at
  162.525, which is neither frequency order nor anything else.
- **FRS and GMRS interleave**: channels 1–7 shared, 8–14 low power on the
  467 MHz interstitials, 15–22 on the 25 kHz grid, and eight repeater inputs
  5 MHz above the last of them.

A test asserts that **every channel lands on its own band's raster**, because
`snap` and `channel` are two answers to the same question. A channel half a
step off would be clickable and then unnameable — the NOAA-at-162.537 fault
from "Channel rasters are per-band data", in a new place.

### Why the allocations are not bands

A `Band` is a promise: the app has something to offer here, so it carries a
mode, a bandwidth, a raster, a colour on the ribbon and possibly a scan chip.
None of that is true of the 600 MHz mobile phone band. Making these `Band`s
would have put twenty new stripes on the ribbon, offered the classifier
priors it cannot use, and — worst — handed `_apply_band_defaults` a mode to
switch the demodulator to on the way past. So they are a separate list with
prose and nothing else, read by one function, only where `find` came back
empty, and only from Standard up. At Simple, a stretch of dial with nothing to
listen to is better left quiet than explained.

### The coverage test found two gaps nobody would have thought of

A test merges the bands into intervals, walks the gaps between them across the
whole 500 kHz – 1.766 GHz tuning range, and demands an entry for each. Two of
them were invisible by eye:

- **117.975–118.000 MHz**, the 25 kHz guard band between the navigation
  beacons and the air traffic channels. It is 25 kHz wide and the app can tune
  into it.
- **162.025–162.400 MHz**, between the top of the marine band and the bottom
  of the weather channels — federal government, and the pool the weather
  channels themselves come out of.

The same test is what will catch a future band being added with an allocation
left overlapping it, and a second test asserts the two lists never both answer
for the same frequency.

### Also

The bookmark a channel is saved from is now named **"Channel 16"** rather than
"Marine VHF", where the station has not named itself — the band is already the
group it is filed under, so repeating it would have been the only thing on the
row that said nothing.

### The names belong on the spectrum, not above it

The first version put all of this in the header: a chip reading **Channel 16**
next to the band name, and the licensed use as prose underneath. It was
correct and it was in the wrong place. A frequency is a position, and the
ribbon is the only part of the screen that draws positions - a name up in the
header says what you are on, and a name on the ribbon says that *and* how wide
the channel is, where its edges are, and what is either side of it.

So the ribbon grew a second lane. The top one is the band plan it always drew,
and from Standard up it now also blocks in and names the stretches no band
covers, so it stops going blank over half the dial. The bottom lane appears
only when the band under the cursor has named channels, and it is a real grid
of them once a channel is at least 1.2% of the window - 96 marine channels in
a 2.4 MHz window is a texture, not a ruler. Below that the grid stays away and
**the channel being listened to is drawn anyway**, which is exactly the
window where somebody most needs telling they are on Channel 16.

`channel_cells` is a plain function, tested without a window, for the same
reason the colour maps and the digit arithmetic are.

### Whether a name fits is a question about a font, so it is asked of the font

The labels were first placed with a fraction-of-the-window rule - draw the
name if its block is more than some percentage of the span. Rendering seven
situations to a PNG and reading them back showed what that cannot see: at
162.550 MHz the ribbon read **"Federal governmentWeather Radio"**, two names
260 kHz apart in a 2.4 MHz window, because a fraction rule never compares two
labels with each other. It also has no idea that "Federal government" and
"2 m" are different lengths, so elsewhere it was hiding short names that
fitted perfectly well.

Names are now measured with `QFontMetrics`, in the font they will actually be
drawn in, against the view box's real width. They are collected first and
drawn last, because whether one can be drawn depends on the others, and
`without_collisions` - pure, and tested - decides by **rank**: the band, the
allocation or the channel the receiver is on outranks a neighbour, always.
The one the fraction rule dropped at 162.550 was the band being listened to,
which is the priority exactly backwards.

Two consequences fall out of that rule and both are deliberate:

- **The tuned name is not measured against its own block.** Weather Radio is
  150 kHz of a 2.4 MHz window and its name is half as wide again as its
  stripe. Every other name that overhangs its block is a name for its
  neighbour and is dropped; that one is the name the user is looking for.
- **It is therefore the only name drawn on a backing.** A label allowed to
  overhang has the block's own edges drawn through the middle of its letters,
  which is what a 25 kHz cell in a 2.4 MHz window does to "Channel 16".

The two lanes are resolved separately, since names on different rows are not a
collision, and the widget's width is part of the redraw cache key - what fits
depends on it, so a resize that did not redraw would leave labels touching.

### What is left

- **Discover cards do not name the channel either.** A card reading "Marine
  radio - 156.800 MHz" could say "Channel 16", and the classifier already has
  the band in its hand when it writes that line.
- **The two halves of a duplex channel are listed but not paired.** The shore
  side of channel 24 is in the list as "Channel 24 (shore)" and the app has no
  idea it is the other end of the same conversation.
- **The channel lists are as US-specific as the bands are.** A European file
  would have marine channels without the A suffixes, no FRS or GMRS at all,
  and PMR446 instead — which is what `region` is for, and it stays a second
  file rather than a second code path.

## Amendment 15 — Zoom, pan and click-to-tune on the spectrum (2026-08-29)

Three requests about the same picture: a **width zoom** on the waterfall and
the spectrum, **horizontal panning** for scanning across a band by eye, and
**click-to-tune on the spectrum** the way the waterfall already does it. All
three are SDR# parity and none of them needs anything from the radio.

### The window and the view are now two different things

Until now the display *was* the window: `sample_rate` hertz, centred on the
tuned frequency, drawn edge to edge. That made "what is sitting next to this
signal?" a question the app could only answer by narrowing the window, which
changes what the receiver is doing rather than what the screen is showing —
and at 2.4 MS/s a 12.5 kHz marine channel is a fifth of one pixel on a
thousand-pixel pane, so it was a question worth being able to ask.

`ui/widgets/viewspan.py` holds the whole of it, and it is two numbers:

    zoom    how many times narrower than the window the view is, never < 1
    offset  where the middle of the view sits, as a fraction of the whole
            window away from its middle

Fractions of the window rather than hertz, so a retune or a change of window
width leaves them meaning the same thing afterwards. Nothing else changes: the
same transform, the same rows of waterfall history, the same one device call
per tune. The arithmetic is pure and tested, and it lives in one place because
**three stacked panes have to agree on it exactly** — a waterfall showing a
slightly different span from the spectrum above it puts every frequency wrong,
which is the fault `AXIS_WIDTH` exists to prevent one layer further down.

### The middle of the picture stopped being the tuned frequency

The band ribbon derived the tuned frequency from the span it was handed, on
the reasoning that the window is always drawn centred on where the radio is
pointed. Panning ends that: the middle of a panned view is wherever the user
dragged to, and the ribbon would have highlighted whichever channel happened
to be in the centre of the screen rather than the one being listened to —
which is the single piece of information that lane exists to carry.
`set_span` takes the tuned frequency explicitly now.

### A pan is about the window it was made in, so a retune discards it

Zoom is a standing preference and survives everything. The offset does not:
it is a fraction, so it survives a retune *arithmetically* and would then be
pointing a zoomed pane several channels away from the station the user had
just asked to hear, because the window moved underneath it. `_sync_view`
re-centres on any change of `center_hz`, which covers click-to-tune, the digit
readout, bookmarks, the Discover step buttons and the way back from an
aircraft excursion, in one place rather than in seven.

### "Fit to what is on screen" had to start meaning it

The automatic fit and the button are the same measurement, and both took the
whole transform. Zoomed into a quiet stretch beside a broadcast station, that
sets the ceiling from a signal the user cannot see and flattens everything
they can. Both take the visible slice now.

### The passband gave up its body and kept its edges

A single click on the spectrum used to be reserved for dragging the passband,
which is why tuning there needed a double click. Clicking says where to listen
far more directly than dragging a block does, so the body of the passband is
no longer movable — and it could not have stayed, because at 8x zoom a
broadcast passband fills the pane and would have swallowed every press meant
for the spectrum behind it.

The edges are still handles, and the hit test for them is in **pixels, not
hertz**: a passband is 200 kHz wide on a broadcast station and 500 Hz on a CW
signal, and a handle has to be the same size under the pointer either way.
Dragging one now changes the bandwidth and *only* the bandwidth — it is
measured from the frequency being listened to rather than from the other edge,
so the passband stays centred where the radio is. Before, an edge drag moved
the centre as well and retuned, which dragged the passband out from under the
pointer halfway through setting a filter width.

### Not calling `super()` is the mechanism, not an omission

pyqtgraph's scene turns a press it has seen into click and drag events for the
items under it. A pan that let the scene see its own press would drag the
passband along with the view. So the press is consumed outright unless it
lands on a passband edge, in which case it is passed straight through and
pyqtgraph handles the whole gesture as it always did. Moves with nothing held
down are always forwarded, or the edges stop lighting up under the pointer.

### A click is a drag that did not go anywhere

Four pixels of slop. Not zero: a mouse moves a pixel or two under a real
finger, and a click that silently became a one-pixel pan would read as
click-to-tune being broken. The pan itself is measured against the frequency
the drag *started* on rather than accumulated from the last move, so what the
clamp discards at the edge of the window is discarded rather than stored —
dragging back off the edge moves the view on the first pixel instead of after
undoing however far the pointer travelled past the end.

### The zoom slider is the one Display row that appears at Simple

Everything else in that section is a Standard control and stays one. The wheel
is the reason: somebody who scrolls over the spectrum by accident has zoomed
in, and at Simple there would otherwise be nothing on screen to say what
happened or how to undo it. It is plain English, it cannot affect what the
radio is doing, and it doubles as where the two gestures are explained.

Its travel is logarithmic. A linear slider spends its first half between 1x
and 32x and its second half between 32x and 64x, which is one useful step and
a hundred useless ones.

**Zoom is deliberately not remembered between sittings.** Opening on a 37 kHz
sliver of the FM band looks like a broken radio, and the display settings that
are restored are restored precisely because none of them can look like a
fault.

### Where the limits come from

`MAX_ZOOM` is 64, which fills the pane with one 12.5 kHz channel out of a
2.4 MHz window — the narrowest thing the app has to show against the widest
window it opens. Past that the transform runs out of bins before the eye runs
out of detail: 4096 bins across 2.4 MHz is 586 Hz each, so a 37 kHz view is
already only 64 of them, and zooming further magnifies the FFT rather than the
radio. A wheel notch is 1.2, which is 23 notches end to end.

### What is left

- **Not yet seen on hardware.** The arithmetic is covered by tests and the
  gestures were driven through the real widgets offscreen at all three levels,
  but nobody has yet dragged across the FM band with a dongle plugged in.
- **The waterfall magnifies its own pixels.** At 64x the history is 64 bins
  across the pane, so the rows go blocky. That is honest about the resolution
  of the transform and the answer is the FFT size control, but a receiver that
  raised the resolution itself as the view narrowed would be better.
- **Discover has no spectrum to zoom.** The panes belong to the listening
  screen; a zoomed sweep view is a different feature.

## Amendment 16 — The Learn tab, and a control that explains itself (2026-08-29)

A fourth screen beside Discover, Listen and Aircraft, and one new gesture that
matters more than the screen does: **clicking the name of a control takes you
to what it means.**

### Why this is the second half of a principle already stated

The project runs on two principles, and until now the second one — *the app
explains itself* — was spent entirely on the classifier. Every `Signal`
carries `reasons`; every card can say "constant power, 150 kHz wide, sits in
the 88–108 MHz broadcast band". That is genuinely the product, and it explains
exactly one thing: what the app just found.

It does not explain the app. Phase 3 took the listening screen from eight
controls to about forty, and forty controls called Squelch, De-emphasis,
Filter edge, Offset tuning and IQ imbalance are friendly only to somebody who
already knows what those words mean — which is precisely the person who did
not need this application. A beginner-facing radio with forty undefined terms
on one screen has quietly capped itself at the users it was built to replace.

So the same argument that produced `reasons` produces this. The difference is
only what is being explained.

### The route that matters is not the tab

A Learn tab reached only by pressing Learn is a manual, and nobody opens a
manual. The moment an explanation is wanted is the moment somebody is looking
at a row called "Threshold", does not know what a threshold is, and wonders.
That is the only moment the app can be certain of, and the only sane thing to
do with it is make the word itself the way in.

Hence `topic=` at the row's own call site, one word beside `level`:

    section.add("RF gain", self.gain, topic="rf-gain")
    section.add_wide(self.squelch_on, topic="squelch")

It has to live there and not in a lookup table keyed on the caption, because
**two rows are called "Threshold" and they are not the same threshold** — one
is the squelch, one is the audio gain rider. Two more are called "Depth", and
two are called "Window", one of which is the FFT taper and the other the width
of dial the dongle captures. A table keyed on what the label says would have
been right about most of them and confidently wrong about six.

Fifty-six rows on the listening screen carry a topic; two on Discover do.

### Two shapes, because a check box carries its own text

A labelled row's caption becomes the link — the caption is what the reader is
looking at and puzzled by. A check box already carries its text and clicking
that text has to keep toggling it, so those rows get a small question mark
beside them instead. `widgets/help.py` holds both, and the panel gathers every
one of them onto a single `helpRequested` so the view above connects once
rather than fifty-six times.

The resting colour is exactly the ordinary caption colour, and only hover
lights up. Fifty-six bright links down a control column would compete with the
controls. The dotted underline is always drawn and always the caption's own
colour, so nothing moves between resting and hovered — a label that *grew* an
underline on hover would nudge the field beside it by a pixel, on every row of
a scrolling column.

### Nothing may look clickable and then do nothing

This is the one failure the feature can have that nobody would ever report. A
caption pointing at a topic with no article falls back to a plain `QLabel`:
the control looks completely normal and simply never offers an explanation
again. Rename a slug and forty captions go quiet at once, with no error, no
crash and nothing on screen to notice.

So the guard is in three places and a test:

- `label_for` asks `learn.has()` before making anything a link;
- an inline `[[slug]]` with nothing behind it renders as **plain prose**, not
  as a dead anchor — a dead anchor survives review because the sentence still
  reads correctly and fails only under a cursor;
- a see-also chip for a missing article is not drawn at all;
- `tests/test_learn.py` reads every `topic="..."` in `ui/` out of the source
  and asserts each one has an article. Reading the source rather than building
  a `ControlPanel` is deliberate: the string in the source is the thing that
  has to be right, and checking it needs neither a Qt application nor an
  `Engine`.

### The content is data, and its order is content

`ui/learn/glossary.yaml` is 72 articles under eight headings, and the headings
are in the order a beginner meets them — Start here, Listening, Seeing the
signal, The receiver, Cleaning up the sound, Finding things, Decoding,
Recording and saving. That ordering is a claim about what to read first, so
the home page **browses by category rather than offering an A-Z**: an
alphabetical index only helps somebody who already knows the vocabulary, which
is the exact thing the reader does not have.

Each article carries a one-sentence summary that has to stand alone (it is all
the browse list shows), the other names the thing goes by, where in the app
the control actually is, and cross-references. Same bargain as the band plan
and the basemap: a rewrite for a different audience, or a second language, is
a second file and not a second code path.

### Two decisions inside the search box

**Aliases are the whole reason it is not a title match.** Nobody looking up
"capcode" knows the article is called POCSAG, and nobody who read "SNR" on a
forum knows this app spells it out. Every article carries the other names it
goes by, and they are searched and shown.

**The exact-match bonus is judged against the whole query, never one word of
it**, and that was a bug found by a test rather than by reasoning. Awarding it
per term let the article whose *slug* is "stereo" beat the one actually called
"Stereo blend" on the query `stereo blend` — the first collected a
whole-article bonus for half of what was typed plus a passing mention of the
other half. An exact match is a claim about what somebody asked for, so it has
to be measured against all of it.

There is also a fallback: every word must match, *unless that finds nothing*,
in which case any word will do. People type questions into search boxes — "why
is my audio quiet" — and requiring "why", "is" and "my" all to appear answers
a perfectly reasonable question with a blank page, which reads as a glossary
that does not cover it.

### Back means two different things, and both are right

An article opened by clicking a control gets a **Back** button that returns to
the screen the control was on; one reached by browsing gets **All topics** and
returns to the home page. Following a link inside an article keeps whichever
journey the reader was on, so somebody who arrived from a control and read two
cross-references deep still gets their control back.

Pressing the Learn tab itself always lands on the home page, even from an open
article. Somebody reaching for the tab has a browsing question rather than the
one specific question that brought them here from a control, and an article
left over from twenty minutes ago is a Learn tab that appears to contain one
entry.

### Nothing here is level-gated, which is an exception on purpose

Every other part of the UI hides what belongs to a higher level. This part
must not. **Levels decide what you may change, never what you may
understand.** Somebody in Simple mode who has read the words "RF gain"
somewhere and wants to know what they mean is exactly the reader this exists
for, and the fact that they cannot yet *see* that control is not a reason to
withhold the explanation of it. `set_level` is accepted and stored and gates
nothing.

### What it cost

Nothing measurable. The screen has no timer, touches no device and does not
poll — it is the only page in the app with nothing live on it, so `start` and
`stop` exist purely to satisfy the protocol the window expects. The content
file is 55 KB of YAML parsed once and cached. A `HelpLabel` is a `QLabel` with
two extra event handlers.

### Verified, 2026-08-29

Driven through the real widgets offscreen, with a stub engine:

- the panel raises `helpRequested` from both shapes of row, and reports only
  the topics that have articles behind them;
- all 72 articles render, every see-also chip resolves, and every inline link
  in the file points at something that exists;
- clicking **RF gain** on the listening screen and **Sensitivity** on Discover
  both open the right article and both come back to the right screen;
- pressing the Learn tab from an open article shows the home page with the
  search box cleared;
- level changes reach the page and hide nothing.

824 tests pass and `ruff check .` is clean.

### What is left

- **Nobody has read it yet.** Seventy-two articles written in one sitting are
  seventy-two chances to have explained something in terms of something else
  the reader also does not know. This is the ADS-B situation again: every
  synthetic test passed and the real sky found the fault. The check is one
  beginner and twenty minutes.
- **The Aircraft screen names no topics.** Its controls are a filter and a
  map, and `adsb` is written and browsable, but nothing on that screen links
  to it.
- **The no-radio screen has no Learn tab.** When the dongle is missing the
  nav is disabled wholesale, which is defensible — that screen is about
  fixing the driver — but Learn is the one page that needs no hardware at
  all, and a first-time user staring at a driver problem is not the worst
  audience for it.
- **The band plan and the glossary do not know about each other.** A band's
  description and an article about that band are written twice, in two files,
  in the same voice. Nothing is wrong yet; they will drift.
- **No article explains the Learn tab.** That is probably correct.

---

## Amendment 17 — Packaging, and a program Windows will not start (2026-08-30)

### What the plan said, and what happened

Phase 5 was three bullet points: a PyInstaller one-folder build, the driver
DLLs bundled into `_internal/drivers/`, and a README noting the SmartScreen
warning and the *More info → Run anyway* path a user clicks to get past it.

The build works. The bundle is complete. The warning it was written for no
longer exists, and what replaced it cannot be clicked past.

**Smart App Control blocks the executable outright.** It is on by default on
clean Windows 11 installs — this machine reports
`VerifiedAndReputablePolicyState: 1` and user-mode code integrity enforced —
and it refuses any binary that is neither signed by a publisher it recognises
nor already vouched for by Microsoft's reputation service. A freshly built,
unsigned PyInstaller executable is neither. The process never starts. What the
user gets is a dialog with no way forward; what the developer gets, if they
know to look, is this:

```
Microsoft-Windows-CodeIntegrity/Operational, event 3077
  ...attempted to load ...\dist\BetterSDR\BetterSDR-Tools.exe that did not
  meet the Enterprise signing level requirements or violated code integrity
  policy (Policy ID:{0283ac0f-fff1-49ae-ada1-8a933130cad6})
```

This is not the SmartScreen reputation prompt the plan anticipated, and no
amount of waiting fixes it. A widely distributed file can earn reputation; a
build somebody made this morning cannot, and never will unless it is signed.

### The decision

**The supported way to get BetterSDR is to clone the repository and run one
command.** `py tools/setup.py` builds the virtual environment, installs the
dependencies, runs the driver check and opens the app; running it again later
is how the app is started, and the second run takes 1.7 seconds.
`BetterSDR.cmd` at the root is the same thing for somebody who would rather
double-click than type.

This works because a Python interpreter is signed and recognised, and
everything it installs arrives as an ordinary package from PyPI — files
Microsoft's reputation service has seen millions of times. Nothing new and
unsigned is ever executed. The cost is a stated prerequisite: Python 3.12 or
newer, which the setup script checks for first and explains rather than
assumes.

**The PyInstaller build stays, unverified.** `BetterSDR.spec` and
`tools/build_app.py` are complete and produce a correct bundle; what cannot be
established on this machine is that the bundle *starts*. It is kept because
the work is done, because the blocking condition is a Windows policy rather
than a defect, and because signing would clear it — an EV code-signing
certificate, roughly $300–400 a year on a hardware token, is the only thing
that satisfies Smart App Control from day one. That is a decision about money
and identity, and it is the user's to make, not a packaging problem to solve.

### The three ways a frozen build failed silently

Every one of these produced a bundle that looked complete.

- **A frozen entry script has no package around it.** `bettersdr/app.py`
  imports its siblings relatively, and as a frozen `__main__` those resolve to
  nothing. PyInstaller's analysis followed them without a single warning and
  produced a 134 MB application containing **no part of BetterSDR at all** —
  which raised `ModuleNotFoundError: No module named 'bettersdr.core'` on the
  first line the user would have seen. The manifest check written to catch
  exactly this class of fault passed the broken bundle, because every file it
  knew to look for was present. Hence `packaging/bettersdr_app.py` and
  `packaging/bettersdr_tools.py`: two-line shims that import the package
  properly, and a build-time check that compares the modules in the bundle's
  archive against the modules on disk.
- **A lazily imported module is invisible to static analysis.** The console
  tools import their command at the moment one is asked for, which keeps Qt
  out of a driver check — and put `bettersdr.listen` beyond anything
  PyInstaller could follow. It shipped missing from a build in which all the
  other 70 modules were present. The spec now reads `diagnose.COMMANDS` for
  its hidden imports rather than restating the list, so a new command cannot
  ship as an executable that cannot find it.
- **`sys._MEIPASS` is where a frozen build's data is, and the executable's own
  folder is not.** `native.driver_dir()` already knew this; `hdradio` reached
  for `Path(sys.argv[0]).parent`, which in a one-folder build is a level above
  everything that was collected. It is now `_roots()`, a helper that exists so
  the answer can be tested rather than inferred.

### Verifying a program that cannot be run

`tools/check_bundle.py` is the answer to "the executable is blocked, so how do
we know any of this works". It unpacks the bundle's module archive over a copy
of its data folder, recreating on disk the tree a frozen process sees, and
then runs the application out of *that* with a bare interpreter and `-S -E`,
so the site directory is off and nothing can quietly come from the developer's
virtual environment instead.

What it establishes, on the bundle as built:

```
ok  the whole application imports: 71 modules
ok  dependencies load from the bundle: numpy 2.5.2, scipy 1.18.1, Qt 6.11.2
ok  the RTL-SDR driver is found: drivers\win-x64\rtlsdr.dll
ok  the HD Radio decoder is found: vendor\nrsc5\win-x64\nrsc5.exe
ok  the band plan loads: 21 bands, 42 allocations
ok  the basemap loads: 3 layers, 2403 lines, 4967 places
ok  the glossary loads: 8 categories, 76 articles
ok  the application icon is found: 69,716 bytes
ok  the sound card is reachable: 37 audio devices
ok  the window opens: 1180 x 760, with its icon
```

It does not cover the PyInstaller bootloader, and it cannot cover whether
Windows will let the thing start. Those need a machine where the build can
run.

### Two measurements worth keeping

- **The bundle is 267 MB in 428 files, and Qt is 101 MB of it.** SciPy is 48,
  NumPy's libraries 21, SciPy's another 20, nrsc5 7, the basemap and the driver
  under 2 between them. Qt's Mesa software OpenGL fallback alone is 20 MB and
  is kept: a radio that shows a black window on a machine with poor graphics
  drivers is a worse failure than a larger download. Excluding the Qt
  subsystems the app never imports is worth having and does not move the
  headline number, because most of what remains arrives as a dependency of
  something that *is* used — Qt6Quick and Qt6Qml are there because the virtual
  keyboard input plugin needs them.
- **Smart App Control's verdict on a new file is not always immediate.** A
  freshly written copy of `scipy/stats/_rcont/rcont.pyd` — byte-identical to
  one that had loaded a minute earlier — was blocked on first import and
  loaded normally on the next attempt. So a build can fail once and pass
  afterwards for reasons that have nothing to do with the build. Worth knowing
  before spending an afternoon on a heisenbug: it is the reputation lookup
  catching up, and it applies to *any* binary the build has just written.

### What is left

- **Nobody has run the packaged executable.** Not on this machine, and not on
  any other. The bundle is verified as far as it can be verified without
  starting it. Anyone with a Windows machine where Smart App Control is off —
  an in-place upgrade to Windows 11 rather than a clean install has it off —
  can settle it by double-clicking `dist/BetterSDR/BetterSDR.exe`.
- **The setup path has been run on one machine.** From a clean copy of the
  checkout with the system Python 3.14.2, it built the environment, installed
  thirteen packages, found the dongle and opened the window — but that machine
  already had a working driver, a good network and no proxy. The failure paths
  it writes messages for have not each been provoked.
- **`core/installer.py` is still not written.** Amendment 1's bundled
  `wdi-simple.exe`, which would bind WinUSB from inside the app, remains the
  gap between "cloned the repository" and "hearing something" for a
  first-time user. The setup script makes that gap visible rather than
  closing it: it runs the driver check and prints the remedy.
- **The `drivers/win-x64/` licence texts are still not in the repository.**
  The RTL-SDR Blog release ships them and this copy does not. That is an
  obligation on the repository as it stands, not only on a packaged build,
  and it is the one open box in THIRD-PARTY.md.


## Amendment 18 — Zadig first, and one name to remember (2026-08-31)

### What prompted it

A second machine, and one sentence: *"I did the zadig steps and it solved all
my issues right away."*

Amendment 17 left the install as a clone and one command, with the driver
treated as a problem to be *diagnosed* — `tools/setup.py` builds an
environment, installs thirteen packages, runs the driver check, and only then,
if the check fails, does the user meet Zadig at all. That ordering is defensible
as engineering and indefensible as a first experience. It arranges for a
beginner's first two minutes with the project to end in a failure, and then
offers the actual instructions as the remedy for it.

The dongle needs WinUSB on interface 0. That is true before any of the software
exists, it takes three minutes with a tool that needs no installing, and once it
is done everything downstream simply works. It is a step, not a fault.

### The new shape

Four steps, written out in **`Getting Started.txt`** at the root of the
checkout:

1. Plug the dongle in.
2. Clone the repository (Python 3.12+ installed).
3. Run Zadig — Options → List All Devices, **Bulk-In, Interface (Interface 0)**,
   WinUSB, Replace Driver.
4. Double-click **`BetterSDR.cmd`**.

Three things follow from that, and each of them is a decision rather than a
detail.

**The instructions are a plain `.txt` in the checkout.** Step 3 happens before
anything is installed, possibly before the user has looked at GitHub's rendering
of anything, and certainly before the app can tell them anything. The only
document that can be read at the moment it is needed is one that opens by being
double-clicked in Explorer. Hence a text file, CRLF, with a `.gitattributes`
that keeps it that way — and hence the README carrying the same four steps
rather than replacing them with a link.

**`BetterSDR.cmd` is the only name.** It installs the first time and starts the
app every time after. Two names — one to install, one to run — is one more
thing to remember than a beginner has spare, and it is the kind of distinction
that is obvious to the person who built it and to nobody else.

**The second run has to feel like a launcher.** That is the warm path in
`setup.py`'s `main`: if the environment is there and the package imports, print
one line and start the app. No banner, no four numbered steps, no driver check.
An installer that reprints its progress on every launch is a launcher nobody
believes, and the user stops reading it — which matters on the day it has
something real to say.

`installed_and_ready()` is the gate, and it is a single subprocess import. It
answers all three questions `build_environment` asks separately — a missing
interpreter, an environment built by a Python since upgraded away, an install
interrupted halfway — because none of those can import the package. Anything it
cannot answer falls through to the long path, which is the one that explains
itself. Measured at **0.3 seconds** before the app itself starts. (`--check`
on a warm environment takes 1.8 s, and every bit of that is the driver check
and its capture test.)

### What setup no longer claims to be

It is not part of the driver story. It runs the check and reports the result,
and when the result is bad it points at step 3 of `Getting Started.txt` rather
than reciting a remedy of its own. `doctor.py` still carries the full Zadig
walkthrough, because the case it exists for — skipped, or done on interface 1 —
is still real. It is the second line of defence now instead of the first.

`core/installer.py` (Amendment 1) is correspondingly less urgent. It would
*remove* step 3, which is worth having, but it is no longer closing a gap
between "cloned the repository" and "hearing something": step 3 closes that,
and it closes it with a tool the user can see working. It also brings a
signed-binary problem of its own to a project that has just spent a phase on
exactly that.

### One small thing that had nowhere else to go

`Exact sample rate is: 1488375.071248 Hz`, on the console, once per HD Radio
session.

librtlsdr prints it from C whenever the RTL2832U's divider cannot hit the
requested rate exactly. Every rate the radio normally uses is exact, so in
practice it is the HD Radio rate and nothing else, and it says only what
`decode/hdradio.py` already knows. It is also unreachable from Python:
the DLL writes to file descriptor 2 directly, so `sys.stderr` never sees it and
`contextlib.redirect_stderr` does nothing at all.

`native.quiet_driver` borrows descriptor 2 for the length of the call, and
**filters rather than mutes**. That is the whole point of it: the DLL's other
messages are `usb_claim_interface error %d`, `No supported tuner found`,
`Failed to submit transfer %i` — every one of them a fault worth seeing. A mute
would have taken those with the noise, and the failure it produced would be
silence at exactly the moment somebody needed a message. So the captured output
is written back minus the one known-noise line, and the known-noise list is one
line long and should stay that way.

The two lines the driver prints when it opens — `Found Rafael Micro R828D
tuner` and `RTL-SDR Blog V4 Detected` — are deliberately left alone. They are
the driver confirming the fork detection that `native.py` exists to get right.

### What is left

- **Nobody has walked the four steps on a clean machine.** They were assembled
  from a machine where step 3 was done in August and a second machine where
  doing it fixed everything. The claim that a beginner can follow
  `Getting Started.txt` unaided is the one that matters and the one not yet
  tested.
- **What Windows would have said about `--recreate` from inside the
  environment is still unknown.** The guard was driven for real against a
  clean copy of the checkout - it refuses, names the fix, and leaves the
  environment where it is - but the failure it prevents was never provoked.
- **Everything Amendment 17 left open is still open** — the packaged executable
  is still unstarted, the setup script's failure paths are still not each
  provoked, and `drivers/win-x64/` still lacks its licence texts.

## Amendment 19 — Repro-Radio, and what a title is not enough to prove (2026-08-31)

### What was asked for

> When enabled, automatically record strong signals while tuned into a
> specific frequency. Store them in a modern, lossy, highly compatible
> compression audio file format. Use the naming convention
> `RR-[Frequency]-[startTime]-[endTime]`. Show a "Maximum recording time"
> feature in hours/minutes.
>
> When on an AM/FM frequency, have an option to record the music, and just the
> music. Save individual songs as audio files, tagged with RDS data (excluding
> advertisements). If duplicate songs are recorded append a number to the end
> of the file name so the user can pick the best one to keep, since the DJ
> might be talking over one or the other. Keep up to five copies of the same
> song, if the song is played an additional time, don't save it.

Four things were settled with the user before any of it was written: **MP3**
rather than Opus or AAC; the **squelch with a hang time** as the gate rather
than a threshold of its own; **two** time caps rather than one; and, on a
station with no usable RDS, *"capture the audio, and treat it like any other
frequency"* — which is to say song capture is an FM-with-RadioText feature and
everything else simply gets recorded normally.

### Why MP3, when Opus is better

Because the question is not about the codec. Every recorder in the app until
now has written WAV, which is what a *measurement* wants: no decisions, byte
exact where it matters. Repro-Radio is the first thing here that is not a
measurement — it runs unattended for hours and it fills somebody's music
folder — so the two things that matter are that it is small enough to leave
running overnight and that every device they own will play it without being
told how.

Opus is better per bit and AAC is better per bit. Neither shows a title in
Windows Explorer, and one of them will not play in a car. At 128 kbps the
encoder is nowhere near the limiting factor anyway: an FM broadcast arrives
with 15 kHz of audio bandwidth and its own noise floor, both coarser than
anything the codec is doing.

Three properties of MP3 turned out to be load-bearing rather than incidental:

- **There is no header to fix at the end.** A file is a run of frames, so a
  recording cut short by a crash or a pulled dongle is a playable file that is
  merely shorter than intended. That is what makes the in-progress name a
  *rename* rather than a rewrite — and it is exactly what the WAV recorders
  cannot do, which is why `Engine.stop` has always had to close them before
  joining the thread.
- **The encoder is cheap.** 6.8 ms per second of 48 kHz stereo on *noise*,
  which is the worst case a lossy encoder can be handed. 0.7% of a core,
  against the WFM demodulator's 6.3%, so it runs inline on the DSP thread like
  the WAV writers with no queue and no thread of its own to get wrong.
- **`lameenc` and `mutagen` are ordinary PyPI wheels** — a 157 KB statically
  built LAME and a pure-Python tag library. That matters more than it sounds:
  Amendment 17's whole conclusion was that nothing on the install path may be
  a new unsigned binary, and a bundled `ffmpeg.exe` would have put the project
  straight back into the Smart App Control problem it had just escaped.

### The clips are the easy half

While the squelch is open, write. When it closes, keep writing for a hang time
— the gap between two overs is silence on the same channel, and a file per
sentence is not what anybody meant. Same reasoning as the monitor's release
timer, and the same failure if it is too short.

`squelch_open` of `None` means no squelch is set, which is the normal state on
a broadcast band. That is treated as **permanently open**: the user asked to
record this frequency, and the honest reading of "nothing is gating this" is
not "record nothing".

Two findings from building it:

**The minimum-clip guard has to measure signal, not file length.** The first
version discarded a recording shorter than 1.5 s, which with a 3 s hang time
could never discard anything at all — every file is at least the hang long. It
counts how long the *gate* was open instead, at half a second, which is
shorter than the shortest thing anybody says and longer than a click.
`test_a_noise_burst_that_opened_the_squelch_is_thrown_away` and
`test_a_short_transmission_is_still_a_transmission` are the two sides of it.

**The per-recording cap is enforced by Repro-Radio, not by
`RecordingLimits`.** Handing `max_seconds` to the recorder would have it stop
itself, and then the only way to tell "roll over to the next file" from "the
disk is filling" is to compare the message string. So the recorder's own stop
now means exactly one thing — the size cap or the disk — and it is always
worth reporting.

`audio/record.py` grew a `_Recorder` base for the limits and the disk guard,
because a recorder that runs unattended for hours is precisely what those were
written for and a second copy would be a second copy to get wrong. Duration is
counted in *frames* there rather than derived from bytes: that is the one
assumption a WAV writer may make and an MP3 writer may not.

### The songs are the hard half, and four things were wrong first

A song is the stretch of broadcast between two changes of RadioText, saved
separately and tagged. That sentence is almost entirely wrong, and each way it
is wrong took a scenario to find.

**RadioText is late, at both ends.** The station's playout updates it some
seconds after the song starts, so a recording that began at the text change
misses the intro of every song. The audio is therefore held in a rolling
buffer and the boundary placed *backwards*: at the last moment the sound
changed between speech and music if there was one, and at a fixed lag if there
was not. The same instant ends the previous segment and begins the next, so a
segue is cut once rather than twice and neither side is counted in both files.

That is also why the song file is **not written as the audio arrives**. It is
written from the buffer, deliberately running 25 seconds behind, so that when
the boundary turns out to have been twenty seconds ago something can still be
done about it. Writing in real time and trimming afterwards is not available:
an MP3 is a stream of frames and there is no going back into one.

**A change of text is not a change of song, and this was the failure that
would have shipped silently.** A great many stations alternate their slogan
with the title every few seconds. Treating each flip as a boundary produces
nothing but eight-second fragments, every one below the minimum song length —
so the feature runs, the button lights up, and nothing is ever saved. Nobody
would report that; it looks like a station that plays no music. A song is
therefore identified by its *tag*: the segment stays open while that same tag
keeps coming back, and closes only when a different one is announced or the
title has been absent for 20 seconds.

**Telling the slogan from the title cannot be done one string at a time.**
`The Best Music Variety` and `Rush - Tom Sawyer` are equally well-formed, and
some stations' slogans parse as an artist and a title outright. Three rules
were tried:

1. *Seen three times before ⇒ the station.* Works all afternoon and then
   quietly refuses to record the third play of somebody's favourite song —
   which is exactly the copy the five-copies rule promised them.
2. *Came back quickly, and its last stretch was short.* This brands the **song
   title** on an alternating station, because there the title's stretches are
   eight seconds too. Found by scenario, not by reading.
3. *Has gone on turning up for longer than a song could last, and has never
   once been left up for as long as it takes to announce one.* This is the
   one. A slogan does both; a title can do neither. Fifteen minutes and
   forty-five seconds, with a ten-minute absence starting the clock again so a
   track replayed in the evening is a fresh occasion.

**A title is not enough to prove a song, and this is what keeps
advertisements out.** An advertisement break is a stretch of RadioText too,
and `Bobs Motors - Best Deals In Town` parses perfectly. So the audio has to
agree: `scan/voice.py` — built for the monitor, and reused here unchanged —
reads a verdict off 0.8 s of audio once a second, and a segment is kept only
if most of it measured as music. The two checks are independent, which is the
whole point: an advertisement has to pass both, and it fails both. The status
line says so in words — *"Bobs Motors - Best Deals In Town did not sound like
music (5% of it did), so it was not kept."*

### Five copies, and counting them from the folder

Up to five, numbered, then nothing. The count is taken from the **filesystem**
rather than from the index, because pruning a folder down to the best take is
exactly what this feature expects somebody to do — and a count that did not
notice would mean that song was never recorded again, which looks like the
station simply not playing it.

The index lives in the songs folder rather than in `%APPDATA%`, so moving the
music folder takes its memory with it and deleting the folder resets the
feature. A corrupt one is an empty one, same rule as `core/settings.py`: this
is called on the DSP thread, and raising there would end an unattended session
over a metadata problem.

`key_for` folds case, accents and punctuation, and a test caught it folding
`Guns N' Roses` to two spaces where `Guns n Roses` folds to one — which would
have given each spelling five copies of its own. What it deliberately does not
fold is anything in brackets: a radio edit and a live version are different
recordings, and somebody choosing between five copies wants to be able to
tell.

### Two smaller decisions

**The volume knob must not reach the file.** `dsp/chain.py` applied volume,
mute and the limiter in one expression at the end of `process`. It is now
`body` then `output`, and Repro-Radio taps `body` — after the AGC, before the
volume — so an unattended session recorded at a whisper is not a folder of
whispered files, and muting the speakers does not silently produce hours of
zeros. The Record audio button still records `process`, because a manual
recording is a record of a listening session and that *is* what was heard.

**A borrowed radio is noticed by its silence, not by a call.** A sweep, the
aircraft screen and a monitor session all `continue` past the audio path
entirely, so they simply stop feeding Repro-Radio. Rather than add a line to
each of them — and one more to remember for whatever borrows the radio next —
`feed` treats more than two seconds without a block as an interruption and
closes what was open, the same as a retune. The rolling buffer is why it
matters rather than merely being tidy: it finds positions by elapsed time, so
one left spanning an excursion would place a song boundary in audio captured
at 1090 MHz.

**The file is opened lazily**, once a segment has lasted 20 seconds. Nothing
is lost by waiting because the audio is in the buffer either way, and it means
a station whose text fragments writes no files at all rather than creating and
deleting one every eight seconds — and a song refused by the five-copies rule
never touches the disk.

### What is left

- **Not yet run against a real station.** Everything above was found against
  synthetic air: four station behaviours driven end to end through the real
  encoder, the real `voice.py` and the real state machine, plus 75 tests. None
  of that is a broadcaster. POCSAG was in exactly this position on 2026-08-29,
  and ADS-B's surface-versus-airborne CPR fault was invisible to every
  synthetic test. **The check is: leave it on 94.9 for an hour with songs on,
  and see whether the files in `Songs` are songs, whether their boundaries are
  where the music starts, and whether any advertisement got through.**
- **The known limitation is written down rather than hidden.** On a station
  whose slogan *also* parses as an artist and a title, the first quarter of an
  hour produces fragments and may keep one wrongly-named file, because until
  the slogan has outlived a song there is no evidence separating them.
  `test_the_station_is_only_learnt_after_a_song_has_gone_by` asserts it, so
  changing it is a decision rather than a surprise.
- **HD Radio carries better metadata than RDS and is not used.** `HdState`
  already has the title and artist, from a decoder that is never wrong about
  them. Song capture is deliberately RDS-only for now; extending it is a small
  change to `_service_repro` and a question about what an HD session's five
  seconds of acquisition does to a boundary.
- **A long article can hijack the Learn search.** Adding `repro-radio` made it
  the first result for *"what does squelch mean"*, because `search()` requires
  every word to match and only falls back when the strict pass finds nothing —
  so the first article long enough to contain "what", "does" and "mean"
  monopolises every question-shaped query. Worked around by rewording the
  article; the underlying behaviour deserves its own fix, and stopwords are
  probably the answer.
- **The frozen build has not been rebuilt.** `lameenc` and `mutagen` are
  ordinary imports so PyInstaller's analysis should find them without hidden
  imports, but that is a prediction and not a measurement, and the packaged
  executable still cannot be started on this machine anyway.

## Amendment 20 — Repro-Radio meets a real station (2026-08-31)

### What was reported

> The radio repo feature doesn't seem to be separating songs. It also needs to
> not record commercials. Also the timestamps on files should be a bit cleaned
> up, maybe more like `[month][day][hour][minute][seconds]` (all just two
> digits).

...and, once it was saving files, two more:

> you've been getting the title and artist switched in the mp3 metadata
>
> also the mp3 files sound terrible / probably too much compression / digital
> radio is already very compressed, i'd say look up what the broadcast
> standard is and keep that

Amendment 19 shipped this feature against synthetic stations and said so
plainly: *"Not yet left running on a real station."* Eight minutes on 96.5 MHz
produced one clip and no songs at all. Everything below is what a real
broadcaster does that a synthetic one did not, and it is the same lesson
ADS-B's surface-versus-airborne CPR fault taught on 2026-08-28.

### The bug, and why every test passed

`RdsState.text` is the RadioText buffer *as it stands*. RDS fills it four
characters at a time, sixteen segments to a message, and the decoder published
whatever was in it. Watching 96.5 for two minutes:

```
[   1.5] '96.5    k FM - The R'
[   1.8] '96.5    k FM - The Real Slim'
[   2.3] '96.5    k FM - The Real Slim Sha        nem'
[   8.3] '96.5 Jack FM - The Real Slim Sha     Eminem'
[  16.3] '96.5 Jack FM - The Real Slim Shady - Eminem'
```

Every one of those parses. Each is a different "song", so the segmenter closed
whatever it had open and started again — several times a second, never once
reaching `MIN_SONG_S`. The feature ran, reported nothing wrong, and saved
nothing.

The synthetic station in `tests/test_repro.py` hands over whole strings,
because that is what a caller *thinks* RadioText is. Nothing in 75 tests could
have caught this, and no amount of testing the segmenter would have: the fault
is one layer down, in what the decoder claims to be offering.

`RdsState.text_steady` is the fix. A display is welcome to watch a message
fill in; anything reading it as data waits. Two things make it hard:

- **A great many stations never toggle the A/B flag**, so nothing clears the
  buffer and a shorter message leaves the tail of a longer one behind. What
  stands in for the flag is noticing that a segment carries something
  different from what is already stored — but only for a segment already had
  in this pass, because on the first assembly every segment differs from the
  spaces underneath it.
- **Covering the last character written is not covering the message.** The
  buffer is spaces underneath, so four segments cover everything there is. A
  carriage return, a full sixteen-segment pass, or a segment arriving again
  unchanged — one of the three has to say there is no more coming.

### Three fields, not two

96.5 transmits `96.5 Jack FM - The Real Slim Shady - Eminem`. Splitting on the
first separator gives an artist called `96.5 Jack FM`. So the message is cut
on *every* occurrence, the fields that are the station naming itself are
dropped, and exactly two have to be left. Three unrecognisable fields are
refused rather than guessed at: `A - B - C` is `Station - Title - Artist` on
one station and `Artist - Title - Part 2` on another.

Recognising the station's own field has an immediate rule and a learnt one.
The immediate one is the dial position — the frequency the receiver is tuned
to, which is how most US stations identify themselves, and which needs no
history. The learnt one is a value that keeps company with too many different
songs, and it needs a *share* as well as a count: after a second Eminem song,
`Eminem` has two sets of companions exactly as the slogan does. What separates
them is that the slogan is in nearly every message.

### Which half is the artist

The one question the string cannot answer, and the user found it before the
learner did. Two message shapes, two conventions:

- `A - B` with nothing set aside is `Artist - Title`. Near-universal.
- `Slogan - A - B` is a station *announcing what is on* rather than labelling
  a file, and reads the other way round — "on 96.5 Jack FM: Seven Nation Army,
  by the White Stripes".

Every message measured on the one station available puts the title first, and
treating both shapes alike named every file on it backwards. That is a prior
rather than a fact, so the learner can still overturn it, from the only
evidence in the stream: an artist comes round again with a different song, and
a title does not.

### A song ends when the music stops

The second live session saved `Teenage Dirtbag` twice — copy 1 and copy 2 of a
record played once. The station had dropped its own title for 32 seconds in
the middle of it, and `TAG_GAP_S` closed the segment on the gap alone. So the
rule is now the title being gone **and** the music having stopped, with
`TAG_LOST_S` as the backstop for a station that stops naming anything and
never stops playing.

Two smaller things fell out of the same session:

- **A boundary must never be further back than the file already reaches.** The
  writer runs `WRITE_LAG_S` behind and an MP3 cannot be rewound, so an older
  boundary silently *appends* whatever came next instead of trimming it.
- **`_sound_changed` fired on every one-second flicker.** Real music does not
  read as music every second — twenty-seven blips of `tone` or `data` inside
  one song — so the last change of reading is a boundary placed at random. The
  questions are "when did this music start" and "when did it stop".

### The advertisement test was measuring the wrong thing

Amendment 19 kept a segment only if most of it measured as music. Raising that
to catch advertisements was the first thing tried and it was wrong. Measured
over a quarter of an hour, five real songs and two real breaks:

| | music | speech |
|---|---|---|
| Notorious B.I.G. — Juicy | 0.30 | 0.05 |
| Teenage Dirtbag | 0.53 | 0.00 |
| Teenage Dirtbag | 0.67 | 0.03 |
| Beck — Loser | 0.53 | 0.00 |
| Beck — Loser | 0.46 | 0.04 |
| advertisement break | **0.14** | **0.18** |
| news and advertisements | **0.37** | **0.25** |

The music shares overlap outright — a rap record read as music less often than
a news bulletin did, because `scan/voice.py` calls a drum machine `data` — so
`MIN_MUSIC_SHARE` at 0.6 refused four of the five songs. The speech shares do
not overlap at all. `MAX_SPEECH_SHARE` is 0.10, between them with a factor of
two either side; `MIN_MUSIC_SHARE` drops to 0.25 and stays a floor rather than
a test. `test_the_thresholds_sit_between_what_was_measured_off_air` keeps the
readings, so anybody raising the music threshold to catch a break finds out
here rather than from an empty folder a week later.

The real first line of defence is the text. 96.5 sends `Accident? Boohoff Law.
Better Off With Boohoff! - 96.5 Jack FM` during a break: the station field is
dropped, one field is left, and no song is claimed at all.

### Why the files sounded bad, and what "keep the broadcast standard" means

Not the bitrate on its own. **Analog FM stereo is band-limited to 15 kHz** —
the 19 kHz pilot has to sit above the audio and the standard leaves it nowhere
else — so everything a demodulator produces above that is hiss the transmitter
never sent. Hiss is the single most expensive thing a lossy encoder can be
handed: it looks like signal at every frequency and there is nothing to mask
it behind, so the bits go there instead of on the record. At 128 kbps that was
audible as cymbals and reverb tails turning to a warble.

So the input is cut at 15 kHz first — `filters.LowPass`, flat to 14 kHz,
−19 dB at 16–18 kHz, −50 dB above 19 kHz, because `lameenc` exposes no lowpass
setter and LAME's own sits above the broadcast limit at every rate this uses.
The rate then only has to carry what the broadcast contains: an HD Radio
hybrid simulcast carries its whole digital payload in about 100 kbps of HDC
and the local HD1 measured 92, so **160 kbps** of MP3 over a 15 kHz source is
well clear of the thing being recorded. 20 kB a second, against the mono WAV
recorder's 96.

### Names

`RR-96.500MHz-0831143012-0831143145.mp3` — month, day, hour, minute, second,
two digits each, in **local** time. Every other timestamp the app writes is
UTC and should stay that way; these are the one set of files a person reads a
time off directly, and a recording made at eight in the evening that calls
itself the next day cannot be matched to what they remember hearing. The year
is in the file's own date. Two files landing on one name across a
daylight-saving change are numbered by `_free_path`, which already existed for
exactly that class of collision.

### What is still not known

- **A station that writes `Artist - Title` after its own slogan.** That is the
  case the order learner exists for, and no local station does it, so the
  learner has been driven only by tests.
- **A station whose slogan carries no dial position.** Recognising it costs
  two songs, and 96.5 puts its frequency in every message.
- **Whether `text_steady` ever refuses a station outright.** A broadcaster
  that sends neither a terminator, nor sixteen segments, nor a repeat would
  never settle — none was found, but none was looked for beyond one station.

## Amendment 21 — A window that fits, and tuning without a wheel (2026-09-02)

### What was reported

> Please touch up the ui to allow for all elements to be easily seeable and
> adjustable. For example, right now on the listen screen, the right hand side
> widgets currently go off the edge of the screen. Also on the aircraft view,
> when an airplane is clicked on, make the list view at least 1/3rd of the
> screen. Also please add buttons to either side of the frequency to allow for
> manual tuning via buttons, if the user doesn't have a scroll wheel (like on a
> laptop touchpad) it makes it hard to manually tune.

Three faults, and the third is the serious one: the app's primary tuning
control could not be operated by a large fraction of the machines it runs on.

### The control column was 33 px too narrow, and had been since Phase 3

`widgets/panel.py` already carried a note about a combo box that demanded
278 px of a 250 px column, and the cure — `fit_to_column`, which stops any one
field asking for more than the column has. That was the right fix for the
wrong half of the problem. What sets the width is not the widest *field*, it
is the widest *row*: a caption, the spacing between the columns, and a field.
Measured at Expert on this machine that is **293 px**, and the vertical
scrollbar the column always has takes **12** more. `PANEL_WIDTH` was 272.

The 33 px went off the right-hand edge, because horizontal scrolling is off
and a `QScrollArea` cannot shrink a child below its minimum. What it cost was
every spin box's arrows, the right edge of every combo box, and — invisibly,
because nobody knew they were there — **the question mark beside every row
that offers an explanation**. The Learn tab's own way in was off the edge of
the screen it was built for.

So the width is measured rather than declared. `ControlPanel.fit_to_contents`
asks the built layout what it needs and adopts that as the minimum. Three
things about that measurement are not obvious, and all three were found by
getting them wrong first:

- **Measure at Expert, not with every row visible.** A hidden row is not in a
  layout's minimum, so measuring after `set_level(SIMPLE)` sizes the column
  for three controls and cuts Expert off — the original fault, arriving by a
  new route. But measuring with *everything* visible is worse, not better: it
  includes rows no level ever shows at once, and gave **381 px** for a column
  that can never display more than 293.
- **A scrollbar that has not been shown is 100 px wide.** `QWidget.width()`
  before the first show is the default, whatever the widget is. Asking the
  scrollbar how wide it *was*, rather than how wide it wants to be, made the
  column **88 px** too wide and nothing on the screen said why.
- **A `QScrollArea` is horizontally Expanding by default**, so in a splitter
  it takes its share of every pixel the window grows by. At Simple, where
  three controls are showing, that had the column half as wide again as it
  needed to be. `Preferred` keeps it at the width it asked for; dragging the
  handle still overrides it, which is what a splitter is for.

The column is a splitter pane now rather than a fixed one — "easily
**adjustable**" was half the request — with its measured minimum as the floor
and 520 px as the ceiling, because nothing in it gets better with more room
and a control column half the window wide is a worse screen than a spectrum.
The width is remembered in `panel_width`, clamped on the way back in by that
same measured minimum: a width saved by an older build, or on a machine with
a different font, must never be able to cut a row off again.

### A control hidden alone leaves half a row behind

`Section.add_wide` wraps a control in a row of its own when it carries a
question mark. `setVisible(False)` on the control then hides the control and
leaves the question mark floating in the column with nothing to its left,
which is what the HD Radio subchannel row did on every station that does not
carry HD — visible in the first screenshot taken of this work and, once seen,
obviously wrong. `panel.set_row_visible` is the way to hide a row, and the row
is remembered on the widget as a Qt property rather than a Python attribute,
because the C++ object's Python wrapper is not guaranteed to be the same one
twice.

### Tuning without a wheel

The digit-wise readout is the fastest tuning control in any SDR application
and `widgets/frequency.py` says so. It was also, until now, the *only* way to
tune by hand at a granularity of the user's choosing, and it was operated
exclusively by the mouse wheel — which on a laptop is a two-finger trackpad
gesture a beginner may never have used deliberately. An application whose
whole argument is that a beginner should not have to know things had put its
primary control behind one.

Three ways in now, and the readout keeps its wheel:

- **Each digit is two buttons.** Clicking its upper half winds it up and its
  lower half down, by the decade the wheel would have moved. The highlight
  follows the half under the pointer rather than the whole digit, so what a
  click is about to do is visible before it happens.
- **A step button either side of the readout**, inside the Discover-list
  buttons that were already there. They are captioned with the size of the
  step — "− 200 kHz" — because that is the only thing about them worth
  knowing, and they auto-repeat, which is what makes them a substitute for a
  wheel rather than a token gesture towards one: the FM band is a hundred
  channels, or about four seconds of holding one down.
- **The step is the band's own channel raster** where it has one, so a press
  is one station on FM broadcast and one channel on Marine VHF. Where a band
  states no raster the step is the largest on a fixed ladder that still fits
  inside the channel being listened through — a step narrower than the filter
  moves the readout without moving the station out of it, and one much wider
  walks past things without ever hearing them.

`step_frequency` snaps before it steps, for the same reason click-to-tune
snaps at all: arriving on 98.437 MHz from a click on the spectrum and pressing
up should give 98.5, which is a station, rather than 98.637, which is nothing.

The readout also has a minimum width now, taken from its own font metrics.
Without one it is Expanding with no floor, and a narrow window squeezed the
digits until they were painted over the buttons either side — which reads as a
rendering fault rather than as a window that is too small. The window itself
has a minimum of 900 × 600 for the same reason.

### The aircraft list

A `QSplitter` opens at the size hints of its panes, and a list of cards hints
at almost nothing until there are cards in it — so the aircraft screen opened
with the map at 87% and the list as a 60 px strip showing the top of one row.
The stretch factors that were meant to make it 3:2 never applied, because
stretch decides how *extra* room is shared out rather than what the opening
sizes are. Stated outright now: 60/40 on arrival.

`_reveal` used to give the list a share only when it had been dragged fully
shut. Now it does so whenever the list has less than a third of the **screen**
— a third of the splitter is a quarter of the window, because the heading and
the button above it are a fifth of it, and this is a promise about what
somebody sees. It is never taken back down: a list dragged larger stays
larger. The map keeps at least 40% of the splitter whatever happens, because
the click that asked for this landed on it.

The map's own right-hand longitude label is dropped rather than drawn off the
edge, which is the same rule as the ribbon's: half a longitude is not a
smaller label, it is a wrong one, and the meridian line still says where it
is.

### What is not covered

Every layout above was driven through the real widgets on the real Windows
font stack and photographed, at 900 × 600, 1000 × 640 and 1180 × 760, at all
three levels. None of it has been seen on a high-DPI display at 150% or 200%
scaling, where the fonts are larger and the measurement — taken from the font
metrics at run time, and so it should follow — has not been checked. The
offscreen Qt platform is no use for this: it ships no fonts and overstates
every string's width by a factor of two, which is worth knowing before
anybody tries to screenshot this app in a test.

Nor has any of it been used with a dongle attached. Nothing here touches the
engine, the reader or a device call — the tuning buttons go through the same
`FrequencyDisplay.set_value` the wheel does — so the risk is that a button
does nothing rather than that the radio does something unexpected.
