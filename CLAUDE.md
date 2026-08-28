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
| **2 — Discovery** (sweep, detect, classify) | Not started, except HD Radio detection (done) |
| **3 — SDR# parity** | Not started |
| **4 — Decoders** (RDS, HD Radio, ADS-B, POCSAG) | Not started |
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
# The GUI. --level is simple/standard/expert; the DSP engine is the same in all three.
.venv/Scripts/python.exe -m bettersdr.app
.venv/Scripts/python.exe -m bettersdr.app --freq 162.55 --level expert
```

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
    frontend.py   gain selection, shared by listener/GUI/scanner
    engine.py     the DSP thread, device ownership, single-slot mailbox
  dsp/
    convert.py    uint8 -> complex64 LUT
    filters.py    streaming FIR decimation, de-emphasis, discriminator, squelch
    demod.py      wfm/nfm/am/usb/lsb/cw/dsb/raw, one shared 3-stage skeleton
    psd.py        Welch PSD in dBFS, peak hold, noise floor, occupied bandwidth
    features.py   classifier features; HD Radio sideband detection
                  agc.py, denoise.py, correct.py to come
  scan/
    bandplan/     us.yaml + loader; feeds both the ribbon and the classifier
                  sweeper.py, detector.py, classifier.py to come
  decode/         rds, adsb, pocsag
  audio/          output.py (jitter buffer + clock drift sync), record.py to come
  ui/
    app shell     main_window.py, levels.py, listen_view.py
    widgets/      spectrum.py, waterfall.py, frequency.py, meter.py,
                  colormaps.py, axes.py
drivers/win-x64/  bundled RTL-SDR Blog driver V1.4.0 (committed on purpose)
tests/
  synth.py        synthetic IQ generator — most tests need no hardware
```

### Threading model

Three threads, one direction of data flow, no locks on the hot path:

1. **Reader thread** — `rtlsdr_read_sync()` in a loop into a preallocated ring buffer. We use `read_sync` rather than `read_async` because it releases the GIL inside the C call and lets the scanner retune synchronously between reads, so one mechanism serves both listening and scanning.
2. **DSP thread** — consumes blocks; writes audio into the sounddevice jitter buffer and display frames into a single-slot mailbox.
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
- Line length 90, ruff with `E,F,I,UP,B,SIM`. Keep `ruff check .` clean.

## Git

`git init` has been run. **Nothing has been committed yet** — the user has not asked for a commit. Ask before committing.

The DLLs in `drivers/win-x64/` are committed deliberately; bundling them is the whole point of the absolute-path load strategy.
