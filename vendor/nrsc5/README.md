# nrsc5 — bundled HD Radio (NRSC-5) decoder

`win-x64/nrsc5.exe` is a **separate program**, not part of BetterSDR. BetterSDR
starts it as a child process and talks to it over pipes. It is bundled so that a
first-time user gets HD Radio without downloading or configuring anything, the
same reasoning that puts the RTL-SDR driver in `drivers/win-x64/`.

## Licence — read before changing how this is used

nrsc5 is **GPL-3.0** (it carries a modified faad2, which is GPL). Full text in
`COPYING`; upstream's own notice in `LICENSE`.

Bundling it as a standalone executable invoked over a pipe is *mere aggregation*
under the GPL, so BetterSDR's own licence is unaffected. **Loading `libnrsc5`
in-process — ctypes, cffi, anything — would be linking, and would place the whole
of BetterSDR under GPL-3.** The process boundary is therefore a licensing
requirement, not just the crash isolation and codec convenience that Amendment 3
of docs/PLAN.md describes. Do not "simplify" it away.

Distributing BetterSDR with this binary means also offering the corresponding
source. The commit below plus this file is that offer; keep them together.

## Provenance

| | |
|---|---|
| Upstream | https://github.com/theori-io/nrsc5 |
| Commit | `b7b821f591d946d8f5e563fe7869566d21736896` (2026-08-24) |
| Built | 2026-08-28 on this machine |
| Toolchain | MSYS2 MINGW64, GCC 16.1.0 |
| Size | 6,600,905 bytes |

Built with upstream's own flags (`support/msys2-build`):

```
cmake -G "MSYS Makefiles" \
    -D USE_STATIC=ON \
    -D USE_SYSTEM_LIBUSB=OFF \
    -D USE_SYSTEM_RTLSDR=OFF \
    -D USE_SYSTEM_LIBAO=OFF \
    -D USE_SYSTEM_FFTW=OFF \
    -D USE_SSE=ON \
    ..
```

`USE_STATIC=ON` is what makes this one file: fftw3f, libusb, librtlsdr, libao and
the patched faad2 are all compiled from source and linked in. Verified with `ldd`
and by running it with only `C:\WINDOWS\system32` on PATH — it imports **nothing
but Windows system DLLs**. No mingw runtime, no `libnrsc5.dll`.

## Rebuilding

Needs MSYS2 (this machine: `C:\Users\Peter\DevTools\msys64`). MSVC **cannot** build
nrsc5 — the build requires autoconf/automake/libtool and GCC-only flags.

```
pacman -S --needed autoconf automake git gzip make \
    ${MINGW_PACKAGE_PREFIX}-gcc ${MINGW_PACKAGE_PREFIX}-cmake \
    ${MINGW_PACKAGE_PREFIX}-libtool patch tar xz
```

**Build from a short path.** faad2's nested try-compile directories overflow the
Windows 260-character `MAX_PATH` limit from a long one, and the failure is
misleading: the compile appears to succeed and then `ar` reports the object file
as "No such file or directory". `~/nrsc5` inside MSYS2 works.

## The interface BetterSDR uses

```
nrsc5.exe -r - --iq-input-format cu8 -o - -t raw <program> 
```

- **stdin** — IQ as `cu8` at **1,488,375 S/s**, byte-identical to what the reader
  thread already puts in the ring buffer. No conversion.
- **stdout** — audio, **signed 16-bit little-endian, 44,100 Hz, 2 channels,
  interleaved L,R**. Note 44.1 kHz, *not* the 48 kHz the rest of the audio path
  runs at, so this needs resampling — and `1488375 / 48000 = 31.0078` is why HD
  cannot use the `dsp/demod.py` skeleton at all.
- **stderr** — metadata as log lines: station name, slogan, title, artist, bit
  rate, MER, BER, plus `Synchronized` / `Lost synchronization`.
- `<program>` selects the subchannel: `0` = HD1, `1` = HD2, and so on.

Verified against `support/sample.xz` from upstream: decodes KUT Austin with
station name, slogan, song title and ~64 kbps audio, over both file input and the
stdin pipe.
