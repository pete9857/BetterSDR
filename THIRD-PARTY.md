# Third-party components

BetterSDR itself is licensed under the GNU General Public License, version 3 or
later — see [LICENSE](LICENSE). This file lists everything else that ships in
the repository or in a packaged build, and under what terms.

## Bundled programs and libraries (shipped in the download)

| Component | Where | Licence | Notes |
|---|---|---|---|
| **nrsc5** | `vendor/nrsc5/win-x64/nrsc5.exe` | GPL-3.0-or-later | A separate program, spoken to over pipes and never linked. Its licence text is at `vendor/nrsc5/COPYING`. |
| **librtlsdr (RTL-SDR Blog fork V1.4.0)** | `drivers/win-x64/rtlsdr.dll` | GPL-2.0-or-later | Loaded by absolute path via `ctypes`. |
| **pthreads-win32** | `drivers/win-x64/pthreadVC2.dll` | LGPL-2.1-or-later | Shipped as part of the RTL-SDR Blog Windows release; a dependency of `rtlsdr.dll`. |
| **Microsoft Visual C++ 2010 runtime** | `drivers/win-x64/msvcr100.dll` | Microsoft redistributable terms | Also part of the RTL-SDR Blog release; a dependency of `rtlsdr.dll`. |
| **Qt 6 / PySide6 / shiboken6** | packaged build only | LGPL-3.0-only (as used here) | Also offered by The Qt Company under GPL-2.0/GPL-3.0 or a commercial licence. |
| **NumPy** | packaged build only | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | |
| **SciPy** | packaged build only | BSD-3-Clause | |
| **pyqtgraph** | packaged build only | MIT | |
| **sounddevice** | packaged build only | MIT | |
| **PortAudio** | packaged build only | MIT | Bundled inside the `sounddevice` wheel. |
| **PyYAML** | packaged build only | MIT | |

## Bundled data

| Component | Where | Terms |
|---|---|---|
| **Natural Earth 1:10m** (land, lakes, states, populated places) | compiled into `bettersdr/ui/basemap/us.bsm` | Public domain. No permission needed, no attribution required. |
| **US Census Bureau gazetteer and `sub-est` population estimates** | compiled into `bettersdr/ui/basemap/us.bsm` | Public domain, as works of the United States government. |

Natural Earth was chosen over OpenStreetMap specifically to avoid ODbL's
attribution and share-alike obligations on a derived database. See the
basemap notes in `CLAUDE.md`.

## Why nrsc5 runs as a child process

BetterSDR is GPL-3 itself, so linking `libnrsc5` would now be permitted. The
subprocess boundary stays for the two reasons that came first: the HDC codec is
proprietary with no public spec, and a decoder that can crash should not be able
to take the radio down with it.

## Before distributing a build

Distributing the **repository** — which is how BetterSDR is meant to be
obtained, see the README — carries none of the obligations below except the
first. They apply to a packaged build in which the third-party libraries are
redistributed as binaries.

- ☐ Copy the upstream licence texts for `rtlsdr.dll`, `pthreadVC2.dll` and
  `msvcr100.dll` into `drivers/win-x64/` — the RTL-SDR Blog release ships them
  and this repository currently does not. **This one applies to the repository
  as it stands**, because those DLLs are committed here.
- ☑ Note the exact nrsc5 revision the bundled `nrsc5.exe` was built from, next
  to a link to its source, to satisfy GPL-3 §6 — done, in
  `vendor/nrsc5/README.md`, and the binary itself reports the same revision
  (`nrsc5.exe -v` prints `b7b821f`).
- ☑ Package Qt as separate shared libraries (PyInstaller one-dir, not
  one-file), which is what keeps the LGPL-3 relinking obligation
  straightforward — `BetterSDR.spec` is `--onedir`, and Qt lands in
  `dist/BetterSDR/_internal/PySide6/` as ordinary replaceable DLLs.
- ☐ Ship the licence texts of the Python dependencies alongside a packaged
  build. `LICENSE` and this file are collected into the bundle already; the
  individual `*.dist-info/LICENSE` files that pip installed are not.
