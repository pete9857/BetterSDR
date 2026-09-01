This project is for educational use only.

It is intended for use on Windows 11, with a RTL-SDR v4 dongle and dipole antenna kit in the United States. v3 and v4L dongles are likely compatible as well.

The reason I'm making this app is because of the steep learning curve typically associated with SDR operation.
BetterSDR is designed to be an entry-level Software Defined Radio controller.
Although the quickstart guide on rtl-sdr.com is very well written, it's rather intimidating for beginners who just want to plug in their dongle and start listening to the myriad of signals around them.
Many programs, including the one listed on the quickstart guide assume that the user is already familiar with common SDR (and radio in-general) terms and what they do.
BetterSDR does not assume this, the goal is to make the process of basic listening and scanning as easy as possible.
Hopefully to inspire users to move on to more advanced hardware and software once they realize the limits of basic equipment and software.

## Getting started

There is a step-by-step version of all of this in
**[Getting Started.txt](Getting%20Started.txt)**, written for somebody who
has not done it before. It is plain text so it can be read in the folder,
before anything is installed. The short version is four steps:

**1. Plug the dongle in.** A USB port on the computer itself rather than a
hub. Windows may install its own TV-tuner driver; let it, step 3 replaces
it.

**2. Get a copy.** You need
[Python 3.12 or newer](https://www.python.org/downloads/) — tick **Add
python.exe to PATH** in the installer — and this repository:

```
git clone https://github.com/pete9857/BetterSDR
cd BetterSDR
```

**3. Assign the WinUSB driver with [Zadig](https://zadig.akeo.ie).** This is
the step that makes the dongle work as a radio, and it is a one-off. Run
Zadig as administrator, tick *Options → List All Devices*, choose
**Bulk-In, Interface (Interface 0)** — interface 0, not interface 1 — check
the target driver says **WinUSB**, and click *Replace Driver*.

Interface 1 correctly stays without a driver afterwards. Picking it by
mistake is the most common way to end up with a dongle that looks installed
and does nothing.

**4. Double-click `BetterSDR.cmd`.**

The first run builds a private Python environment in `.venv`, installs what
the radio needs, checks the dongle and opens the app; it takes a minute or
two. Every run after that just opens the app, in about a second. That one
file is the whole interface — there is nothing else to remember.

From a terminal in the same folder, `BetterSDR` in Command Prompt or
`.\BetterSDR` in PowerShell does the same thing, and takes the same flags:

```
BetterSDR.cmd --check      check the driver and the dongle, then stop
BetterSDR.cmd --update     reinstall the dependencies
BetterSDR.cmd --recreate   throw the environment away and build it again
BetterSDR.cmd --dev        also install pytest and ruff
```

### If the dongle is not found

`BetterSDR.cmd --check` names the specific problem and the remedy, and
BetterSDR itself opens on a walkthrough rather than an error when the radio
is not ready. Almost always the answer is step 3, on interface 0. The same
check is available directly once the environment exists:

```
.venv/Scripts/python.exe -m bettersdr.core.device --info
```

Exit code 0 means the radio is ready.

### Why there is no `.exe` to download

There is a PyInstaller build — `py tools/build_app.py` produces a one-folder
application in `dist/` — but it is **not the supported way to run BetterSDR,
and on a current Windows 11 machine it may not run at all.**

Smart App Control is switched on by default on clean Windows 11 installs. It
refuses to start any program that is neither signed by a publisher it
recognises nor already known to Microsoft's reputation service, and a freshly
built, unsigned application is neither. Unlike the older SmartScreen prompt
there is no *More info → Run anyway*: the process simply never starts, and
nothing the user sees explains why. Clearing that needs a code-signing
certificate, which is a decision about money and identity rather than about
software.

Python, by contrast, is signed and recognised, and everything the setup
command installs arrives as an ordinary package. That is the whole reason the
clone-and-run route is the one documented above.

## Credits

The aircraft map is drawn from [Natural Earth](https://www.naturalearthdata.com/)
1:10m vectors and the US Census Bureau's gazetteer, both public domain. HD Radio
is decoded by [nrsc5](https://github.com/theori-io/nrsc5), which is bundled as a
separate program and spoken to over pipes. See
[THIRD-PARTY.md](THIRD-PARTY.md) for the full list and the licence terms.

## Licence

Copyright (C) 2026 Stone Merchant, LLC

BetterSDR is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

BetterSDR is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

The bundled RTL-SDR driver, the bundled nrsc5 decoder and the third-party
libraries a packaged build carries have licences of their own. They are all
listed in [THIRD-PARTY.md](THIRD-PARTY.md).
