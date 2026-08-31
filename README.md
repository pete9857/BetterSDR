This project is for educational use only.

It is intended for use on Windows 11, with a RTL-SDR v4 dongle and dipole antenna kit in the United States. v3 and v4L dongles are likely compatible as well.

The reason I'm making this app is because of the steep learning curve typically associated with SDR operation.
BetterSDR is designed to be an entry-level Software Defined Radio controller.
Although the quickstart guide on rtl-sdr.com is very well written, it's rather intimidating for beginners who just want to plug in their dongle and start listening to the myriad of signals around them.
Many programs, including the one listed on the quickstart guide assume that the user is already familiar with common SDR (and radio in-general) terms and what they do.
BetterSDR does not assume this, the goal is to make the process of basic listening and scanning as easy as possible.
Hopefully to inspire users to move on to more advanced hardware and software once they realize the limits of basic equipment and software.

## Getting started

You need [Python 3.12 or newer](https://www.python.org/downloads/) — tick
**Add python.exe to PATH** in the installer — and a copy of this repository:

```
git clone https://github.com/pete9857/BetterSDR
cd BetterSDR
py tools/setup.py
```

That one command builds a private Python environment in `.venv`, installs
everything the radio needs, checks your dongle and opens the app. It takes a
minute or two the first time. Running it again is how you start BetterSDR
afterwards, and the second run takes about a second.

If you would rather not use a terminal at all, double-click **BetterSDR.cmd**
in the folder you cloned. It does exactly the same thing.

```
py tools/setup.py --check      check the driver and the dongle, then stop
py tools/setup.py --update     reinstall the dependencies
py tools/setup.py --recreate   throw the environment away and build it again
```

### If the dongle is not found

The setup command finishes by running the driver check, and BetterSDR itself
opens on a walkthrough rather than an error when the radio is not ready. The
usual cause on a new machine is that Windows has bound its own TV-tuner driver
to the dongle; the check names the remedy. You can run it on its own at any
time:

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
