@echo off
rem The front door. Double-click it, or type BetterSDR in this folder.
rem
rem The first run installs; every run after that starts the app in about a
rem second. Everything it does is in tools\setup.py - all this does is find
rem a Python to run that with.
rem
rem A batch file rather than an executable on purpose. Smart App Control
rem blocks an unsigned program outright on a clean Windows 11 machine, and
rem there is no way past the dialog; a script handed to the signed cmd.exe
rem is not a program by that definition and runs normally.
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    py tools\setup.py %*
    goto :done
)

where python >nul 2>&1
if %errorlevel%==0 (
    python tools\setup.py %*
    goto :done
)

echo.
echo Python was not found on this computer.
echo.
echo BetterSDR needs Python 3.12 or newer. Install it from
echo     https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run this again.
echo.
echo See "Getting Started.txt" in this folder for the whole setup.
echo.
pause
exit /b 1

:done
if errorlevel 1 pause
