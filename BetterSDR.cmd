@echo off
rem Double-click front door for BetterSDR.
rem
rem A batch file rather than an executable on purpose. Smart App Control
rem blocks an unsigned program outright on a clean Windows 11 machine, and
rem there is no way past the dialog; a script handed to the signed cmd.exe
rem is not a program by that definition and runs normally. Everything it
rem does is in tools\setup.py - this only finds a Python to run it with.
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
pause
exit /b 1

:done
if errorlevel 1 pause
