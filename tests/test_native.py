"""What the driver itself prints, and what reaches the user.

librtlsdr writes to the process's stderr from C, so none of it goes
through `sys.stderr` and none of it can be filtered by anything Python
puts there. `native.quiet_driver` borrows file descriptor 2 for the
length of a call instead.

The rule it enforces is the point of these tests: exactly one line the
driver prints is routine - the achieved sample rate, which it reports for
every rate the RTL2832U's divider cannot hit exactly, and which for this
app means the HD Radio rate and nothing else. Every other line it can
print is a fault. So the mechanism filters rather than mutes, and a test
that only checked the noise was gone would pass just as well for a
version that swallowed a claim-interface error.
"""

from __future__ import annotations

import os
from pathlib import Path

from bettersdr.core import native

NOISE = b"Exact sample rate is: 1488375.071248 Hz\n"
FAULT = b"usb_claim_interface error -6\n"


def _write_at_the_c_level(*lines: bytes) -> None:
    """Write past `sys.stderr`, the way the DLL does."""
    for line in lines:
        os.write(2, line)


def test_the_achieved_sample_rate_does_not_reach_the_console(capfd):
    with native.quiet_driver():
        _write_at_the_c_level(NOISE)
    assert capfd.readouterr().err == ""


def test_a_real_driver_fault_still_does(capfd):
    with native.quiet_driver():
        _write_at_the_c_level(FAULT)
    assert capfd.readouterr().err == FAULT.decode()


def test_the_noise_is_dropped_out_of_the_middle_and_the_rest_kept(capfd):
    with native.quiet_driver():
        _write_at_the_c_level(FAULT, NOISE, FAULT)
    assert capfd.readouterr().err == (FAULT + FAULT).decode()


def test_stderr_comes_back_when_the_call_raises(capfd):
    """A device call that fails must not leave the console redirected."""
    try:
        with native.quiet_driver():
            raise RuntimeError("the call failed")
    except RuntimeError:
        pass

    _write_at_the_c_level(FAULT)
    assert capfd.readouterr().err == FAULT.decode()


def test_setting_a_rate_is_the_only_call_that_borrows_the_console():
    """It is the one chatty call; anything else is a fault worth seeing."""
    root = Path(__file__).resolve().parent.parent
    source = (root / "bettersdr" / "core" / "device.py").read_text(
        encoding="utf-8"
    )
    assert source.count("quiet_driver()") == 1
