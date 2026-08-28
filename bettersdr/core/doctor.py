"""Driver diagnostics for the RTL-SDR dongle on Windows.

Getting the USB driver right is the single biggest barrier between a beginner
and their first signal, so we diagnose it precisely rather than reporting
"device not found" and leaving the user to search forums.

This module is pure logic with no UI, so both the `--info` CLI and the
first-run wizard can share it. It talks to the Windows Configuration Manager
API (cfgmgr32) directly, which is faster and more dependable than shelling out
to PowerShell.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_uint32, c_void_p, create_unicode_buffer
from dataclasses import dataclass, field
from enum import Enum

# Realtek vendor ID; the two product IDs RTL-SDR dongles enumerate as.
VENDOR_ID = "VID_0BDA"
PRODUCT_IDS = ("PID_2838", "PID_2832")

_CR_SUCCESS = 0
_FILTER_ENUMERATOR = 0x00000001
_FILTER_PRESENT = 0x00000100
_LOCATE_NORMAL = 0

_DN_HAS_PROBLEM = 0x00000400

_CM_DRP_DEVICEDESC = 0x01
_CM_DRP_SERVICE = 0x05

# Configuration Manager problem codes we can give specific advice about.
PROBLEM_NOT_CONFIGURED = 1
PROBLEM_FAILED_START = 10
PROBLEM_DISABLED = 22
PROBLEM_FAILED_INSTALL = 28

# Kernel services that mean the dongle is bound to a driver we can use.
_USABLE_SERVICES = {"winusb", "libusbk", "libusb0"}
# The Windows TV-tuner driver: present, working, and useless to us.
_DVBT_SERVICES = {"rtl2832uusbdevice", "rtl2838uusbdevice", "rtl2832u"}


class DriverState(Enum):
    NOT_PRESENT = "not_present"
    NO_DRIVER = "no_driver"
    DVB_T_DRIVER = "dvbt_driver"
    DISABLED = "disabled"
    USABLE = "usable"
    UNKNOWN = "unknown"


@dataclass
class UsbNode:
    """One PnP node belonging to the dongle."""

    instance_id: str
    description: str
    service: str
    status: int
    problem: int

    @property
    def interface_number(self) -> int | None:
        """0 or 1 for a composite device's interfaces, None for the parent."""
        marker = "&MI_"
        if marker not in self.instance_id:
            return None
        try:
            return int(self.instance_id.split(marker)[1][:2])
        except ValueError:
            return None

    @property
    def is_composite_parent(self) -> bool:
        return self.interface_number is None

    @property
    def has_problem(self) -> bool:
        return bool(self.status & _DN_HAS_PROBLEM) or self.problem != 0


@dataclass
class Diagnosis:
    state: DriverState
    nodes: list[UsbNode] = field(default_factory=list)
    # The node the SDR driver needs to be bound to.
    target: UsbNode | None = None

    @property
    def ok(self) -> bool:
        return self.state is DriverState.USABLE

    @property
    def headline(self) -> str:
        return _HEADLINES[self.state]

    @property
    def remedy(self) -> list[str]:
        """Ordered, plain-English steps the user should take."""
        return list(_REMEDIES[self.state])


_HEADLINES = {
    DriverState.NOT_PRESENT: "No RTL-SDR dongle found",
    DriverState.NO_DRIVER: "Dongle found, but Windows has no driver for it",
    DriverState.DVB_T_DRIVER: "Dongle found, but it is set up as a TV tuner",
    DriverState.DISABLED: "Dongle found, but it is disabled in Device Manager",
    DriverState.USABLE: "Dongle found and ready to use",
    DriverState.UNKNOWN: "Dongle found, but its driver state is unclear",
}

_ZADIG_STEPS = (
    "Download Zadig from https://zadig.akeo.ie and run it (no install needed).",
    "In Zadig, tick Options then List All Devices.",
    "Pick the entry named Bulk-In, Interface (Interface 0) from the dropdown.",
    "IMPORTANT: choose Interface 0, not Interface 1. Interface 1 will not work.",
    "Check the driver on the right says WinUSB, then click Replace Driver.",
    "Wait for it to finish, then come back here and check again.",
)

_REMEDIES = {
    DriverState.NOT_PRESENT: (
        "Plug the dongle into a USB port.",
        "Prefer a port directly on the computer over a hub.",
        "If it is already plugged in, try a different port, then check again.",
    ),
    DriverState.NO_DRIVER: (
        "Windows does not ship a driver for this dongle, which is normal.",
        "You need to assign it the WinUSB driver using a free tool called Zadig.",
        *_ZADIG_STEPS,
    ),
    DriverState.DVB_T_DRIVER: (
        "Windows installed its TV-tuner driver, which cannot be used for radio.",
        "You need to replace it with WinUSB using a free tool called Zadig.",
        *_ZADIG_STEPS,
        "Note: this stops the dongle working as a TV tuner. That is expected.",
    ),
    DriverState.DISABLED: (
        "Open Device Manager.",
        "Find the dongle, right-click it, and choose Enable device.",
        "Then check again.",
    ),
    DriverState.USABLE: (),
    DriverState.UNKNOWN: (
        "The driver is not one we recognise.",
        "Running Zadig and assigning WinUSB to Interface 0 usually resolves it.",
        *_ZADIG_STEPS,
    ),
}


# --------------------------------------------------------------------------
# cfgmgr32 access
# --------------------------------------------------------------------------


def _cfgmgr() -> ctypes.WinDLL:
    lib = ctypes.WinDLL("cfgmgr32")
    lib.CM_Get_Device_ID_List_SizeW.argtypes = (
        POINTER(c_uint32),
        ctypes.c_wchar_p,
        c_uint32,
    )
    lib.CM_Get_Device_ID_ListW.argtypes = (
        ctypes.c_wchar_p,
        c_void_p,
        c_uint32,
        c_uint32,
    )
    lib.CM_Locate_DevNodeW.argtypes = (POINTER(c_uint32), ctypes.c_wchar_p, c_uint32)
    lib.CM_Get_DevNode_Status.argtypes = (
        POINTER(c_uint32),
        POINTER(c_uint32),
        c_uint32,
        c_uint32,
    )
    lib.CM_Get_DevNode_Registry_PropertyW.argtypes = (
        c_uint32,
        c_uint32,
        POINTER(c_uint32),
        c_void_p,
        POINTER(c_uint32),
        c_uint32,
    )
    return lib


def _device_ids(lib: ctypes.WinDLL) -> list[str]:
    """Every present USB device instance ID."""
    flags = _FILTER_ENUMERATOR | _FILTER_PRESENT
    size = c_uint32(0)
    if lib.CM_Get_Device_ID_List_SizeW(byref(size), "USB", flags) != _CR_SUCCESS:
        return []
    buf = create_unicode_buffer(size.value)
    if lib.CM_Get_Device_ID_ListW("USB", buf, size.value, flags) != _CR_SUCCESS:
        return []
    # A double-null-terminated list, so split on NUL and drop the empties.
    return [item for item in buf[: size.value].split("\0") if item]


def _registry_property(lib: ctypes.WinDLL, devinst: int, prop: int) -> str:
    length = c_uint32(0)
    lib.CM_Get_DevNode_Registry_PropertyW(devinst, prop, None, None, byref(length), 0)
    if length.value == 0:
        return ""
    buf = create_unicode_buffer(length.value // 2 + 1)
    result = lib.CM_Get_DevNode_Registry_PropertyW(
        devinst, prop, None, buf, byref(length), 0
    )
    return buf.value if result == _CR_SUCCESS else ""


def _node(lib: ctypes.WinDLL, instance_id: str) -> UsbNode | None:
    devinst = c_uint32(0)
    located = lib.CM_Locate_DevNodeW(byref(devinst), instance_id, _LOCATE_NORMAL)
    if located != _CR_SUCCESS:
        return None
    status, problem = c_uint32(0), c_uint32(0)
    got = lib.CM_Get_DevNode_Status(byref(status), byref(problem), devinst, 0)
    if got != _CR_SUCCESS:
        status.value, problem.value = 0, 0
    return UsbNode(
        instance_id=instance_id,
        description=_registry_property(lib, devinst.value, _CM_DRP_DEVICEDESC),
        service=_registry_property(lib, devinst.value, _CM_DRP_SERVICE),
        status=status.value,
        problem=problem.value,
    )


def _matches_dongle(instance_id: str) -> bool:
    upper = instance_id.upper()
    return VENDOR_ID in upper and any(pid in upper for pid in PRODUCT_IDS)


def diagnose() -> Diagnosis:
    """Inspect the dongle's current driver binding."""
    lib = _cfgmgr()
    nodes = []
    for instance_id in _device_ids(lib):
        if not _matches_dongle(instance_id):
            continue
        node = _node(lib, instance_id)
        if node is not None:
            nodes.append(node)

    if not nodes:
        return Diagnosis(state=DriverState.NOT_PRESENT)

    # The SDR driver binds to interface 0 of the composite device. If the
    # dongle enumerated as a plain (non-composite) device, that node is the
    # target instead.
    interfaces = [n for n in nodes if not n.is_composite_parent]
    if interfaces:
        target = next((n for n in interfaces if n.interface_number == 0), interfaces[0])
    else:
        target = nodes[0]

    return Diagnosis(state=_classify(target), nodes=nodes, target=target)


def _classify(target: UsbNode | None) -> DriverState:
    if target is None:
        return DriverState.UNKNOWN

    service = target.service.lower()
    if service in _USABLE_SERVICES and not target.has_problem:
        return DriverState.USABLE
    if service in _DVBT_SERVICES:
        return DriverState.DVB_T_DRIVER
    if target.problem == PROBLEM_DISABLED:
        return DriverState.DISABLED
    if target.problem in (PROBLEM_FAILED_INSTALL, PROBLEM_NOT_CONFIGURED) or not service:
        return DriverState.NO_DRIVER
    return DriverState.UNKNOWN


__all__ = ["Diagnosis", "DriverState", "UsbNode", "diagnose"]
