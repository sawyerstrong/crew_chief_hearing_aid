"""Read-only joystick polling via the Windows Multimedia API.

Exists because SDL/pygame degrades force feedback. SDL opens the device — and
with RawInput and HIDAPI disabled it still acquires it through DirectInput —
which disturbs the FFB effects the sim has loaded. The wheel goes light and
floaty the moment the pipeline starts. Disabling SDL's hints was not enough.

`joyGetPosEx` never opens anything. It reads state the driver already
maintains, with no device handle, no cooperative level, and no acquisition, so
it cannot interfere with whatever else owns the wheel. That property is the
entire reason this module exists.

The cost is a **32-button ceiling** — `dwButtons` is a DWORD bitmask, so
buttons 32+ are unreachable. For a push-to-talk button that is nearly always
fine, and correctness of the sim's FFB matters far more than reaching button 90.

Devices are identified by manufacturer/product ID plus name rather than by
slot, because the slot is reassigned when a device is power-cycled — which is
exactly the case that used to lose push-to-talk.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from dataclasses import dataclass

log = logging.getLogger(__name__)

JOYERR_NOERROR = 0
JOY_RETURNBUTTONS = 0x00000080
MAX_BUTTONS = 32


class JOYCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("szPname", wintypes.WCHAR * 32),
        ("wXmin", wintypes.UINT),
        ("wXmax", wintypes.UINT),
        ("wYmin", wintypes.UINT),
        ("wYmax", wintypes.UINT),
        ("wZmin", wintypes.UINT),
        ("wZmax", wintypes.UINT),
        ("wNumButtons", wintypes.UINT),
        ("wPeriodMin", wintypes.UINT),
        ("wPeriodMax", wintypes.UINT),
        ("wRmin", wintypes.UINT),
        ("wRmax", wintypes.UINT),
        ("wUmin", wintypes.UINT),
        ("wUmax", wintypes.UINT),
        ("wVmin", wintypes.UINT),
        ("wVmax", wintypes.UINT),
        ("wCaps", wintypes.UINT),
        ("wMaxAxes", wintypes.UINT),
        ("wNumAxes", wintypes.UINT),
        ("wMaxButtons", wintypes.UINT),
        ("szRegKey", wintypes.WCHAR * 32),
        ("szOEMVxD", wintypes.WCHAR * 260),
    ]


class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("dwXpos", wintypes.DWORD),
        ("dwYpos", wintypes.DWORD),
        ("dwZpos", wintypes.DWORD),
        ("dwRpos", wintypes.DWORD),
        ("dwUpos", wintypes.DWORD),
        ("dwVpos", wintypes.DWORD),
        ("dwButtons", wintypes.DWORD),
        ("dwButtonNumber", wintypes.DWORD),
        ("dwPOV", wintypes.DWORD),
        ("dwReserved1", wintypes.DWORD),
        ("dwReserved2", wintypes.DWORD),
    ]


def _winmm():
    try:
        return ctypes.WinDLL("winmm")
    except (OSError, AttributeError) as exc:  # pragma: no cover - Windows only
        raise RuntimeError("winmm unavailable (Windows only)") from exc


@dataclass(frozen=True)
class WinmmDevice:
    slot: int
    device_id: str
    name: str
    button_count: int

    def __str__(self) -> str:
        return f"{self.name} ({self.button_count} buttons)"


def _device_id(caps: JOYCAPSW) -> str:
    """Stable identity across replugs.

    Manufacturer and product IDs plus the product name. The slot deliberately
    plays no part — it is reassigned on power-cycle, which is the failure this
    identity exists to survive.
    """
    return f"{caps.wMid:04x}:{caps.wPid:04x}:{caps.szPname}"


def enumerate_devices() -> list[WinmmDevice]:
    winmm = _winmm()
    out: list[WinmmDevice] = []
    for slot in range(winmm.joyGetNumDevs()):
        caps = JOYCAPSW()
        if winmm.joyGetDevCapsW(slot, ctypes.byref(caps), ctypes.sizeof(caps)) != JOYERR_NOERROR:
            continue  # slot empty
        if not caps.szPname:
            continue
        out.append(
            WinmmDevice(
                slot=slot,
                device_id=_device_id(caps),
                name=caps.szPname,
                button_count=caps.wNumButtons,
            )
        )
    return out


def resolve(device_id: str) -> WinmmDevice | None:
    """Find a device by its stable id, whatever slot it now occupies."""
    return next((d for d in enumerate_devices() if d.device_id == device_id), None)


def read_buttons(slot: int) -> int | None:
    """Button bitmask for a slot, or None if the device is gone.

    Read-only: no handle, no acquisition, nothing that could disturb another
    process's force feedback.
    """
    winmm = _winmm()
    info = JOYINFOEX()
    info.dwSize = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = JOY_RETURNBUTTONS
    if winmm.joyGetPosEx(slot, ctypes.byref(info)) != JOYERR_NOERROR:
        return None
    return info.dwButtons


def is_button_down(slot: int, button_index: int) -> bool | None:
    """True/False, or None if the device is unreadable."""
    if not 0 <= button_index < MAX_BUTTONS:
        return None
    mask = read_buttons(slot)
    if mask is None:
        return None
    return bool(mask & (1 << button_index))
