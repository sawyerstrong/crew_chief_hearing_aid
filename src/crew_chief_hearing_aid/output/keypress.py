"""Synthetic keypress sink (Windows).

Sends hardware-style scan codes via SendInput rather than virtual-key codes.
CrewChief reads controllers and keyboard through DirectInput, which consumes
scan codes; a VK-only injection can be invisible to it. Both are sent by
default, which costs nothing and covers either reader.

F13-F24 are the intended targets: no physical keyboard emits them, so there is
no chance of colliding with a game binding or a Windows shortcut.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes

from ..intent.phrases import Intent

log = logging.getLogger(__name__)

# Set-1 scan codes. F13-F24 are not reliably derivable via MapVirtualKey on all
# keyboard layouts, so they are tabulated explicitly.
# Numpad is the working keyspace. F13-F24 were the original choice — no
# hardware emits them, so nothing could collide — but CrewChief has no mapping
# for them: injection is received (F12 binds) and F13 is simply not recognised.
#
# The numpad is the next-best thing: CrewChief maps it, and on a sim rig with a
# wheel and button box it is usually untouched. Extended-flag keys (NumpadEnter,
# NumpadDivide) are excluded — NumpadEnter shares scan code 0x1C with Return and
# is distinguished only by the extended flag, which is more risk than the extra
# slot is worth.
SCAN_CODES: dict[str, int] = {
    "NUMPAD0": 0x52, "NUMPAD1": 0x4F, "NUMPAD2": 0x50, "NUMPAD3": 0x51,
    "NUMPAD4": 0x4B, "NUMPAD5": 0x4C, "NUMPAD6": 0x4D, "NUMPAD7": 0x47,
    "NUMPAD8": 0x48, "NUMPAD9": 0x49,
    "NUMPADMULTIPLY": 0x37, "NUMPADSUBTRACT": 0x4A,
    "NUMPADADD": 0x4E, "NUMPADDECIMAL": 0x53,
    # Kept for diagnostics (send-key F12 is how the range was tested) and in
    # case a rig has spare F-keys. Not used by the shipped config: iRacing
    # claims most of them.
    "F1": 0x3B, "F2": 0x3C, "F3": 0x3D, "F4": 0x3E,
    "F5": 0x3F, "F6": 0x40, "F7": 0x41, "F8": 0x42,
    "F9": 0x43, "F10": 0x44, "F11": 0x57, "F12": 0x58,
}

VIRTUAL_KEYS: dict[str, int] = {
    **{f"NUMPAD{i}": 0x60 + i for i in range(10)},  # VK_NUMPAD0..9
    "NUMPADMULTIPLY": 0x6A,
    "NUMPADADD": 0x6B,
    "NUMPADSUBTRACT": 0x6D,
    "NUMPADDECIMAL": 0x6E,
    **{f"F{i}": 0x70 + (i - 1) for i in range(1, 13)},  # VK_F1..VK_F12
}

# Friendlier spellings accepted in config.
_KEY_ALIASES = {
    **{f"NUM{i}": f"NUMPAD{i}" for i in range(10)},
    "NUMPADSTAR": "NUMPADMULTIPLY",
    "NUMPADMUL": "NUMPADMULTIPLY",
    "NUMPADPLUS": "NUMPADADD",
    "NUMPADMINUS": "NUMPADSUBTRACT",
    "NUMPADDOT": "NUMPADDECIMAL",
    "NUMPADPERIOD": "NUMPADDECIMAL",
}

# CrewChief exposes more bindable actions (25+) than there are F13-F24 keys (12),
# so combos are required for full coverage. Ctrl/Shift/Alt x 12 gives 48 slots.
MODIFIER_SCAN: dict[str, int] = {"ctrl": 0x1D, "shift": 0x2A, "alt": 0x38}
MODIFIER_VK: dict[str, int] = {"ctrl": 0x11, "shift": 0x10, "alt": 0x12}

_MODIFIER_ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "shft": "shift",
    "menu": "alt",
}


def parse_key(spec: str) -> tuple[tuple[str, ...], str]:
    """Split "ctrl+shift+F13" into (("ctrl", "shift"), "F13").

    Modifier order is normalised so that "shift+ctrl+F13" and "ctrl+shift+F13"
    are the same binding — otherwise the duplicate-key check in config loading
    would miss a genuine collision.
    """
    # Split on the LAST '+' only: numpad key names contain no '+', but
    # "NUMPADADD" spelled as "numpad+" would otherwise split wrongly.
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty key spec {spec!r}")
    key = parts[-1].upper()
    key = _KEY_ALIASES.get(key, key)
    mods: list[str] = []
    for raw in parts[:-1]:
        mod = _MODIFIER_ALIASES.get(raw, raw)
        if mod not in MODIFIER_SCAN:
            raise ValueError(f"unknown modifier {raw!r} in {spec!r}")
        if mod not in mods:
            mods.append(mod)
    # Canonical order, not the order typed.
    ordered = tuple(m for m in ("ctrl", "shift", "alt") if m in mods)
    return ordered, key


def normalize_key(spec: str) -> str:
    """Canonical form used for collision detection: "CTRL+SHIFT+F13"."""
    mods, key = parse_key(spec)
    return "+".join([*(m.upper() for m in mods), key])

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    """Never used — but its size defines the union's size, and therefore
    sizeof(INPUT). SendInput validates cbSize against its own sizeof(INPUT) and
    silently returns 0 on a mismatch, so omitting this member makes every call
    fail with no diagnostic. MOUSEINPUT is the largest member (32 bytes on x64
    vs KEYBDINPUT's 24)."""

    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


# use_last_error=True is required for get_last_error() to return anything.
# Without it every failure reports GetLastError=0, which is how a hard
# structure-size bug looked like a benign no-op.
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT


def _send(inputs: list[_INPUT]) -> int:
    arr = (_INPUT * len(inputs))(*inputs)
    sent = _user32.SendInput(len(inputs), arr, ctypes.sizeof(_INPUT))
    if sent != len(inputs):
        err = ctypes.get_last_error()
        hint = ""
        if err == 5:
            hint = (
                " — access denied: the foreground window belongs to an elevated "
                "process. Run this as administrator, or run CrewChief unelevated."
            )
        elif err == 87:
            hint = " — invalid parameter: sizeof(INPUT) mismatch"
        log.error(
            "SendInput delivered %d/%d events (GetLastError=%d)%s",
            sent,
            len(inputs),
            err,
            hint,
        )
    return sent


def _event(*, scan: int = 0, vk: int = 0, keyup: bool, scancode: bool) -> _INPUT:
    flags = KEYEVENTF_KEYUP if keyup else 0
    if scancode:
        flags |= KEYEVENTF_SCANCODE
        ki = _KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
    else:
        ki = _KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return _INPUT(type=INPUT_KEYBOARD, ki=ki)


class KeypressSink:
    def __init__(self, hold_ms: int = 150, *, send_scancode: bool = True, send_vk: bool = True):
        # CrewChief polls at hold_button_poll_frequency (100ms default), so a
        # hold shorter than roughly 120ms can fall between polls and be missed.
        self.hold_ms = max(hold_ms, 120)
        self.send_scancode = send_scancode
        self.send_vk = send_vk

    def preflight(self, intents: list[Intent]) -> list[str]:
        problems: list[str] = []
        for intent in intents:
            try:
                mods, key = parse_key(intent.key)
            except ValueError as exc:
                problems.append(f"intent {intent.id!r}: {exc}")
                continue
            if mods:
                # CrewChief stores a binding as action + deviceGuid +
                # buttonIndex. There is no modifier field, so a combo can be
                # sent but never bound — it would look like a working config
                # that silently never fires.
                problems.append(
                    f"intent {intent.id!r}: CrewChief cannot bind modifier combos "
                    f"({intent.key!r}); use a single key"
                )
                continue
            if self.send_scancode and key not in SCAN_CODES:
                problems.append(f"intent {intent.id!r}: no scan code known for key {key!r}")
            if self.send_vk and key not in VIRTUAL_KEYS:
                problems.append(f"intent {intent.id!r}: no virtual key known for {key!r}")
        if not hasattr(ctypes, "windll"):
            problems.append('keypress sink requires Windows; use sink = "log" elsewhere')
        return problems

    def fire(self, intent: Intent) -> bool:
        try:
            mods, key = parse_key(intent.key)
        except ValueError as exc:
            log.error("intent %s: %s", intent.id, exc)
            return False

        down: list[_INPUT] = []
        up: list[_INPUT] = []

        # Modifiers press first and release last, mirroring a real keyboard.
        for mod in mods:
            if self.send_scancode:
                down.append(_event(scan=MODIFIER_SCAN[mod], keyup=False, scancode=True))
            if self.send_vk:
                down.append(_event(vk=MODIFIER_VK[mod], keyup=False, scancode=False))

        if self.send_scancode and key in SCAN_CODES:
            down.append(_event(scan=SCAN_CODES[key], keyup=False, scancode=True))
            up.append(_event(scan=SCAN_CODES[key], keyup=True, scancode=True))
        if self.send_vk and key in VIRTUAL_KEYS:
            down.append(_event(vk=VIRTUAL_KEYS[key], keyup=False, scancode=False))
            up.append(_event(vk=VIRTUAL_KEYS[key], keyup=True, scancode=False))

        if not up:
            log.error("intent %s: key %r is not sendable", intent.id, intent.key)
            return False

        for mod in reversed(mods):
            if self.send_scancode:
                up.append(_event(scan=MODIFIER_SCAN[mod], keyup=True, scancode=True))
            if self.send_vk:
                up.append(_event(vk=MODIFIER_VK[mod], keyup=True, scancode=False))

        _send(down)
        time.sleep(self.hold_ms / 1000.0)
        _send(up)
        log.info("fired %s -> %s (%s)", intent.id, normalize_key(intent.key), intent.action)
        return True

    def close(self) -> None:
        return None
