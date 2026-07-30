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
SCAN_CODES: dict[str, int] = {
    "F13": 0x64, "F14": 0x65, "F15": 0x66, "F16": 0x67,
    "F17": 0x68, "F18": 0x69, "F19": 0x6A, "F20": 0x6B,
    "F21": 0x6C, "F22": 0x6D, "F23": 0x6E, "F24": 0x76,
    "F1": 0x3B, "F2": 0x3C, "F3": 0x3D, "F4": 0x3E,
    "F5": 0x3F, "F6": 0x40, "F7": 0x41, "F8": 0x42,
    "F9": 0x43, "F10": 0x44, "F11": 0x57, "F12": 0x58,
}

VIRTUAL_KEYS: dict[str, int] = {
    **{f"F{i}": 0x70 + (i - 1) for i in range(1, 13)},   # VK_F1..VK_F12
    **{f"F{i}": 0x7C + (i - 13) for i in range(13, 25)},  # VK_F13..VK_F24
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
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty key spec {spec!r}")
    key = parts[-1].upper()
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


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _send(inputs: list[_INPUT]) -> int:
    arr = (_INPUT * len(inputs))(*inputs)
    sent = ctypes.windll.user32.SendInput(len(inputs), arr, ctypes.sizeof(_INPUT))
    if sent != len(inputs):
        err = ctypes.get_last_error()
        log.error("SendInput delivered %d/%d events (GetLastError=%d)", sent, len(inputs), err)
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
                _mods, key = parse_key(intent.key)
            except ValueError as exc:
                problems.append(f"intent {intent.id!r}: {exc}")
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
