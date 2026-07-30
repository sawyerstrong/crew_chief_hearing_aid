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


def _event(key: str, *, keyup: bool, scancode: bool) -> _INPUT:
    flags = KEYEVENTF_KEYUP if keyup else 0
    if scancode:
        flags |= KEYEVENTF_SCANCODE
        ki = _KEYBDINPUT(wVk=0, wScan=SCAN_CODES[key], dwFlags=flags, time=0, dwExtraInfo=0)
    else:
        ki = _KEYBDINPUT(wVk=VIRTUAL_KEYS[key], wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
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
            key = intent.key.upper()
            if self.send_scancode and key not in SCAN_CODES:
                problems.append(f"intent {intent.id!r}: no scan code known for key {intent.key!r}")
            if self.send_vk and key not in VIRTUAL_KEYS:
                problems.append(f"intent {intent.id!r}: no virtual key known for {intent.key!r}")
        if not hasattr(ctypes, "windll"):
            problems.append("keypress sink requires Windows; use sink = \"log\" elsewhere")
        return problems

    def fire(self, intent: Intent) -> bool:
        key = intent.key.upper()
        down: list[_INPUT] = []
        up: list[_INPUT] = []
        if self.send_scancode and key in SCAN_CODES:
            down.append(_event(key, keyup=False, scancode=True))
            up.append(_event(key, keyup=True, scancode=True))
        if self.send_vk and key in VIRTUAL_KEYS:
            down.append(_event(key, keyup=False, scancode=False))
            up.append(_event(key, keyup=True, scancode=False))
        if not down:
            log.error("intent %s: key %r is not sendable", intent.id, intent.key)
            return False

        _send(down)
        time.sleep(self.hold_ms / 1000.0)
        _send(up)
        log.info("fired %s -> %s (%s)", intent.id, key, intent.action)
        return True

    def close(self) -> None:
        return None
