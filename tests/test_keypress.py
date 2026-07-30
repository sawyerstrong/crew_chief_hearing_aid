"""Keypress sink tests.

The struct-size test exists because getting it wrong is silent: SendInput
validates cbSize against its own sizeof(INPUT) and returns 0 without setting a
useful error. Shipped that way once — every injection failed as a no-op while
the log said GetLastError=0, because use_last_error was also unset.
"""

from __future__ import annotations

import ctypes
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

from crew_chief_hearing_aid.output import keypress as kp  # noqa: E402


class TestStructLayout:
    def test_input_matches_the_win32_definition(self):
        """sizeof(INPUT) is 40 on x64, 28 on x86.

        The union must be sized by its LARGEST member (MOUSEINPUT), not by the
        one we happen to use (KEYBDINPUT). Declaring only KEYBDINPUT yields 32
        on x64, which SendInput rejects — returning 0 events sent.
        """
        expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        assert ctypes.sizeof(kp._INPUT) == expected

    def test_union_is_sized_by_mouseinput(self):
        assert ctypes.sizeof(kp._MOUSEINPUT) > ctypes.sizeof(kp._KEYBDINPUT)
        assert ctypes.sizeof(kp._INPUTUNION) == ctypes.sizeof(kp._MOUSEINPUT)


class TestKeyTables:
    def test_f13_to_f24_all_have_scan_and_virtual_codes(self):
        for i in range(13, 25):
            key = f"F{i}"
            assert key in kp.SCAN_CODES, f"{key} missing a scan code"
            assert key in kp.VIRTUAL_KEYS, f"{key} missing a virtual key"

    def test_modifiers_have_both_encodings(self):
        assert set(kp.MODIFIER_SCAN) == set(kp.MODIFIER_VK)


class TestKeyParsing:
    def test_plain_key(self):
        assert kp.parse_key("F13") == ((), "F13")

    def test_single_modifier(self):
        assert kp.parse_key("ctrl+F13") == (("ctrl",), "F13")

    def test_modifier_order_is_normalised(self):
        """Otherwise the duplicate-binding check misses a real collision."""
        assert kp.normalize_key("shift+ctrl+F13") == kp.normalize_key("ctrl+shift+F13")

    def test_case_insensitive(self):
        assert kp.normalize_key("CTRL+f13") == kp.normalize_key("ctrl+F13")

    def test_aliases(self):
        assert kp.parse_key("control+F13") == (("ctrl",), "F13")

    def test_unknown_modifier_raises(self):
        with pytest.raises(ValueError, match="unknown modifier"):
            kp.parse_key("hyper+F13")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty key spec"):
            kp.parse_key("")


class TestPreflight:
    def test_accepts_every_configured_key(self):
        from pathlib import Path

        from crew_chief_hearing_aid.config import load_config

        config = load_config(user_path=Path("nonexistent-user-config.toml"))
        assert kp.KeypressSink().preflight(config.intents) == []

    def test_rejects_an_unknown_key(self):
        from crew_chief_hearing_aid.intent.phrases import Intent

        bad = Intent(id="x", action="a", key="F99", phrases=("p",), description="d")
        assert kp.KeypressSink().preflight([bad])
