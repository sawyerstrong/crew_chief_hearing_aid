"""Push-to-talk tests.

Only the paths that do not need a physical wheel. Button-down detection,
release timing, and the min_hold gate are hardware-verified at the rig
(AC3.1-3.5), not here — CI has no joystick.
"""

from __future__ import annotations

import pytest

from crew_chief_hearing_aid.audio.ptt import (
    JoystickButton,
    JoystickUnavailable,
    NullPTT,
    WheelPTT,
)


class TestNullPTT:
    """The degraded path. A missing wheel must never crash the pipeline."""

    def test_is_always_up(self):
        ptt = NullPTT()
        ptt.open()
        assert ptt.is_down() is False
        ptt.close()


class TestWheelPTT:
    def test_unopened_reports_up_rather_than_raising(self):
        """A pipeline frame must never raise because PTT was not opened."""
        assert WheelPTT("some-guid", 3).is_down() is False

    def test_unresolvable_guid_raises_with_the_available_list(self, monkeypatch):
        """AC3.5: loud, actionable failure — never a silent no-op.

        The whole point of binding by GUID is that a re-enumerated wheel is
        detected rather than silently mapping PTT onto a different device.
        """
        import crew_chief_hearing_aid.audio.ptt as ptt_mod

        class _FakeJoystick:
            def __init__(self, _i):
                pass

            def init(self):
                pass

            def get_guid(self):
                return "a-different-wheel"

            def get_name(self):
                return "Some Other Wheel"

            def get_numbuttons(self):
                return 12

        class _FakePygame:
            class display:
                @staticmethod
                def get_init():
                    return True

                @staticmethod
                def init():
                    pass

            class joystick:
                @staticmethod
                def quit():
                    pass

                @staticmethod
                def init():
                    pass

                @staticmethod
                def get_count():
                    return 1

                Joystick = _FakeJoystick

        monkeypatch.setattr(ptt_mod, "_require_pygame", lambda: _FakePygame)

        with pytest.raises(JoystickUnavailable) as exc:
            WheelPTT("the-guid-i-was-bound-to", 3).open()
        assert "Some Other Wheel" in str(exc.value)

    def test_button_index_beyond_device_raises(self, monkeypatch):
        import crew_chief_hearing_aid.audio.ptt as ptt_mod

        class _FakeJoystick:
            def __init__(self, _i):
                pass

            def init(self):
                pass

            def get_guid(self):
                return "matching-guid"

            def get_name(self):
                return "Small Wheel"

            def get_numbuttons(self):
                return 4

        class _FakePygame:
            class display:
                @staticmethod
                def get_init():
                    return True

                @staticmethod
                def init():
                    pass

            class joystick:
                @staticmethod
                def quit():
                    pass

                @staticmethod
                def init():
                    pass

                @staticmethod
                def get_count():
                    return 1

                Joystick = _FakeJoystick

        monkeypatch.setattr(ptt_mod, "_require_pygame", lambda: _FakePygame)

        with pytest.raises(JoystickUnavailable, match="4 buttons"):
            WheelPTT("matching-guid", 99).open()


class TestSdlHints:
    def test_ffb_hostile_drivers_are_disabled(self):
        """SDL's RawInput and HIDAPI joystick drivers take over the device in a
        way that disturbs DirectInput force feedback — a wheel goes light and
        floaty in the sim the moment this process opens it. We only read button
        state, so both are pure downside."""
        from crew_chief_hearing_aid.audio.ptt import _SDL_HINTS

        assert _SDL_HINTS["SDL_JOYSTICK_RAWINPUT"] == "0"
        assert _SDL_HINTS["SDL_JOYSTICK_HIDAPI"] == "0"

    def test_background_events_are_enabled(self):
        """Not optional: the sim has focus while racing, so without this the
        button state never updates."""
        from crew_chief_hearing_aid.audio.ptt import _SDL_HINTS

        assert _SDL_HINTS["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] == "1"

    def test_hints_are_applied_before_pygame_is_imported(self, monkeypatch):
        import crew_chief_hearing_aid.audio.ptt as ptt_mod
        from crew_chief_hearing_aid.audio.ptt import _SDL_HINTS

        for name in _SDL_HINTS:
            monkeypatch.delenv(name, raising=False)
        ptt_mod._require_pygame()
        import os

        for name, value in _SDL_HINTS.items():
            assert os.environ.get(name) == value


class TestReconnect:
    """The wheel being powered off must not permanently kill push-to-talk."""

    def _ptt_with_failing_stick(self, monkeypatch, reopen_ok: bool):
        import crew_chief_hearing_aid.audio.ptt as ptt_mod

        class _DeadStick:
            def get_button(self, _i):
                raise RuntimeError("Invalid joystick device number")

        ptt = WheelPTT("guid", 3)
        ptt._pygame = type("P", (), {"event": type("E", (), {"pump": staticmethod(lambda: None)})})
        ptt._stick = _DeadStick()
        ptt._was_connected = True

        calls = {"n": 0}

        def fake_open(self):
            calls["n"] += 1
            if not reopen_ok:
                raise JoystickUnavailable("still gone")
            self._stick = type("S", (), {"get_button": staticmethod(lambda _i: True)})()
            self._device_name = "R3"

        monkeypatch.setattr(ptt_mod.WheelPTT, "open", fake_open)
        return ptt, calls

    def test_lost_device_reports_up_rather_than_raising(self, monkeypatch):
        """An exception here would propagate into the audio loop and stop the
        pipeline over a button."""
        ptt, _ = self._ptt_with_failing_stick(monkeypatch, reopen_ok=False)
        assert ptt.is_down() is False

    def test_attempts_reconnect_after_loss(self, monkeypatch):
        ptt, calls = self._ptt_with_failing_stick(monkeypatch, reopen_ok=False)
        ptt.is_down()
        assert calls["n"] == 1

    def test_reconnect_is_rate_limited(self, monkeypatch):
        """Re-enumeration is not free; retrying on every 32ms frame would burn
        the CPU budget the pipeline needs."""
        ptt, calls = self._ptt_with_failing_stick(monkeypatch, reopen_ok=False)
        for _ in range(20):
            ptt.is_down()
        assert calls["n"] == 1

    def test_recovers_when_the_device_returns(self, monkeypatch):
        ptt, _ = self._ptt_with_failing_stick(monkeypatch, reopen_ok=True)
        assert ptt.is_down() is False  # first call notices the loss
        assert ptt.is_down() is True  # reconnected handle works


class TestJoystickButton:
    def test_readable_in_setup_output(self):
        button = JoystickButton("guid", "R3 Racing Wheel and Pedals", 7)
        assert str(button) == "R3 Racing Wheel and Pedals button 7"
