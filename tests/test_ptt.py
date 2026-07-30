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


class TestJoystickButton:
    def test_readable_in_setup_output(self):
        button = JoystickButton("guid", "R3 Racing Wheel and Pedals", 7)
        assert str(button) == "R3 Racing Wheel and Pedals button 7"
