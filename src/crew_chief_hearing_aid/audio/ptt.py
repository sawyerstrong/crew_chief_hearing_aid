"""Push-to-talk on a wheel button (D9).

A keyboard key is unreachable by touch inside a headset, so PTT lives on the
wheel. One button is an accepted spend — the constraint was never "zero
buttons", it was "don't burn one per command".

Bound by **device GUID, not index**. Joystick indices reshuffle when USB
devices re-enumerate, exactly as audio device indices do; a GUID survives a
replug. Binding by index is how you end up with push-to-talk silently mapped to
a pedal set after a reboot.

Button release is the utterance endpoint, which is why v1 needs no VAD on the
critical path at all.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


class JoystickUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class JoystickButton:
    device_guid: str
    device_name: str
    button_index: int

    def __str__(self) -> str:
        return f"{self.device_name} button {self.button_index}"


def _require_pygame():
    # pygame prints a support banner to stdout on import, which would corrupt
    # the output of `setup-ptt` and `bindings`. Must be set before the import.
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise JoystickUnavailable(
            "push-to-talk needs a joystick backend: pip install pygame"
        ) from exc
    return pygame


def _init(pygame) -> None:
    # SDL routes joystick state through the event queue, so `event.pump()` --
    # and therefore `get_button()` ever changing -- requires the video
    # subsystem. Initialising it normally would open a window and steal focus
    # from the sim, so use the dummy driver: a real event queue, no window.
    #
    # Learned by running it: joystick.init() alone raises
    # "video system not initialized" on the first pump().
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if not pygame.display.get_init():
        pygame.display.init()
    pygame.joystick.quit()
    pygame.joystick.init()


def list_joysticks() -> list[tuple[str, str, int]]:
    """(guid, name, button_count) for each connected device."""
    pygame = _require_pygame()
    _init(pygame)
    out: list[tuple[str, str, int]] = []
    for i in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(i)
        js.init()
        out.append((js.get_guid(), js.get_name(), js.get_numbuttons()))
    return out


def capture_button(timeout_s: float = 30.0, poll_hz: int = 60) -> JoystickButton | None:
    """Block until a button is pressed, and report which one.

    The setup flow: prompt, watch every connected device, take the first
    button-down. Same shape as CrewChief's own binding dialog, so it should
    feel familiar.
    """
    pygame = _require_pygame()
    _init(pygame)

    count = pygame.joystick.get_count()
    if count == 0:
        raise JoystickUnavailable("no joystick or wheel detected")

    sticks = []
    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        sticks.append(js)

    # Baseline the current state so a button already held when setup starts is
    # not captured instantly.
    pygame.event.pump()
    held = {
        (js.get_guid(), b)
        for js in sticks
        for b in range(js.get_numbuttons())
        if js.get_button(b)
    }

    deadline = time.monotonic() + timeout_s
    interval = 1.0 / poll_hz
    while time.monotonic() < deadline:
        pygame.event.pump()
        for js in sticks:
            guid = js.get_guid()
            for b in range(js.get_numbuttons()):
                pressed = bool(js.get_button(b))
                if pressed and (guid, b) not in held:
                    return JoystickButton(guid, js.get_name(), b)
                if not pressed:
                    held.discard((guid, b))
        time.sleep(interval)
    return None


class WheelPTT:
    """Polls one wheel button. `is_down()` is the whole interface."""

    def __init__(self, device_guid: str, button_index: int) -> None:
        self.device_guid = device_guid
        self.button_index = button_index
        self._pygame = None
        self._stick = None
        self._device_name = "<unresolved>"

    def open(self) -> None:
        """Resolve the GUID to a live device. Raises if it cannot be found."""
        pygame = _require_pygame()
        _init(pygame)
        self._pygame = pygame

        available: list[str] = []
        for i in range(pygame.joystick.get_count()):
            js = pygame.joystick.Joystick(i)
            js.init()
            available.append(f"{js.get_name()} ({js.get_guid()})")
            if js.get_guid() == self.device_guid:
                if self.button_index >= js.get_numbuttons():
                    raise JoystickUnavailable(
                        f"{js.get_name()} has {js.get_numbuttons()} buttons; "
                        f"config asks for index {self.button_index}"
                    )
                self._stick = js
                self._device_name = js.get_name()
                log.info("push-to-talk on %s button %d", self._device_name, self.button_index)
                return

        # AC3.5: loud, with the list, never a silent no-op.
        raise JoystickUnavailable(
            f"no device with guid {self.device_guid!r}. Connected:\n  "
            + ("\n  ".join(available) if available else "(none)")
        )

    def is_down(self) -> bool:
        if self._stick is None or self._pygame is None:
            return False
        self._pygame.event.pump()
        return bool(self._stick.get_button(self.button_index))

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.joystick.quit()
            self._pygame = None
            self._stick = None


class NullPTT:
    """Stand-in when PTT is disabled or unavailable. Always up."""

    def open(self) -> None:
        return None

    def is_down(self) -> bool:
        return False

    def close(self) -> None:
        return None
