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


# SDL hints, all set before pygame is imported — SDL reads them at init and
# ignores later changes.
#
# RAWINPUT and HIDAPI are the important two. SDL 2.0.16+ enables both by
# default, and they take over the device in a way that disturbs DirectInput
# force feedback: a wheel goes light and floaty in the sim the moment this
# process opens it. We only ever read button state, so both drivers are pure
# downside here — the plain DirectInput path reads buttons fine and leaves
# FFB alone.
#
# ALLOW_BACKGROUND_EVENTS is required rather than optional: the sim has focus
# while racing, so without it button state never updates.
_SDL_HINTS = {
    "SDL_JOYSTICK_RAWINPUT": "0",
    "SDL_JOYSTICK_HIDAPI": "0",
    "SDL_JOYSTICK_THREAD": "1",
    "SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS": "1",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
}


def _require_pygame():
    for name, value in _SDL_HINTS.items():
        os.environ.setdefault(name, value)
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
    """Polls one wheel button. `is_down()` is the whole interface.

    Survives the wheel being power-cycled: the SDL handle goes stale on
    disconnect, so a lost device is re-resolved by GUID in the background.
    Binding by GUID rather than index is what makes that possible — after a
    replug the index may differ, but the GUID is the same wheel.
    """

    # Re-enumeration is not free, so back off between attempts rather than
    # retrying on every 32ms audio frame.
    RECONNECT_INTERVAL_S = 2.0

    def __init__(self, device_guid: str, button_index: int) -> None:
        self.device_guid = device_guid
        self.button_index = button_index
        self._pygame = None
        self._stick = None
        self._device_name = "<unresolved>"
        self._last_reconnect = 0.0
        self._was_connected = False

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

    def _try_reconnect(self) -> None:
        """Re-resolve the GUID after a disconnect. Never raises."""
        now = time.monotonic()
        if now - self._last_reconnect < self.RECONNECT_INTERVAL_S:
            return
        self._last_reconnect = now
        try:
            self.open()
            log.info("push-to-talk reconnected to %s", self._device_name)
        except JoystickUnavailable:
            pass  # still gone; try again after the interval

    def is_down(self) -> bool:
        if self._stick is None or self._pygame is None:
            self._try_reconnect()
            return False

        try:
            self._pygame.event.pump()
            pressed = bool(self._stick.get_button(self.button_index))
        except Exception:  # noqa: BLE001 - a lost device raises from SDL
            # Powering the wheel off invalidates the handle. Drop it and let
            # the backoff re-resolve; never propagate into the audio loop.
            if self._was_connected:
                log.warning("push-to-talk device lost; will reconnect")
            self._stick = None
            self._was_connected = False
            self._try_reconnect()
            return False

        self._was_connected = True
        return pressed

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
