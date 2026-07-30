"""Input device enumeration and name-based resolution.

Device *indices* are deliberately never persisted. They reshuffle across
reboots and USB replugs, and a rig with several microphones plus NVIDIA
Broadcast's virtual endpoints will silently start recording from the wrong one.
Config stores a name substring; resolution happens at startup and fails loudly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool = False

    def __str__(self) -> str:
        marker = " (system default)" if self.is_default else ""
        return f"[{self.index}] {self.name} — {self.channels}ch @ {self.default_samplerate:.0f}Hz{marker}"


class DeviceResolutionError(RuntimeError):
    pass


def list_input_devices() -> list[InputDevice]:
    import sounddevice as sd

    try:
        default_index = sd.default.device[0]
    except (TypeError, IndexError):  # pragma: no cover - backend dependent
        default_index = None

    devices: list[InputDevice] = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) < 1:
            continue
        devices.append(
            InputDevice(
                index=index,
                name=info["name"],
                channels=info["max_input_channels"],
                default_samplerate=info.get("default_samplerate", 0.0),
                is_default=(index == default_index),
            )
        )
    return devices


def resolve_input_device(name_fragment: str | None) -> InputDevice:
    """Resolve a case-insensitive substring to exactly one input device.

    An empty fragment falls back to the system default, but logs loudly —
    "whatever Windows thinks is default" is how you end up transcribing a
    webcam microphone two feet from a direct-drive wheel.
    """
    devices = list_input_devices()
    if not devices:
        raise DeviceResolutionError("no audio input devices found")

    if not name_fragment:
        default = next((d for d in devices if d.is_default), devices[0])
        log.warning(
            "audio.input_device is unset; falling back to system default %r. "
            "Set it explicitly — the default is not stable across reboots.",
            default.name,
        )
        return default

    fragment = name_fragment.lower()
    matches = [d for d in devices if fragment in d.name.lower()]
    if not matches:
        available = "\n  ".join(str(d) for d in devices)
        raise DeviceResolutionError(
            f"no input device matching {name_fragment!r}. Available:\n  {available}"
        )
    if len(matches) > 1:
        # Ambiguity is resolved rather than raised: WASAPI/MME/DirectSound each
        # expose the same hardware under near-identical names, so a substring
        # legitimately hits several. Lowest index is the most stable pick.
        chosen = min(matches, key=lambda d: d.index)
        log.info(
            "%r matched %d devices; using %s",
            name_fragment,
            len(matches),
            chosen,
        )
        return chosen
    return matches[0]
