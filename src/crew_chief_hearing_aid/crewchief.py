"""Read-only introspection of an installed CrewChief V4.

Used for preflight and for importing CrewChief's own phrase list. Nothing here
writes to CrewChief's files.

CrewChief stores its settings in a ClickOnce user.config under
%LOCALAPPDATA%\\Britton_IT_Ltd\\CrewChiefV4.exe_Url_<hash>\\<version>\\user.config
and reads a user phrase override from %LOCALAPPDATA%\\CrewChiefV4\\.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)$")


def local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))


def sound_pack_dir() -> Path:
    return local_appdata() / "CrewChiefV4"


def user_phrase_override_path() -> Path:
    """Where CrewChief looks for a user-supplied phrase file.

    Dropping a curated copy here is the zero-code fix for short greedy aliases
    ("lap time", "gap ahead") cannibalising longer commands.
    """
    return sound_pack_dir() / "speech_recognition_config.txt"


def _version_key(name: str) -> tuple[int, ...]:
    m = _VERSION_RE.match(name)
    return tuple(int(g) for g in m.groups()) if m else (0, 0, 0, 0)


def find_user_config() -> Path | None:
    """Locate the newest CrewChief user.config, or None if not installed."""
    root = local_appdata() / "Britton_IT_Ltd"
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for app_dir in root.glob("CrewChiefV4.exe_Url_*"):
        for version_dir in app_dir.iterdir():
            if version_dir.is_dir() and (version_dir / "user.config").is_file():
                candidates.append(version_dir)
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: _version_key(p.name))
    return newest / "user.config"


def read_settings(path: Path | None = None) -> dict[str, str]:
    """Parse user.config into {setting_name: value}."""
    path = path or find_user_config()
    if path is None or not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for setting in root.iter("setting"):
        name = setting.get("name")
        if not name:
            continue
        value_el = setting.find("value")
        out[name] = (value_el.text or "") if value_el is not None else ""
    return out


@dataclass(frozen=True)
class BindingReport:
    bound: dict[str, str]
    unbound: list[str]

    @property
    def any_bound(self) -> bool:
        return bool(self.bound)


def binding_report(settings: dict[str, str] | None = None) -> BindingReport:
    """Which CrewChief actions currently have a controller/key assigned.

    An index of -1 with an empty device guid means unassigned. The action-name
    to setting-name mapping is not fully documented, so this reports every
    `*_button_index` setting it finds rather than guessing at a lookup table —
    read it alongside CrewChief's Add/Remove Actions dialog.
    """
    settings = settings if settings is not None else read_settings()
    bound: dict[str, str] = {}
    unbound: list[str] = []
    for name, value in settings.items():
        if not name.endswith("_button_index"):
            continue
        action = name[: -len("_button_index")]
        guid = settings.get(f"{action}_device_guid", "").strip()
        try:
            index = int(value)
        except ValueError:
            index = -1
        if index >= 0 or guid:
            bound[action] = f"index={index} guid={guid or '-'}"
        else:
            unbound.append(action)
    return BindingReport(bound=bound, unbound=sorted(unbound))


def recognition_health(settings: dict[str, str] | None = None) -> list[str]:
    """Config smells that degrade CrewChief's own recogniser.

    Reported as advice, not errors — crew_chief_hearing_aid bypasses CrewChief's SRE entirely,
    but if you still use it these are the settings that matter.
    """
    settings = settings if settings is not None else read_settings()
    if not settings:
        return ["CrewChief settings not found — is it installed?"]

    notes: list[str] = []
    if not settings.get("NAUDIO_RECORDING_DEVICE_GUID", "").strip():
        notes.append(
            "CrewChief speech input device is unset; it will read the Windows default "
            "device regardless of what the dropdown shows."
        )
    try:
        threshold = float(settings.get("minimum_voice_recognition_confidence_system_sre", "0"))
        if threshold >= 0.75:
            notes.append(
                f"System SRE confidence threshold is {threshold:.2f}; identical utterances "
                "commonly score 0.65-0.99, so this rejects correct recognitions."
            )
    except ValueError:
        pass
    if settings.get("disable_alternative_voice_commands", "").lower() == "false":
        notes.append(
            "Alternative voice commands are enabled, so short aliases like 'lap time' "
            "can capture longer sentences and fire the wrong command at high confidence."
        )
    if settings.get("CHANNEL_OPEN_FUNCTION_button_index", "-1") == "-1" and not settings.get(
        "CHANNEL_OPEN_FUNCTION_device_guid", ""
    ).strip():
        notes.append("CrewChief push-to-talk button is unassigned.")
    return notes
