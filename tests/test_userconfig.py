"""User-config round-tripping.

Setup writes values the user later has to read and reason about — which mic,
which wheel button. Stripping the explanatory comments the first time setup
touched the file would be actively hostile, so comment preservation is a
tested property, not an implementation detail.
"""

from __future__ import annotations

import pytest

from crew_chief_hearing_aid.userconfig import (
    describe_changes,
    ensure_user_config,
    set_values,
)


@pytest.fixture
def user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path / "crew_chief_hearing_aid" / "config.toml"


class TestEnsure:
    def test_creates_from_shipped_defaults(self, user_config):
        path = ensure_user_config()
        assert path.exists()
        assert "[[intents]]" in path.read_text(encoding="utf-8")

    def test_is_idempotent_and_does_not_clobber(self, user_config):
        ensure_user_config()
        set_values({"ptt": {"button_index": 9}})
        ensure_user_config()
        assert "button_index = 9" in user_config.read_text(encoding="utf-8")


class TestSetValues:
    def test_writes_scalar_types_correctly(self, user_config):
        set_values(
            {"ptt": {"enabled": True, "device_guid": "abc123", "button_index": 7}}
        )
        text = user_config.read_text(encoding="utf-8")
        assert "enabled = true" in text
        assert 'device_guid = "abc123"' in text
        assert "button_index = 7" in text

    def test_preserves_comments(self, user_config):
        set_values({"ptt": {"button_index": 4}})
        text = user_config.read_text(encoding="utf-8")
        assert "# Push-to-talk on a wheel button" in text
        assert "# Ignore taps shorter than this" in text

    def test_preserves_unrelated_sections(self, user_config):
        set_values({"ptt": {"button_index": 4}})
        text = user_config.read_text(encoding="utf-8")
        assert 'model = "tiny.en"' in text
        assert "[[intents]]" in text

    def test_creates_the_file_if_absent(self, user_config):
        assert not user_config.exists()
        set_values({"ptt": {"button_index": 1}})
        assert user_config.exists()

    def test_result_still_loads(self, user_config):
        """The written file must survive a real config load — a malformed write
        would only surface at the next launch, i.e. at the rig."""
        from crew_chief_hearing_aid.config import load_config

        set_values(
            {
                "ptt": {"enabled": True, "device_guid": "guid", "button_index": 3},
                "audio": {"input_device": "USB PnP"},
            }
        )
        config = load_config(user_path=user_config)
        assert config.get("ptt", "device_guid") == "guid"
        assert config.get("audio", "input_device") == "USB PnP"
        assert len(config.intents) == 27


class TestDescribeChanges:
    def test_renders_a_pasteable_block(self):
        rendered = describe_changes(
            {"ptt": {"enabled": True, "device_guid": "abc", "button_index": 7}}
        )
        assert "[ptt]" in rendered
        assert "enabled = true" in rendered
        assert 'device_guid = "abc"' in rendered
        assert "button_index = 7" in rendered
