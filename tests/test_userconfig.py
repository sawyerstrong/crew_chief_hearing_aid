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
    def test_creates_a_thin_override_not_a_copy(self, user_config):
        import tomllib

        path = ensure_user_config()
        assert path.exists()
        # Parse rather than substring-match: the template *mentions*
        # [[intents]] in a comment explaining how to override them.
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
        assert set(doc) == {"audio", "ptt", "llm"}
        assert "intents" not in doc

    def test_shadow_detection(self, user_config):
        from crew_chief_hearing_aid.userconfig import shadows_shipped_intents

        ensure_user_config()
        assert not shadows_shipped_intents(user_config)
        user_config.write_text(
            user_config.read_text(encoding="utf-8")
            + '\n[[intents]]\nid = "x"\naction = "a"\nkey = "NUMPAD1"\n'
            'description = "d"\nphrases = ["p"]\n',
            encoding="utf-8",
        )
        assert shadows_shipped_intents(user_config)

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
        assert "# Filled in by" in text
        assert "MERGED OVER" in text

    def test_does_not_copy_the_shipped_defaults(self, user_config):
        """The user config must stay a thin override.

        Copying the defaults freezes the action list and key map at install
        time — a later change to the shipped config would be silently shadowed
        forever. This bit once already: a stale copy kept serving the old
        F13-F24 map after the shipped default moved to the numpad.
        """
        import tomllib

        set_values({"ptt": {"button_index": 4}})
        with user_config.open("rb") as fh:
            doc = tomllib.load(fh)
        assert "intents" not in doc
        assert "asr" not in doc

    def test_unset_sections_fall_through_to_defaults(self, user_config):
        from crew_chief_hearing_aid.config import load_config

        set_values({"ptt": {"button_index": 4}})
        config = load_config(user_path=user_config)
        assert config.get("asr", "model") == "tiny.en"
        assert config.intents  # from the shipped defaults, not the override

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
        # 14 numpad keys — CrewChief has no F13-F24 mapping and no modifier
        # support, so one action per recognised single key is the ceiling.
        assert len(config.intents) == 14


class TestDescribeChanges:
    def test_renders_a_pasteable_block(self):
        rendered = describe_changes(
            {"ptt": {"enabled": True, "device_guid": "abc", "button_index": 7}}
        )
        assert "[ptt]" in rendered
        assert "enabled = true" in rendered
        assert 'device_guid = "abc"' in rendered
        assert "button_index = 7" in rendered
