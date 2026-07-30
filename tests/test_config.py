import pytest

from crew_chief_hearing_aid.config import load_config

DEFAULT_TOML = """
[audio]
input_device = "Something"
sample_rate = 16000

[intent]
token_threshold = 0.72

[[intents]]
id = "a"
action = "A"
key = "F13"
phrases = ["do the first thing"]

[[intents]]
id = "b"
action = "B"
key = "F14"
phrases = ["do the second thing"]
"""


@pytest.fixture
def default_file(tmp_path):
    path = tmp_path / "config.default.toml"
    path.write_text(DEFAULT_TOML, encoding="utf-8")
    return path


def test_loads_defaults(default_file, tmp_path):
    config = load_config(user_path=tmp_path / "absent.toml", default_path=default_file)
    assert config.get("audio", "input_device") == "Something"
    assert len(config.intents) == 2
    assert config.intent_by_id("a").key == "F13"


def test_user_config_deep_merges(default_file, tmp_path):
    user = tmp_path / "config.toml"
    user.write_text('[audio]\ninput_device = "My Mic"\n', encoding="utf-8")
    config = load_config(user_path=user, default_path=default_file)
    assert config.get("audio", "input_device") == "My Mic"
    # Untouched keys in the same section survive the merge.
    assert config.get("audio", "sample_rate") == 16000
    assert config.get("intent", "token_threshold") == 0.72


def test_duplicate_output_key_is_fatal(default_file, tmp_path):
    """Two intents on one key means one silently never fires."""
    user = tmp_path / "config.toml"
    user.write_text(
        '[[intents]]\nid = "a"\naction = "A"\nkey = "F13"\nphrases = ["x"]\n'
        '[[intents]]\nid = "b"\naction = "B"\nkey = "f13"\nphrases = ["y"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both bound to"):
        load_config(user_path=user, default_path=default_file)


def test_missing_defaults_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(user_path=tmp_path / "no.toml", default_path=tmp_path / "missing.toml")


def test_shipped_default_config_is_valid(tmp_path):
    """The config in the repo must actually load — it is the onboarding path."""
    config = load_config(user_path=tmp_path / "no-user-config.toml")
    assert config.intents
    keys = [i.key.upper() for i in config.intents]
    assert len(keys) == len(set(keys))
    for intent in config.intents:
        assert intent.phrases, f"{intent.id} has no phrases"
        assert intent.action, f"{intent.id} has no CrewChief action label"
