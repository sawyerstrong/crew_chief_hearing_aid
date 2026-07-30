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
description = "The first thing."
phrases = ["do the first thing"]

[[intents]]
id = "b"
action = "B"
key = "F14"
description = "The second thing."
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
        '[[intents]]\nid = "a"\naction = "A"\nkey = "F13"\ndescription = "d"\n'
        'phrases = ["x"]\n'
        '[[intents]]\nid = "b"\naction = "B"\nkey = "f13"\ndescription = "d"\n'
        'phrases = ["y"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both bound to"):
        load_config(user_path=user, default_path=default_file, phrase_source={})


def test_modifier_order_collides(default_file, tmp_path):
    """ctrl+shift+F13 and shift+ctrl+F13 are the same binding.

    Without order normalisation the duplicate check misses a real collision and
    one of the two intents silently never fires.
    """
    user = tmp_path / "config.toml"
    user.write_text(
        '[[intents]]\nid = "a"\naction = "A"\nkey = "ctrl+shift+F13"\n'
        'description = "d"\nphrases = ["x"]\n'
        '[[intents]]\nid = "b"\naction = "B"\nkey = "shift+ctrl+F13"\n'
        'description = "d"\nphrases = ["y"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both bound to"):
        load_config(user_path=user, default_path=default_file, phrase_source={})


def test_phrases_import_from_sre_key(default_file, tmp_path):
    user = tmp_path / "config.toml"
    user.write_text(
        '[[intents]]\nid = "a"\naction = "A"\nkey = "F13"\ndescription = "d"\n'
        'sre_key = "HOWS_MY_FUEL"\n',
        encoding="utf-8",
    )
    config = load_config(
        user_path=user,
        default_path=default_file,
        phrase_source={"HOWS_MY_FUEL": ["how's my fuel", "how's my fuel level"]},
    )
    assert config.intent_by_id("a").phrases == ("how's my fuel", "how's my fuel level")


def test_unknown_sre_key_is_fatal(default_file, tmp_path):
    """A CrewChief update renaming a key must fail loudly, not import nothing."""
    user = tmp_path / "config.toml"
    user.write_text(
        '[[intents]]\nid = "a"\naction = "A"\nkey = "F13"\ndescription = "d"\n'
        'sre_key = "RENAMED_UPSTREAM"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RENAMED_UPSTREAM"):
        load_config(user_path=user, default_path=default_file, phrase_source={})


def test_no_phrases_and_no_sre_key_is_fatal(default_file, tmp_path):
    user = tmp_path / "config.toml"
    user.write_text(
        '[[intents]]\nid = "a"\naction = "A"\nkey = "F13"\ndescription = "d"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="neither phrases nor"):
        load_config(user_path=user, default_path=default_file, phrase_source={})


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
