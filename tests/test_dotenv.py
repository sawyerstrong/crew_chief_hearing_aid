"""`.env` loading.

The precedence test is the one that matters: a real environment variable must
win over the file. Getting that backwards means a stale .env silently overrides
what the operator exported, and the failure is invisible.

No test here ever asserts on a secret's value — only on names and provenance.
"""

from __future__ import annotations

import pytest

from crew_chief_hearing_aid.dotenv import (
    describe_source,
    find_dotenv,
    load,
    looks_like_anthropic_key,
    parse,
    set_value,
)


class TestParse:
    def test_basic(self):
        assert parse("FOO=bar") == {"FOO": "bar"}

    def test_export_prefix(self):
        assert parse("export FOO=bar") == {"FOO": "bar"}

    def test_quotes_are_stripped(self):
        assert parse('FOO="bar"') == {"FOO": "bar"}
        assert parse("FOO='bar'") == {"FOO": "bar"}

    def test_quoted_value_keeps_a_hash(self):
        """A '#' inside quotes is data, not a comment — and API keys can
        legitimately contain one."""
        assert parse('FOO="ab#cd"') == {"FOO": "ab#cd"}

    def test_unquoted_trailing_comment_is_stripped(self):
        assert parse("FOO=bar  # a note") == {"FOO": "bar"}

    def test_comments_and_blanks_ignored(self):
        assert parse("# note\n\nFOO=bar\n") == {"FOO": "bar"}

    def test_value_containing_equals(self):
        assert parse("FOO=a=b=c") == {"FOO": "a=b=c"}

    def test_empty_value(self):
        assert parse("FOO=") == {"FOO": ""}


class TestLoad:
    @pytest.fixture
    def env_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
        return p

    def test_sets_an_unset_variable(self, env_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert load(env_file) == ["ANTHROPIC_API_KEY"]
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "from-file"

    def test_real_environment_variable_wins(self, env_file, monkeypatch):
        """The load-bearing precedence rule.

        An exported variable is an explicit act; a .env is a per-machine
        convenience. Overriding the former would silently ignore the operator.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environment")
        assert load(env_file) == []
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "from-environment"

    def test_override_is_opt_in(self, env_file, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environment")
        assert load(env_file, override=True) == ["ANTHROPIC_API_KEY"]

    def test_empty_values_are_skipped(self, tmp_path, monkeypatch):
        """The shipped .env.example has a bare `ANTHROPIC_API_KEY=`; copying it
        without filling it in must not set an empty key that then fails at the
        API with a confusing error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        p = tmp_path / ".env"
        p.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
        assert load(p) == []

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load(tmp_path / "absent.env") == []


class TestDescribeSource:
    def test_reports_not_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert describe_source("ANTHROPIC_API_KEY", tmp_path / "absent") == "not set"

    def test_reports_environment(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        assert "environment" in describe_source("ANTHROPIC_API_KEY", tmp_path / "absent")

    def test_never_returns_the_value(self, monkeypatch, tmp_path):
        secret = "sk-ant-super-secret-value"
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
        p = tmp_path / ".env"
        p.write_text(f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8")
        described = describe_source("ANTHROPIC_API_KEY", p)
        assert secret not in described
        # Not even a suffix: for a project-scoped key the tail is the secret.
        assert secret[-6:] not in described


class TestSetValue:
    @pytest.fixture
    def env_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text(
            "# a comment\nOTHER=keepme\nANTHROPIC_API_KEY=\n", encoding="utf-8"
        )
        return p

    def test_replaces_in_place(self, env_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        set_value("ANTHROPIC_API_KEY", "sk-ant-newvalue", env_file)
        assert parse(env_file.read_text(encoding="utf-8")) == {
            "OTHER": "keepme",
            "ANTHROPIC_API_KEY": "sk-ant-newvalue",
        }

    def test_preserves_comments_and_other_keys(self, env_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        set_value("ANTHROPIC_API_KEY", "sk-ant-x", env_file)
        text = env_file.read_text(encoding="utf-8")
        assert "# a comment" in text
        assert "OTHER=keepme" in text

    def test_appends_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        p = tmp_path / ".env"
        p.write_text("OTHER=x\n", encoding="utf-8")
        set_value("ANTHROPIC_API_KEY", "sk-ant-y", p)
        assert parse(p.read_text(encoding="utf-8"))["ANTHROPIC_API_KEY"] == "sk-ant-y"

    def test_does_not_duplicate_on_repeat(self, env_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        set_value("ANTHROPIC_API_KEY", "sk-ant-1", env_file)
        set_value("ANTHROPIC_API_KEY", "sk-ant-2", env_file)
        occurrences = [
            ln
            for ln in env_file.read_text(encoding="utf-8").splitlines()
            if ln.startswith("ANTHROPIC_API_KEY=")
        ]
        assert len(occurrences) == 1

    def test_updates_the_live_environment(self, env_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        set_value("ANTHROPIC_API_KEY", "sk-ant-live", env_file)
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-live"

    def test_never_logs_the_value(self, env_file, monkeypatch, caplog):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        secret = "sk-ant-do-not-log-this-value"
        with caplog.at_level("DEBUG"):
            set_value("ANTHROPIC_API_KEY", secret, env_file)
        assert secret not in caplog.text
        assert secret[-8:] not in caplog.text


class TestKeyShapeCheck:
    def test_accepts_a_plausible_key(self):
        assert looks_like_anthropic_key("sk-ant-" + "x" * 30)

    def test_rejects_other_shapes(self):
        assert not looks_like_anthropic_key("sk-proj-abc123")
        assert not looks_like_anthropic_key("")
        assert not looks_like_anthropic_key("sk-ant-")  # too short

    def test_returns_only_a_bool(self):
        """Never a preview, prefix, or suffix — for a project-scoped key the
        tail is as sensitive as the whole string."""
        assert isinstance(looks_like_anthropic_key("sk-ant-" + "x" * 30), bool)


class TestNoSecretsInArgv:
    def test_there_is_no_api_key_flag(self):
        """A secret passed as a CLI argument is recorded in shell history,
        visible to `ps`, and captured by command logging. set-api-key prompts
        via getpass instead."""
        from crew_chief_hearing_aid.__main__ import build_parser

        help_text = build_parser().format_help()
        assert "--api-key" not in help_text
        assert "set-api-key" in help_text


class TestRepoHygiene:
    def test_env_is_gitignored(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        ignore = (root / ".gitignore").read_text(encoding="utf-8")
        assert "\n.env\n" in ignore
        assert "!.env.example" in ignore

    def test_example_exists_and_has_no_value(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        example = root / ".env.example"
        assert example.exists()
        assert parse(example.read_text(encoding="utf-8")) == {"ANTHROPIC_API_KEY": ""}

    def test_no_env_file_is_tracked_by_git(self):
        """Belt and braces: gitignore only helps if nothing was committed
        before it was added."""
        import pathlib
        import subprocess

        root = pathlib.Path(__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "ls-files", ".env", ".env.*"],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout.split()
        assert [f for f in out if f != ".env.example"] == []

    def test_find_dotenv_stays_inside_the_repo(self):
        found = find_dotenv()
        if found is not None:
            root = __import__("pathlib").Path(__file__).resolve().parent.parent
            assert root in found.parents or found.parent == root
