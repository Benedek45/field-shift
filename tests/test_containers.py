"""Docker construction is tested without Docker, a shell, credentials or a provider."""

import csv
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fieldshift.agent import chat, containers, pilot, providers


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "run with spaces, and commas"
    for name in ("cfg", "data", "work"):
        (root / name).mkdir(parents=True)
    return root


def test_default_context_and_docker_host_are_left_to_docker(run_dir):
    env = {"DOCKER_HOST": "tcp://engine.example:2376", "DOCKER_CONTEXT": "current"}
    command = containers.run_command(run_dir, ["--version"], environ=env)
    assert command[:2] == ["docker", "run"]
    assert "--context" not in command and containers.DEFAULT_IMAGE in command
    assert env == {"DOCKER_HOST": "tcp://engine.example:2376", "DOCKER_CONTEXT": "current"}


def test_explicit_context_applies_to_launch_inspect_and_stop(run_dir):
    env = {"FIELDSHIFT_DOCKER_CONTEXT": "my-server", "FIELDSHIFT_IMAGE": "example/agent:local"}
    command = containers.run_command(run_dir, ["--version"], environ=env)
    assert command[:4] == ["docker", "--context", "my-server", "run"]
    assert "example/agent:local" in command
    for action in (("stop", "owned-run"), ("image", "inspect", containers.image_name(env))):
        assert containers.docker_command(*action, environ=env) == ["docker", "--context", "my-server", *action]


def test_only_three_absolute_mounts_with_spaces_and_commas(run_dir, monkeypatch):
    monkeypatch.chdir(run_dir.parent)
    command = containers.run_command(Path(run_dir.name), ["run", "hello; $(not-a-shell)"], environ={})
    mounts = [command[i + 1] for i, word in enumerate(command) if word == "--mount"]
    assert len(mounts) == 3
    for text, directory in zip(mounts, ("cfg", "data", "work"), strict=True):
        assert next(csv.reader([text])) == [
            "type=bind", f"src={run_dir.resolve() / directory}", f"dst=/{directory}"]
    assert command[-1] == "hello; $(not-a-shell)"


def test_host_gateway_and_writable_host_uid(run_dir):
    command = containers.run_command(run_dir, ["--version"], environ={})
    assert command[command.index("--add-host") + 1] == "host.docker.internal:host-gateway"
    if hasattr(os, "getuid"):
        assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "HOME=/tmp/fieldshift-home" in command


def test_windows_without_posix_uid_uses_image_user(run_dir, monkeypatch):
    monkeypatch.delattr(containers.os, "getuid", raising=False)
    assert "--user" not in containers.run_command(run_dir, ["--version"], environ={})


def test_environment_values_never_enter_argv(run_dir):
    env = {"FIELDSHIFT_PASS_ENV": "API_KEY FARMSIM_TOKEN API_KEY", "API_KEY": "synthetic-secret",
           "FARMSIM_TOKEN": "synthetic-service-secret"}
    command = containers.run_command(run_dir, ["run", "hi"], environ=env)
    names = [command[i + 1] for i, word in enumerate(command) if word == "-e"]
    assert names == ["API_KEY", "FARMSIM_TOKEN"]
    assert "synthetic-secret" not in " ".join(command)
    assert "synthetic-service-secret" not in " ".join(command)


@pytest.mark.parametrize("name", ["A=b", "-e", "9KEY", "$(id)", "A;id", "A.B", "A/B", "éKEY"])
def test_malicious_pass_environment_names_rejected(run_dir, name):
    with pytest.raises(ValueError, match="names only"):
        containers.run_command(run_dir, [], environ={"FIELDSHIFT_PASS_ENV": name})


@pytest.mark.parametrize("flag", ["FIELDSHIFT_ANTHROPIC", "FIELDSHIFT_OPENAI"])
def test_private_subscription_adapters_remain_unavailable(run_dir, flag):
    with pytest.raises(ValueError, match="Subscription-login"):
        containers.run_command(run_dir, [], environ={flag: "1"})


def test_missing_mount_rejected(tmp_path):
    with pytest.raises(ValueError, match="cfg/"):
        containers.run_command(tmp_path, [], environ={})


def test_container_name_is_one_safe_argument(run_dir):
    env = {"FIELDSHIFT_CONTAINER_NAME": "fieldshift-owned-123"}
    command = containers.run_command(run_dir, [], environ=env)
    assert command[command.index("--name") + 1] == "fieldshift-owned-123"
    with pytest.raises(ValueError, match="container name"):
        containers.run_command(run_dir, [], environ={"FIELDSHIFT_CONTAINER_NAME": "--privileged"})


def test_pilot_timeout_stops_in_the_launch_context(tmp_path, monkeypatch):
    route = providers.resolve("custom/test", {"FIELDSHIFT_BASE_URL": "https://example.invalid/v1"})
    monkeypatch.setenv("FIELDSHIFT_DOCKER_CONTEXT", "episode-context")
    monkeypatch.setenv("DOCKER_HOST", "tcp://example.invalid:2376")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["shell"] is False
        if "run" in command:
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot.subprocess, "run", run)
    args = SimpleNamespace(out=tmp_path, redo=False, evidence_root=Path(__file__).resolve().parents[1] / "data/evidence",
                           agent_prompt="farm-advisor.md", container_service="http://host.docker.internal:9", timeout=1)
    job = {"case": {"id": "timeout"}, "setup": "llm", "model": route.model, "rep": 0, "farm_id": "f1"}
    result = pilot.run_one(job, args, "brief", "synthetic-token", route)
    assert result["status"] == "timeout" and len(calls) == 2
    launch, stop = calls
    assert launch[0][:3] == stop[0][:3] == ["docker", "--context", "episode-context"]
    assert stop[0][3] == "stop"
    assert launch[1]["env"] == stop[1]["env"]
    assert stop[1]["env"]["DOCKER_HOST"] == "tcp://example.invalid:2376"


def test_chat_timeout_uses_same_context_and_still_scrubs(run_dir, monkeypatch):
    monkeypatch.setenv("FIELDSHIFT_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FIELDSHIFT_API_KEY", "synthetic-timeout-test-key")
    monkeypatch.setenv("FIELDSHIFT_DOCKER_CONTEXT", "chat-context")
    agent = chat.OpenCodeAgent("custom/test", 1)
    agent.prepare(run_dir, "brief")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["shell"] is False
        if "run" in command:
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chat.subprocess, "run", run)
    result = agent.turn(run_dir, "hello", session_id=None, keep_session=False,
                        farm_id="f1", port=9, token="synthetic-token")
    assert result["status"] == "timeout" and result["credential_scrub"]["verified_clean"]
    assert len(calls) == 2
    assert calls[0][0][:3] == calls[1][0][:3] == ["docker", "--context", "chat-context"]
    assert calls[1][0][3] == "stop" and calls[0][1]["env"] == calls[1][1]["env"]
    assert not (run_dir / ".turn-in-progress").exists()


def test_chat_scrubs_even_when_docker_cleanup_fails(run_dir, monkeypatch):
    monkeypatch.setenv("FIELDSHIFT_BASE_URL", "https://example.invalid/v1")
    agent = chat.OpenCodeAgent("custom/test", 1)
    agent.prepare(run_dir, "brief")
    scrubbed = []
    scrub = chat.scrub_credentials

    def run(command, **kwargs):
        raise FileNotFoundError("Docker executable unavailable")

    def recording_scrub(*args, **kwargs):
        scrubbed.append(True)
        return scrub(*args, **kwargs)

    monkeypatch.setattr(chat.subprocess, "run", run)
    monkeypatch.setattr(chat, "scrub_credentials", recording_scrub)
    with pytest.raises(FileNotFoundError, match="Docker executable"):
        agent.turn(run_dir, "hello", session_id=None, keep_session=False,
                   farm_id="f1", port=9, token="synthetic-token")
    assert scrubbed == [True] and not (run_dir / ".turn-in-progress").exists()
