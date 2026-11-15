"""Offline tests for model routing and key handling (no provider, no container)."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fieldshift.agent import chat, containers, pilot, providers
from fieldshift.app import advisor_core as core

FAKE = "FAKE-PROVIDER-KEY-" + "Qx9" * 20  # synthetic, never a real credential
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("model", "provider", "variable"), [
    ("opencode-go/gpt-6-luna", "opencode-go", "OPENCODE_API_KEY"),
    ("openrouter/anthropic/claude-sonnet-5.5", "openrouter", "OPENROUTER_API_KEY"),
    ("anthropic/claude-sonnet-5-5", "anthropic", "ANTHROPIC_API_KEY"),
    ("openai/gpt-6-luna", "openai", "OPENAI_API_KEY"),
    ("google/gemini-2.5-pro", "google", "GEMINI_API_KEY"),
    ("xai/grok-4.7", "xai", "XAI_API_KEY"),
    ("deepseek/deepseek-v4-flash", "deepseek", "DEEPSEEK_API_KEY"),
    ("mistral/devstral-latest", "mistral", "MISTRAL_API_KEY"),
    ("groq/openai/gpt-oss-120b", "groq", "GROQ_API_KEY"),
])
def test_catalog_routes_use_the_provider_variable(model, provider, variable):
    route = providers.resolve(model, {variable: FAKE})
    assert (route.provider, route.opencode_model, route.key_names) == (provider, model, (variable,))
    assert route.env == {variable: FAKE} and route.providers_config == {}
    assert FAKE not in repr(route) and FAKE not in json.dumps(route.describe())


def test_catalog_is_vendored_with_date_and_covers_the_main_providers():
    catalog = providers.load_catalog()
    assert catalog["source"] == "https://models.dev/api.json" and catalog["fetched"]
    for name in ("opencode-go", "openrouter", "anthropic", "openai", "google", "xai", "deepseek", "mistral", "groq"):
        assert catalog["providers"][name]["env"]


def test_go_alias_fills_the_variable_opencode_reads():
    route = providers.resolve("opencode-go/gpt-6-luna", {"OPENCODE_GO_API_KEY": FAKE})
    assert route.env == {"OPENCODE_API_KEY": FAKE} and route.key_names == ("OPENCODE_API_KEY",)


def test_openrouter_keeps_the_nested_model_id():
    route = providers.resolve("openrouter/anthropic/claude-sonnet-5.5", {"OPENROUTER_API_KEY": FAKE})
    assert route.provider == "openrouter"
    assert route.opencode_model == "openrouter/anthropic/claude-sonnet-5.5"
    assert providers.opencode_config(route)["small_model"] == "openrouter/anthropic/claude-sonnet-5.5"


def test_missing_key_names_the_variable_without_values():
    with pytest.raises(providers.MissingKeyError) as caught:
        providers.resolve("openrouter/openai/gpt-6-luna", {"ANTHROPIC_API_KEY": FAKE})
    assert "OPENROUTER_API_KEY" in str(caught.value) and FAKE not in str(caught.value)
    with pytest.raises(providers.MissingKeyError, match="OPENCODE_API_KEY, OPENCODE_GO_API_KEY"):
        providers.resolve("opencode-go/gpt-6-luna", {})
    with pytest.raises(providers.MissingKeyError, match="GOOGLE_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, GEMINI_API_KEY"):
        providers.resolve("google/gemini-2.5-pro", {})


@pytest.mark.parametrize(("model", "variable"), [("anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
                                                  ("openai/gpt-6-luna", "OPENAI_API_KEY")])
def test_anthropic_and_openai_need_an_api_key_not_a_subscription(model, variable):
    assert providers.resolve(model, {variable: FAKE}).env == {variable: FAKE}
    with pytest.raises(providers.SubscriptionModelError) as caught:
        providers.resolve(model, {})
    assert variable in str(caught.value) and "Subscription logins" in str(caught.value)
    assert pilot.SubscriptionModelError is providers.SubscriptionModelError


def test_unknown_and_unsupported_providers_are_explained():
    with pytest.raises(providers.UnknownProviderError, match="openrouter"):
        providers.resolve("openroutr/openai/gpt-6-luna", {})
    with pytest.raises(providers.ProviderError, match="custom route"):
        providers.resolve("vllm/qwen", {})
    with pytest.raises(providers.ProviderError, match="device login"):
        providers.resolve("github-copilot/gpt-6-luna", {"GITHUB_TOKEN": FAKE})
    with pytest.raises(providers.ProviderError, match="<provider>/<model id>"):
        providers.resolve("gpt-6-luna", {})


def test_custom_endpoint_block_has_names_only_and_reaches_the_host(tmp_path):
    env = {"FIELDSHIFT_BASE_URL": "http://localhost:11434/v1", "FIELDSHIFT_API_KEY": FAKE}
    route = providers.resolve("custom/qwen3.5:9b", env)
    assert route.opencode_model == "custom/qwen3.5-9b" and route.key_names == ("FIELDSHIFT_API_KEY",)
    block = route.providers_config["custom"]
    assert block["env"] == ["FIELDSHIFT_API_KEY"]
    assert block["package"] == "@opencode/ai/providers/openai-compatible"
    assert block["settings"]["baseURL"] == "http://host.docker.internal:11434/v1"
    assert block["models"] == {"qwen3.5-9b": {"modelID": "qwen3.5:9b", "name": "qwen3.5:9b"}}
    assert any("host.docker.internal" in n for n in route.notes)
    (tmp_path / "cfg/opencode").mkdir(parents=True)
    text = providers.write_config(tmp_path, route).read_text()
    assert FAKE not in text and json.loads(text)["small_model"] == "custom/qwen3.5-9b"


def test_custom_endpoint_without_key_and_remote_url():
    route = providers.resolve("custom/org/model-x", {"FIELDSHIFT_BASE_URL": "https://llm.example.org/v1"})
    assert route.key_names == () and route.env == {} and "env" not in route.providers_config["custom"]
    assert route.providers_config["custom"]["settings"]["baseURL"] == "https://llm.example.org/v1"
    assert route.providers_config["custom"]["models"]["org-model-x"]["modelID"] == "org/model-x"
    with pytest.raises(providers.MissingKeyError, match="FIELDSHIFT_BASE_URL"):
        providers.resolve("custom/model", {})
    with pytest.raises(providers.ProviderError, match="FIELDSHIFT_API_KEY"):
        providers.resolve("custom/model", {"FIELDSHIFT_BASE_URL": "http://user:pw@localhost:8000/v1"})


def test_local_discovery_points_at_the_host():
    route = providers.resolve("ollama/qwen3.5:9b", {})
    assert route.opencode_model == "ollama/qwen3.5:9b" and route.key_names == ()
    assert route.providers_config == {"ollama": {"settings": {"baseURL": "http://host.docker.internal:11434/v1"}}}


def test_key_files_are_read_and_the_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "KEY_FILES", ("fieldshift.env", ".env", ".agent-env"))
    (tmp_path / "fieldshift.env").write_text("OPENROUTER_API_KEY=from-fieldshift-env\nXAI_API_KEY=\n")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-dot-env\nGROQ_API_KEY=from-dot-env\n")
    merged = providers.key_environment({"GROQ_API_KEY": "from-environment"}, root=tmp_path)
    assert merged["OPENROUTER_API_KEY"] == "from-fieldshift-env" and merged["GROQ_API_KEY"] == "from-environment"
    assert "XAI_API_KEY" not in merged
    assert providers.detected_key_names(merged) == ["GROQ_API_KEY", "OPENROUTER_API_KEY"]


def test_key_file_example_lists_names_only_and_key_file_is_ignored():
    lines = [line for line in (REPO / "fieldshift.env.example").read_text().splitlines()
             if line and not line.startswith("#")]
    assert lines and all(line.endswith("=") for line in lines)
    assert {"OPENROUTER_API_KEY=", "OPENCODE_API_KEY=", "FIELDSHIFT_BASE_URL="} <= set(lines)
    ignored = subprocess.run(["git", "check-ignore", "-q", "fieldshift.env"], cwd=REPO, check=False)
    assert ignored.returncode == 0


# -- the key reaches the container only by name --------------------------------------------------

def _fake_launcher(monkeypatch, module, calls):
    real = subprocess.run

    def run(command, *args, **kwargs):
        if command[0] == "docker" and "run" in command:
            assert kwargs["shell"] is False
            calls.append({"argv": list(command), "env": dict(kwargs["env"])})
            kwargs["stdout"].write(json.dumps({"type": "text", "sessionID": "ses_fake",
                                               "part": {"text": "ok"}}) + "\n")
            return subprocess.CompletedProcess(command, 0)
        return real(command, *args, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", run)


def _files_containing(root, value):
    return [str(p) for p in Path(root).rglob("*") if p.is_file() and value.encode() in p.read_bytes()]


def test_pilot_passes_the_key_by_name_only(tmp_path, monkeypatch):
    calls = []
    _fake_launcher(monkeypatch, pilot, calls)
    route = providers.resolve("openrouter/anthropic/claude-sonnet-5.5", {"OPENROUTER_API_KEY": FAKE})
    args = SimpleNamespace(out=tmp_path, redo=False, evidence_root=REPO / "data/evidence", agent_prompt="farm-advisor.md",
                           container_service="http://host.docker.internal:9", timeout=30)
    job = {"case": {"id": "c"}, "setup": "full", "model": route.model, "rep": 0, "farm_id": "f1"}
    meta = pilot.run_one(job, args, "brief", "service-token", route)
    (call,) = calls
    assert FAKE not in " ".join(call["argv"])
    assert call["argv"][call["argv"].index("--model") + 1] == "openrouter/anthropic/claude-sonnet-5.5"
    assert call["env"]["OPENROUTER_API_KEY"] == FAKE
    assert call["env"]["FIELDSHIFT_PASS_ENV"].split() == ["OPENROUTER_API_KEY", "FARMSIM_URL", "FARMSIM_FARM",
                                                          "FARMSIM_TOKEN"]
    config = json.loads((tmp_path / meta["run"] / "cfg/opencode/opencode.json").read_text())
    assert config["model"] == config["small_model"] == route.opencode_model
    assert meta["route"]["key_names"] == ["OPENROUTER_API_KEY"]
    assert _files_containing(tmp_path, FAKE) == []
    with pytest.raises(ValueError, match="route is for"):
        pilot.run_one({**job, "model": "openai/gpt-6-luna"}, args, "brief", "t", route)


def test_chat_turn_writes_config_and_passes_the_key_by_name_only(tmp_path, monkeypatch):
    calls = []
    _fake_launcher(monkeypatch, chat, calls)
    monkeypatch.setenv("FIELDSHIFT_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("FIELDSHIFT_API_KEY", FAKE)
    agent = chat.OpenCodeAgent("custom/qwen3.5:9b", 30)
    agent.prepare(tmp_path / "run", "brief")
    result = agent.turn(tmp_path / "run", "hello", session_id=None, keep_session=False, farm_id="f1", port=9,
                        token="service-token")
    (call,) = calls
    assert FAKE not in " ".join(call["argv"]) and call["env"]["FIELDSHIFT_API_KEY"] == FAKE
    assert "FIELDSHIFT_API_KEY" in call["env"]["FIELDSHIFT_PASS_ENV"].split()
    assert call["argv"][call["argv"].index("--model") + 1] == "custom/qwen3.5-9b"
    config = json.loads((tmp_path / "run/cfg/opencode/opencode.json").read_text())
    assert config["providers"]["custom"]["settings"]["baseURL"] == "http://host.docker.internal:11434/v1"
    assert result["route"]["key_names"] == ["FIELDSHIFT_API_KEY"] and result["reply"] == "ok"
    assert result["credential_scrub"]["verified_clean"]
    assert _files_containing(tmp_path, FAKE) == []


def test_launcher_puts_names_not_values_on_the_docker_command_line(tmp_path, monkeypatch):
    for directory in ("cfg", "data", "work"):
        (tmp_path / directory).mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE)
    monkeypatch.setenv("FIELDSHIFT_PASS_ENV", "OPENROUTER_API_KEY")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(containers.subprocess, "run", run)
    assert containers.main([str(tmp_path), "run", "hi"]) == 0
    (argv,) = calls
    assert FAKE not in argv and argv[argv.index("-e") + 1] == "OPENROUTER_API_KEY"
    monkeypatch.setenv("FIELDSHIFT_PASS_ENV", "A=b")
    with pytest.raises(SystemExit) as caught:
        containers.main([str(tmp_path), "run"])
    assert caught.value.code == 2 and len(calls) == 1


# -- the app's model picker ----------------------------------------------------------------------

def test_app_typed_key_goes_to_the_provider_variable():
    access = core.model_environment("openrouter/openai/gpt-6-luna", FAKE, environ={})
    assert access["env"] == {"OPENROUTER_API_KEY": FAKE} and access["error"] is None
    assert access["route"]["key_names"] == ["OPENROUTER_API_KEY"] and access["from_page"] == ["OPENROUTER_API_KEY"]
    assert FAKE not in json.dumps({k: v for k, v in access.items() if k != "env"})


def test_app_key_typed_for_another_provider_is_not_reused():
    access = core.model_environment("anthropic/claude-sonnet-5-5", FAKE, key_model="openrouter/x", environ={})
    assert access["env"] == {} and "ANTHROPIC_API_KEY" in access["error"] and "Subscription" in access["error"]


def test_app_custom_endpoint_and_detected_names():
    access = core.model_environment("custom/qwen3.5:9b", "", "http://localhost:11434/v1",
                                    environ={"GROQ_API_KEY": FAKE})
    assert access["env"] == {"FIELDSHIFT_BASE_URL": "http://localhost:11434/v1"} and access["error"] is None
    assert access["detected"] == ["GROQ_API_KEY"] and access["key_variable"] == "FIELDSHIFT_API_KEY"
    go = core.model_environment("opencode-go/gpt-6-luna", environ={"OPENCODE_GO_API_KEY": FAKE})
    assert go["error"] is None and go["env"] == {} and go["route"]["key_names"] == ["OPENCODE_API_KEY"]


def test_app_spawn_hands_the_key_to_the_worker_environment_only(tmp_path, monkeypatch):
    seen = {}

    class Popen:
        def __init__(self, argv, **kwargs):
            seen.update(argv=argv, env=kwargs["env"])
            self.pid = 1
    monkeypatch.setattr(core.subprocess, "Popen", Popen)
    core.spawn(tmp_path, "solve", env={"OPENROUTER_API_KEY": FAKE})
    assert seen["env"]["OPENROUTER_API_KEY"] == FAKE and FAKE not in " ".join(seen["argv"])
    assert "OPENROUTER_API_KEY" in (tmp_path / "worker.log").read_text()
    assert _files_containing(tmp_path, FAKE) == []
