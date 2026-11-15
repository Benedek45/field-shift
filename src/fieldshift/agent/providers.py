"""Which provider serves an advisor model, which key it needs, and the OpenCode config for one run.

A model is written `<provider>/<model id>`; the provider is the part before the first slash, so
`openrouter/anthropic/claude-sonnet-5.5` is OpenRouter's model `anthropic/claude-sonnet-5.5`.

Routes:
- catalog providers (models.dev, vendored in `provider_catalog.json`): the key comes from the
  variable(s) OpenCode reads for that provider, e.g. OPENROUTER_API_KEY, ANTHROPIC_API_KEY;
- `custom/<model id>`: any OpenAI-compatible server at FIELDSHIFT_BASE_URL, key FIELDSHIFT_API_KEY
  (optional, local servers usually need none);
- `ollama/<model>` and `lmstudio/<model>`: OpenCode's local discovery, pointed at the host.

Keys are read from the environment first, then from the gitignored files `fieldshift.env`, `.env`
and `.agent-env` in the repository root. They reach the container only by variable name; values
are never written to the run's files or put on a command line.

Refresh the catalog with `uv run python -m fieldshift.agent.providers --update-catalog`.
"""

import argparse
import difflib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[3]
CATALOG_PATH = Path(__file__).with_name("provider_catalog.json")
CATALOG_URL = "https://models.dev/api.json"
KEY_FILES = ("fieldshift.env", ".env", ".agent-env")  # earlier files win; the environment wins over all
CUSTOM = "custom"
BASE_URL_VAR = "FIELDSHIFT_BASE_URL"
CUSTOM_KEY_VAR = "FIELDSHIFT_API_KEY"
COMPATIBLE_PACKAGE = "@opencode/ai/providers/openai-compatible"
CONTAINER_HOST = "host.docker.internal"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
# Our own name for OpenCode Go's key, accepted for the variable OpenCode reads.
ALIASES = {"OPENCODE_API_KEY": ("OPENCODE_GO_API_KEY",)}
# OpenCode probes these on 127.0.0.1, which inside the container is the container itself.
LOCAL_DISCOVERY = {"ollama": ("http://127.0.0.1:11434/v1", ()), "lmstudio": ("http://127.0.0.1:1234/v1", ("LMSTUDIO_API_KEY",))}
SUBSCRIPTION_NOTE = ("Subscription logins (Claude Pro/Max, ChatGPT Plus/Pro) are not supported by this repository; "
                     "use an API key, OpenRouter, or a custom endpoint.")
NOT_SUPPORTED = {
    "github-copilot": "GitHub Copilot needs an interactive device login, which the container cannot do.",
    "google-vertex": "Vertex needs a Google credentials file and project, which are not passed into the container.",
    "google-vertex-anthropic": "This provider ID is retired in OpenCode 2; Vertex is not supported here either.",
    "azure-cognitive-services": "This provider ID is retired in OpenCode 2; use azure/<deployment>.",
    "vllm": ("OpenCode's vLLM discovery starts with tools disabled and the advisor needs tools; use the custom "
             f"route: {BASE_URL_VAR}=http://localhost:8000/v1 and model custom/<served model name>."),
}


class ProviderError(ValueError):
    """The model string cannot be served as written."""


class UnknownProviderError(ProviderError):
    pass


class MissingKeyError(ProviderError):
    pass


class SubscriptionModelError(MissingKeyError):
    """anthropic/* or openai/* without an API key: only a subscription login could serve it."""


@dataclass(frozen=True)
class Route:
    model: str            # as requested
    provider: str
    opencode_model: str   # the model reference OpenCode is given
    key_names: tuple      # variables passed into the container, by name
    env: dict = field(repr=False)  # name -> value, only for the launcher's process environment
    providers_config: dict = field(default_factory=dict)
    notes: tuple = ()

    def secrets(self):
        """Key values to search for and scrub; never print them."""
        return [v for v in self.env.values() if v]

    def describe(self):
        """A summary with names only, safe for run.json and logs."""
        return {"model": self.model, "provider": self.provider, "opencode_model": self.opencode_model,
                "key_names": list(self.key_names), "notes": list(self.notes)}


def load_catalog(path=CATALOG_PATH):
    return json.loads(Path(path).read_text())


def key_environment(environ=None, root=REPO):
    """The environment plus keys from the local key files (the environment wins)."""
    merged = {}
    for name in reversed(KEY_FILES):
        path = Path(root) / name
        if path.is_file():
            merged.update({k: v for k, v in dotenv_values(path).items() if v})
    merged.update({k: v for k, v in (os.environ if environ is None else environ).items() if v})
    return merged


def known_key_names(catalog=None):
    catalog = catalog or load_catalog()
    names = {n for p in catalog["providers"].values() for n in p["env"]}
    names |= {a for aliases in ALIASES.values() for a in aliases}
    return names | {BASE_URL_VAR, CUSTOM_KEY_VAR}


def detected_key_names(environ, catalog=None):
    """Names (never values) of provider variables that are set."""
    return sorted(n for n in known_key_names(catalog) if environ.get(n))


def container_url(url):
    """(url, rewritten): a URL on this machine, made reachable from inside the container."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ProviderError(f"{BASE_URL_VAR} must be an http(s) URL such as http://localhost:11434/v1")
    if parts.username or parts.password:
        raise ProviderError(f"Put credentials in {CUSTOM_KEY_VAR}, not in {BASE_URL_VAR}.")
    if parts.hostname not in LOCAL_HOSTS:
        return url, False
    netloc = CONTAINER_HOST + (f":{parts.port}" if parts.port else "")
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc)), True


def _model_key(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-") or "model"


def _split(model):
    provider, _, model_id = str(model).strip().partition("/")
    if not provider or not model_id:
        raise ProviderError(f"Write the model as <provider>/<model id>, e.g. openrouter/openai/gpt-6-luna (got {model!r}).")
    return provider, model_id


def _custom(model, model_id, environ):
    base = environ.get(BASE_URL_VAR, "")
    if not base:
        raise MissingKeyError(f"{model} needs {BASE_URL_VAR}, the OpenAI-compatible endpoint (for example "
                              f"http://localhost:11434/v1 for Ollama); {CUSTOM_KEY_VAR} is optional.")
    url, rewritten = container_url(base)
    key = environ.get(CUSTOM_KEY_VAR, "")
    alias = _model_key(model_id)
    block = {"name": "Custom endpoint", "package": COMPATIBLE_PACKAGE, "settings": {"baseURL": url},
             "models": {alias: {"modelID": model_id, "name": model_id}}}
    if key:
        block["env"] = [CUSTOM_KEY_VAR]
    notes = [f"{BASE_URL_VAR} points at this machine; the container uses {url}."] if rewritten else []
    if not key:
        notes.append(f"{CUSTOM_KEY_VAR} is not set; requests carry no key.")
    return Route(model, CUSTOM, f"{CUSTOM}/{alias}", (CUSTOM_KEY_VAR,) if key else (),
                 {CUSTOM_KEY_VAR: key} if key else {}, {CUSTOM: block}, tuple(notes))


def _local(model, provider, model_id, environ):
    default, key_vars = LOCAL_DISCOVERY[provider]
    url, rewritten = container_url(environ.get(BASE_URL_VAR) or default)
    env = {n: environ[n] for n in key_vars if environ.get(n)}
    note = (f"{provider} on this machine is reached from the container at {url}.",) if rewritten else ()
    return Route(model, provider, model, tuple(env), env, {provider: {"settings": {"baseURL": url}}}, note)


def resolve(model, environ=None, catalog=None):
    """The Route for a model, or a ProviderError that names what is missing (never a value)."""
    environ = key_environment() if environ is None else environ
    provider, model_id = _split(model)
    if provider == CUSTOM:
        return _custom(model, model_id, environ)
    if provider in LOCAL_DISCOVERY:
        return _local(model, provider, model_id, environ)
    if provider in NOT_SUPPORTED:
        raise ProviderError(f"{model}: {NOT_SUPPORTED[provider]}")
    catalog = catalog or load_catalog()
    entry = catalog["providers"].get(provider)
    if entry is None:
        close = difflib.get_close_matches(provider, list(catalog["providers"]) + [CUSTOM], n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise UnknownProviderError(
            f"Unknown provider {provider!r} in {model}.{hint} Use a provider from the models.dev catalog "
            f"(e.g. openrouter, anthropic, openai, opencode-go) or custom/<model id> with {BASE_URL_VAR}.")
    env = {}
    for name in entry["env"]:
        value = environ.get(name) or next((environ[a] for a in ALIASES.get(name, ()) if environ.get(a)), "")
        if value:
            env[name] = value
    if not env:
        names = list(entry["env"]) + [a for n in entry["env"] for a in ALIASES.get(n, ())]
        wanted = names[0] if len(names) == 1 else "one of " + ", ".join(names)
        message = f"{model} needs an API key: set {wanted} (environment or fieldshift.env)."
        if provider in ("anthropic", "openai"):
            raise SubscriptionModelError(message + " " + SUBSCRIPTION_NOTE)
        raise MissingKeyError(message)
    return Route(model, provider, model, tuple(env), env)


def opencode_config(route):
    """The run's opencode.json: names only; small_model keeps title/summary calls on the same provider."""
    config = {"$schema": "https://opencode.ai/config.json", "model": route.opencode_model,
              "small_model": route.opencode_model}
    if route.providers_config:
        config["providers"] = route.providers_config
    return config


def write_config(run_dir, route):
    path = Path(run_dir) / "cfg" / "opencode" / "opencode.json"
    path.write_text(json.dumps(opencode_config(route), indent=2) + "\n")
    return path


def launch_env(route, base=None, extra=None):
    """Process environment for vm/oc.sh: the key under the variable OpenCode reads, plus the pass list."""
    names = list(route.key_names) + list(extra or {})
    env = dict(os.environ if base is None else base)
    env.update(route.env)
    env.update(extra or {})
    env["FIELDSHIFT_PASS_ENV"] = " ".join(dict.fromkeys(names))
    return env


# -- catalog refresh ----------------------------------------------------------------------------

def build_catalog(api):
    providers = {pid: {"name": p.get("name", pid), "env": list(p.get("env") or [])}
                 for pid, p in sorted(api.items()) if p.get("env")}
    return {"source": CATALOG_URL, "fetched": datetime.now(UTC).strftime("%Y-%m-%d"),
            "note": "Provider ID -> environment variables OpenCode reads for its key (models.dev `env`).",
            "providers": providers}


def update_catalog(path=CATALOG_PATH):
    # models.dev answers 403 to urllib's default User-Agent.
    request = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "fieldshift-provider-catalog"})
    with urllib.request.urlopen(request, timeout=60) as response:
        catalog = build_catalog(json.loads(response.read()))
    Path(path).write_text(json.dumps(catalog, indent=1) + "\n")
    return catalog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update-catalog", action="store_true", help=f"refresh {CATALOG_PATH.name} from models.dev")
    p.add_argument("--check", metavar="MODEL", help="show how a model would be served (variable names only)")
    args = p.parse_args(argv)
    if args.update_catalog:
        catalog = update_catalog()
        print(f"{len(catalog['providers'])} providers -> {CATALOG_PATH} ({catalog['fetched']})")
    if args.check:
        try:
            print(json.dumps(resolve(args.check).describe(), indent=2))
        except ProviderError as exc:
            sys.exit(str(exc))
    if not (args.update_catalog or args.check):
        print("Detected key variables:", ", ".join(detected_key_names(key_environment())) or "none")


if __name__ == "__main__":
    main()
