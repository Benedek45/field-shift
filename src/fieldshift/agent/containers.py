"""Portable Docker argv construction; never invoke a host shell or expose key values."""

import argparse
import csv
import io
import os
import re
import subprocess
from pathlib import Path

DEFAULT_IMAGE = "fieldshift-agent:oc2.3.0"
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)


def docker_command(*args, environ=None):
    """Use the current Docker endpoint unless an explicit per-command context is set."""
    env = os.environ if environ is None else environ
    context = env.get("FIELDSHIFT_DOCKER_CONTEXT")
    return ["docker", *(["--context", context] if context else []), *args]


def image_name(environ=None):
    env = os.environ if environ is None else environ
    image = env.get("FIELDSHIFT_IMAGE", DEFAULT_IMAGE)
    if not image or image.startswith("-") or any(c.isspace() for c in image):
        raise ValueError("FIELDSHIFT_IMAGE must be a Docker image reference")
    return image


def _mount(path, target):
    # Docker --mount uses CSV, not shell quoting. This also handles commas in paths.
    output = io.StringIO()
    csv.writer(output, lineterminator="").writerow(["type=bind", f"src={path}", f"dst={target}"])
    return output.getvalue()


def run_command(run_dir, args, environ=None):
    """Build a shell-free run command mounting only this episode's cfg/data/work.

    Pass the same environment to subprocess.run (and timeout cleanup). Secrets are
    inherited by Docker using ``-e NAME``; values never enter this argv.
    """
    env = os.environ if environ is None else environ
    names = env.get("FIELDSHIFT_PASS_ENV", "").split()
    if any(ENV_NAME.fullmatch(name) is None for name in names):
        raise ValueError("FIELDSHIFT_PASS_ENV must list variable names only")
    if any(env.get(flag) == "1" for flag in ("FIELDSHIFT_ANTHROPIC", "FIELDSHIFT_OPENAI")):
        raise ValueError("Subscription-login adapters are not part of this repository; use an API key")
    root = Path(run_dir).expanduser().resolve()
    options = ["run", "--rm", "-i"]
    name = env.get("FIELDSHIFT_CONTAINER_NAME")
    if name:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError("FIELDSHIFT_CONTAINER_NAME must be a Docker container name")
        options += ["--name", name]
    # Linux Engine does not create Docker Desktop's host alias automatically.
    options += ["--add-host", "host.docker.internal:host-gateway"]
    if hasattr(os, "getuid"):
        options += ["--user", f"{os.getuid()}:{os.getgid()}"]
    # A host UID need not own the image's /home/advisor. Keep HOME ephemeral.
    options += ["--env", "HOME=/tmp/fieldshift-home"]
    for directory in ("cfg", "data", "work"):
        source = root / directory
        if not source.is_dir():
            raise ValueError(f"Run directory must contain {directory}/")
        options += ["--mount", _mount(source, f"/{directory}")]
    for variable in dict.fromkeys(names):
        options += ["-e", variable]
    return docker_command(*options, "--entrypoint", "opencode", image_name(env), *args, environ=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    try:
        command = run_command(parsed.run_dir, parsed.args)
        return subprocess.run(command, shell=False, check=False).returncode
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
