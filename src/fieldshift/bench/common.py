"""Shared local-only benchmark resources and canonical cycles."""

import contextlib
import secrets
import threading
from pathlib import Path

from fieldshift.agent.a1_analysis import winter_key
from fieldshift.agent.runtime import Runtime
from fieldshift.agent.service import make_server

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/processed/bench"
RUNS = Path.home() / ".opencode-vm/fieldshift/runs/bench"


def canonical(rotation):
    """Keep double crops, reduce exact repeats, then remove cycle phase."""
    if not isinstance(rotation, list) or not rotation:
        raise ValueError("Rotation must be a nonempty list")
    years = [(y["crop"], winter_key(y.get("winter")) or "", y.get("double_crop") or "") for y in rotation]
    n = len(years)
    period = next(p for p in range(1, n + 1) if n % p == 0 and years == years[:p] * (n // p))
    years = years[:period]
    return min(tuple(years[i:] + years[:i]) for i in range(period))


def choose(board):
    if not board["rows"]:
        raise ValueError("No feasible rotations")
    existing = board["existing"]
    return existing["rotation"] if existing["tied_with_best"] else board["rows"][0]["rotation"]


def unique(board):
    """Require a single canonical winner, beyond every returned tie tolerance."""
    if not board["rows"]:
        return False
    if board.get("all_pool_unique") is False:
        return False
    key = canonical(board["rows"][0]["rotation"])
    if board["existing"]["tied_with_best"] and canonical(board["existing"]["rotation"]) != key:
        return False
    return all(not row["tied_with_best"] or canonical(row["rotation"]) == key for row in board["rows"])


@contextlib.contextmanager
def local_service(port=8611):
    """Own token, isolated registry/cache/log, never touch service 8601."""
    DATA.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(root=DATA / "service")
    token = secrets.token_urlsafe(32)
    server = make_server(token, "0.0.0.0", port, runtime, DATA / "service_log.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield runtime, token
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
