"""Authenticated local simulator HTTP API (stdlib, no extra dependencies)."""

import argparse
import hmac
import json
import logging
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from fieldshift.agent import farms, us_units
from fieldshift.agent.brief import build
from fieldshift.agent.runtime import DEFAULT_RUNTIME, InputError, clean
from fieldshift.sim.forecast import ForecastUnavailable

MAX_BODY = 64 * 1024
LOG_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


def studies_for(runtime, farm_id, payload):
    """Ranked study cards from the fixed library, with farm fit; read-only, never the web."""
    from fieldshift.agent import evidence, studies
    studies.validate(payload)
    profiles = runtime.__dict__.setdefault("study_profiles", {})
    with runtime.lock:
        farm = runtime.farm(farm_id)
        if farm_id not in profiles:
            profiles[farm_id] = studies.farm_profile(farm)
    root = getattr(farm, "evidence_root", None) or evidence.default_root()
    return {"farm_id": farm_id, **studies.search(root, payload, profiles[farm_id])}


def handler(token, runtime, log_path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FieldShiftAgent/1"

        def log_message(self, *_):
            # No bearer token, farm record or paths in ordinary HTTP stderr.
            pass

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def do_PUT(self):
            self.dispatch()

        def do_DELETE(self):
            self.dispatch()

        def do_OPTIONS(self):
            self.dispatch()

        def do_HEAD(self):
            self.dispatch()

        def do_PATCH(self):
            self.dispatch()

        def do_TRACE(self):
            self.dispatch()

        def do_CONNECT(self):
            self.dispatch()

        def dispatch(self):
            started = time.perf_counter()
            farm_id, rotations, status, ranking_request = None, None, 500, None
            try:
                auth = self.headers.get("Authorization", "")
                if not hmac.compare_digest(auth.encode(), ("Bearer " + token).encode()):
                    status = 401
                    self.send_json(status, {"error": "unauthorized", "message": "Supply Authorization: Bearer <service token>"})
                    return
                parts = urlsplit(self.path).path.strip("/").split("/")
                if len(parts) != 3 or parts[0] != "farm":
                    status = 404
                    self.send_json(status, {"error": "not_found", "message": "Use /farm/<farm_id>/options, /brief, /simulate or /scoreboard"})
                    return
                farm_id, route = parts[1:]
                if self.command == "GET" and route == "brief":
                    text = build(farm_id, runtime)
                    status = 200
                    self.send_body(status, text.encode(), "text/markdown; charset=utf-8")
                elif self.command == "GET" and route == "options":
                    with runtime.lock:
                        data = runtime.farm(farm_id).options()
                    status = 200
                    self.send_json(status, data)
                elif self.command == "GET" and route == "state":
                    from fieldshift.sim.state import load
                    with runtime.lock:
                        data, _ = load(runtime.farm(farm_id))
                    status = 200
                    self.send_json(status, {"farm_id": farm_id, **data})
                elif self.command == "POST" and route in ("simulate", "scoreboard", "economics", "forecast", "form",
                                                          "studies"):
                    if self.headers.get("Transfer-Encoding"):
                        raise InputError("Chunked requests unsupported; send Content-Length with a JSON body")
                    if self.headers.get_content_type() != "application/json":
                        raise InputError("Content-Type must be application/json")
                    try:
                        size = int(self.headers.get("Content-Length", "0"))
                    except ValueError as exc:
                        raise InputError("Content-Length must be an integer") from exc
                    if size <= 0 or size > MAX_BODY:
                        raise InputError(f"JSON body must contain 1–{MAX_BODY} bytes")
                    self.connection.settimeout(15)
                    raw = self.rfile.read(size)
                    try:
                        payload = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise InputError("Invalid JSON; NaN/Infinity are not allowed") from exc
                    rotations = payload.get("rotations") if isinstance(payload, dict) else None
                    # Chat sessions only: the host owns form versions and the farmer's confirmation.
                    chat = getattr(runtime, "chat_form", None)
                    if route == "form" and chat is None:
                        raise InputError("Form edits are only available in a chat session")
                    if route == "economics" and chat is not None:
                        raise InputError("In chat, propose prices and costs with ./farmsim form; the farmer confirms them")
                    if chat is not None and route in ("simulate", "scoreboard", "forecast"):
                        chat.gate(farm_id)
                    if route == "studies":
                        data = studies_for(runtime, farm_id, payload)
                    elif route == "form":
                        data = chat.propose(farm_id, payload)
                    elif route == "scoreboard":
                        ranking_request = payload
                        data = runtime.scoreboard(farm_id, payload)
                    elif route == "economics":
                        data = {"farm_id": farms.edit_economics(farm_id, payload, runtime.root),
                                 "parent_farm_id": farm_id, "method": "Immutable economics fork; physical caches reused unless capacity changes"}
                    elif route == "forecast":
                        from fieldshift.sim.forecast import build as forecast
                        with runtime.lock:
                            data = {"farm_id": farm_id, **forecast(runtime.farm(farm_id), payload)}
                    else:
                        data = runtime.simulate(farm_id, payload)
                    if route in ("simulate", "scoreboard", "forecast"):
                        # Farmer-facing US units are added here, outside the fingerprinted runtime/scoreboard.
                        data = us_units.with_note(data)
                    status = 200
                    self.send_json(status, data)
                else:
                    status = 405
                    self.send_json(status, {"error": "method_not_allowed", "message": "GET /options or /brief; POST /simulate or /scoreboard"})
            except farms.FarmError as exc:
                status = 404 if "farm_id" in str(exc) else 422
                self.send_json(status, {"error": "farm_error", "message": str(exc)})
            except ForecastUnavailable as exc:
                status = 503
                self.send_json(status, {"error": "forecast_unavailable", "message": str(exc)})
            except (InputError, ValueError) as exc:
                status = 422
                self.send_json(status, {"error": "invalid_request", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                status = 408
            except Exception:
                LOGGER.exception("Internal service failure")
                status = 500
                self.send_json(status, {"error": "simulation_unavailable",
                                        "message": "Server could not load data or run the accelerator. Ask the operator to inspect server logs; "
                                                   "MPS/CUDA and local data/model files are required. No CPU fallback."})
            finally:
                entry = {"time": datetime.now(UTC).isoformat(), "method": self.command,
                         "path": urlsplit(self.path).path,
                          "farm_id": farm_id, "rotations": rotations, "status": status,
                          "scoreboard_request": ranking_request,
                         "latency_seconds": time.perf_counter() - started}
                with LOG_LOCK:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a") as out:
                        out.write(json.dumps(clean(entry), allow_nan=False) + "\n")

        def send_json(self, status, data):
            self.send_body(status, json.dumps(clean(data), allow_nan=False).encode(), "application/json; charset=utf-8")

        def send_body(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def make_server(token, host="0.0.0.0", port=8601, runtime=None, log_path=None):
    if not isinstance(token, str) or not token.strip() or "\n" in token or "\r" in token:
        raise ValueError("A nonempty single-line bearer token is required")
    server = ThreadingHTTPServer((host, port), handler(token, runtime or DEFAULT_RUNTIME,
                                                    Path(log_path or "data/processed/engine_v26/service_log.jsonl")))
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8601)
    parser.add_argument("--host", "--bind", default="0.0.0.0", help="Use 0.0.0.0 for Docker host.docker.internal")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    try:
        server = make_server(args.token, args.host, args.port)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Field Shift agent service on {args.host}:{args.port}; bearer authentication required", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
