"""Chat form versions with a host-side confirmation gate.

The model may only propose a form patch (POST /farm/<id>/form). Nothing is applied until the
host reads a plain confirmation in the farmer's own next message (`ChatForms.confirm`). Ranking
routes (`ChatForms.gate`) accept only the latest confirmed farm version. Every applied version is
a new immutable farm record; versions and proposals are kept on disk for audit and resume.
"""

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path

from fieldshift.agent import brief, farms, form
from fieldshift.planner.data import county_options

YES = re.compile(r"\b(yes|yeah|yep|yup|correct|confirm(?:ed)?|right|looks (?:good|right|fine)|sounds good|go ahead|"
                 r"ok|okay|sure|agreed|exactly|perfect|use (?:them|those|these))\b", re.IGNORECASE)
# Anything that may carry new information or a reservation makes the message an answer, not a "yes".
NOT_PLAIN = re.compile(r"\b(no|nope|not|never|but|actually|except|change|instead|wrong|however|although|though|wait|"
                       r"rather|only|unless|other|different|more|less|add|remove|drop|switch|mostly|almost|partly|"
                       r"if|maybe|perhaps|irrigat\w*|rotation|corn|soy\w*|wheat|price|cost)\b|n't\b|\d|\?", re.IGNORECASE)
WEIGHT_WORDS = {"income": "money", "water": "water use", "soil": "soil building", "nitrogen": "less nitrate loss",
                "erosion": "less erosion", "cover": "ground cover", "forage": "forage", "resilience": "bad-year protection",
                "market": "local market"}
PRESET_WORDS = {"profit": "profit", "steady": "steady income", "soil_builder": "soil builder", "water_saver": "water saver"}


def is_confirmation(message):
    """A short message that says yes and carries no correction, number, question or new fact."""
    text = message.strip()
    return bool(text) and len(text) <= 160 and bool(YES.search(text)) and not NOT_PLAIN.search(text)


def merge(base, patch):
    """Nested objects merge; lists and scalars replace; null removes the key."""
    if not isinstance(patch, dict):
        raise farms.FarmError("A form patch must be a JSON object")
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict):
            out[key] = merge(out.get(key, {}) if isinstance(out.get(key), dict) else {}, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def changes(old, new, prefix=""):
    """Dotted paths whose value differs between two raw forms."""
    out = []
    for key in sorted(set(old) | set(new)):
        a, b, path = old.get(key), new.get(key), prefix + key
        if isinstance(a, dict) and isinstance(b, dict):
            out += changes(a, b, path + ".")
        elif a != b:
            out.append(path)
    return out


def digest(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.chmod(path, 0o600)


def rotation_text(rotation):
    parts = []
    for s in rotation:
        text = s["crop"].replace("_", " ")
        if s.get("winter"):
            text += " + " + s["winter"].replace("_", " ") + " over winter"
        if s.get("double_crop"):
            text += " + double-crop " + s["double_crop"]
        parts.append(text)
    return " → ".join(parts)


def _value(v):
    if isinstance(v, dict) and set(v) == {"value", "unit"}:
        return f"{v['value']} {v['unit']}"
    if isinstance(v, dict):
        return "; ".join(f"{k.replace('_', ' ')} {_value(x)}" for k, x in v.items())
    if isinstance(v, list):
        return ", ".join(str(x).replace("_", " ") for x in v)
    if v is True:
        return "yes"
    return str(v)


class ChatForms:
    """Form versions of one chat. `runtime.chat_form` points here while a chat service runs."""

    def __init__(self, runtime, directory):
        self.runtime, self.dir = runtime, Path(directory)
        self.versions, self.pending, self.confirmed_farm_id = [], None, None

    # -- state -----------------------------------------------------------------------------
    @property
    def current(self):
        return self.versions[-1]

    @property
    def confirmed(self):
        return next((v for v in reversed(self.versions) if v["farm_id"] == self.confirmed_farm_id), None)

    def save(self):
        write(self.dir / "state.json", {"versions": self.versions, "pending": self.pending,
                                        "confirmed_farm_id": self.confirmed_farm_id})

    @classmethod
    def load(cls, runtime, directory):
        self = cls(runtime, directory)
        state = json.loads((self.dir / "state.json").read_text())
        self.versions, self.pending, self.confirmed_farm_id = (state["versions"], state["pending"],
                                                               state["confirmed_farm_id"])
        return self

    def _register(self, raw):
        normal = form.validate(raw)
        fid = farms.add(normal["farm"], self.runtime.root)
        self.runtime.farm(fid)  # reject an infeasible stated cycle before anyone sees the proposal
        return fid, normal

    def seed(self, raw):
        """Version 0: the host's draft (county only plus placeholders), never confirmed."""
        with self.runtime.lock:
            fid, normal = self._register(raw)
            self.versions = [{"version": 0, "farm_id": fid, "parent_farm_id": None, "input": raw,
                              "normalised": normal, "confirmed": False, "confirmation": None,
                              "time": time.time()}]
            write(self.dir / "v0-draft.json", self.versions[0])
            self.save()
            return fid

    # -- model side (service thread) -------------------------------------------------------
    def propose(self, farm_id, patch):
        with self.runtime.lock:
            if farm_id != self.current["farm_id"]:
                raise farms.FarmError("This is not the current farm version; use the farm id in the chat state")
            raw = merge(self.current["input"], patch)
            fid, normal = self._register(raw)
            pid = digest(raw)[:12]
            self.pending = {"id": pid, "farm_id": fid, "base_farm_id": farm_id, "input": raw, "normalised": normal,
                            "changes": changes(self.current["input"], raw), "patch": patch, "time": time.time()}
            write(self.dir / f"proposal-{pid}.json", self.pending)
            self.save()
            return {"confirmation_required": True, "proposal": pid, "changed_fields": self.pending["changes"],
                    "proposed_form": raw, "weights": normal["weights"], "limits": normal["limits"],
                    "weather_view": normal["weather_view"],
                    "message": "Nothing is applied yet. The host shows the farmer a table of these values under "
                               "your reply and applies them only if the farmer's next message confirms. Do not "
                               "draw your own table. Ranking uses the last confirmed version until then."}

    def gate(self, farm_id):
        if farm_id != self.confirmed_farm_id:
            raise farms.FarmError(
                "Ranking and simulation need a form the farmer has confirmed. Propose values with "
                "./farmsim form; the host shows them and applies them after the farmer says yes.")

    # -- host side -------------------------------------------------------------------------
    def confirm(self, message):
        """Apply the pending proposal if `message` (the farmer's own words) plainly confirms it."""
        with self.runtime.lock:
            if not self.pending or not is_confirmation(message):
                return None
            p, before = self.pending, self.current
            version = {"version": len(self.versions), "farm_id": p["farm_id"], "parent_farm_id": p["base_farm_id"],
                       "input": p["input"], "normalised": p["normalised"], "confirmed": True,
                       "confirmation": {"proposal": p["id"], "farmer_message": message, "changes": p["changes"]},
                       "time": time.time()}
            old = brief.build(before["farm_id"], self.runtime, include_outlook=False)
            new = brief.build(p["farm_id"], self.runtime, include_outlook=False)
            old_sections = {s.split("\n", 1)[0]: s for s in old.split("\n## ")}
            changed = ["## " + s for s in new.split("\n## ")[1:] if old_sections.get(s.split("\n", 1)[0]) != s]
            self.versions.append(version)
            self.pending, self.confirmed_farm_id = None, p["farm_id"]
            write(self.dir / f"v{version['version']}.json", version)
            self.save()
            return {**version, "changed_brief_sections": changed, "brief": new}

    def table(self, proposal=None):
        """Farmer-unit table of a proposal (default: the pending one), changed rows marked."""
        p = proposal or self.pending
        raw, normal = p["input"], p["normalised"]
        changed = p.get("changes", [])
        location = raw["location"]
        farm = self.runtime.farm(p["farm_id"])
        name = location.get("county") or county_options().get(location.get("adm_id"), "").split(" · ")[0]
        irrigated = "yes" if location["irrigated"] else "no (rainfed)"
        if location.get("well_capacity") is not None:
            irrigated += ", well capacity " + _value(location["well_capacity"])
        if "current_rotation" in raw:
            rotation = rotation_text(normal["farm"]["existing_rotation"]) + " (your words)"
        else:
            rotation = (rotation_text([{"crop": c} for c in farm.site["existing"]])
                        + " (inferred from satellite crop maps of the county, not your field)")
        preset = raw.get("priorities", {}).get("preset", "steady")
        weights = ", ".join(f"{WEIGHT_WORDS[k]} {v:g}" for k, v in normal["weights"].items() if v)
        limits = "; ".join(f"{k.replace('_', ' ')}: {_value(v)}" for k, v in raw.get("limits", {}).items()) or "none"
        econ = _value(raw["economics"]) if raw.get("economics") else "regional budget prices and costs (none of your own)"
        view = ("past weather 1981-2025 plus this season's outlook" if normal["weather_view"] == "outlook"
                else "past weather 1981-2025")
        rows = [(("location.county", "location.adm_id"), "County", name),
                (("location.irrigated", "location.well_capacity"), "Irrigated", irrigated),
                ("current_rotation", "Current rotation", rotation),
                ("priorities", "Priorities", f"{PRESET_WORDS.get(preset, preset)} preset → {weights} (0-5 scale)"),
                ("limits", "Hard limits", limits), ("economics", "Your prices and costs", econ),
                ("weather_view", "Weather used", view)]
        if raw.get("field_facts"):
            rows.append(("field_facts", "Field facts", _value(raw["field_facts"])))
        if raw.get("free_text"):
            rows.append(("free_text", "Your notes", raw["free_text"].replace("|", "/").replace("\n", " ")))
        lines = ["| Item | Value |", "|---|---|"]
        for keys, label, value in rows:
            mark = any(c == k or c.startswith(k + ".") for c in changed for k in ([keys] if isinstance(keys, str) else keys))
            lines.append(f"| {label} | {value}{' **(changed)**' if mark else ''} |")
        return "\n".join(lines)
