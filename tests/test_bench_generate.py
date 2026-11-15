import json

import numpy as np

from fieldshift.bench import generate


def test_certificate_checks_full_pool_with_normalised_float_weather(tmp_path, monkeypatch):
    record = {"overrides": {}, "economics": {}}
    monkeypatch.setattr(generate.farms, "get", lambda *_: record)
    monkeypatch.setattr(generate.farms, "physical_record", lambda _: {"county": "test"})
    monkeypatch.setattr(generate, "_fingerprint", lambda: "fixed")
    key = generate.scoring._hash(
        [generate.scoring.VERSION, generate.search.VERSION, "fixed", {"county": "test"}, 0.0, 0.0])
    root = tmp_path / "scoreboard"
    root.mkdir()
    (root / (key + ".json")).write_text(json.dumps({"rows": [{}] * 41}))
    np.savez(root / (key + ".npz"), runs=np.zeros((41, 1, 1)))
    rotations = [[{"crop": c}] for c in ("corn", "soybean")]

    def rank(data, runs, weights, limits, top, edits):
        assert top == 41
        return {"existing": {"tied_with_best": False, "rotation": rotations[0]},
                "rows": [{"rotation": rotations[0], "tied_with_best": True, "score": 100, "tie_tolerance": 0},
                         {"rotation": rotations[1], "tied_with_best": True, "score": 90, "tie_tolerance": 11}]}

    monkeypatch.setattr(generate.scoring, "ranking", rank)
    runtime = type("Runtime", (), {"root": tmp_path})()
    result = {"farm_id": "opaque", "weights_normalised": {"income": 1}, "limits_applied": {}}
    assert not generate.certify(runtime, result)
    assert result["all_pool_unique"] is False
    assert result["uniqueness_certificate"]["distinct_tied_cycles"] == 2
    assert result["uniqueness_certificate"]["smallest_lead_beyond_tolerance"] == -1
