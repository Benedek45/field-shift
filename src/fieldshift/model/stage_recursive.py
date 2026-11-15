"""Freeze private Kaggle inputs outside the repository; this does not push/run jobs."""

import argparse
import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

from fieldshift.model.recursive_data import ROOT
from fieldshift.model.recursive_train import BASELINE, FEATURES


def stage(directory, owner, dataset_slug="fieldshift-recursive-r3-20261114"):
    directory = directory.expanduser().resolve()
    project = Path.cwd().resolve()
    if directory == project or project in directory.parents:
        raise ValueError("Kaggle working directory must be outside the project")
    directory.mkdir(parents=True, exist_ok=False)
    files = [
        Path("data/processed/weather/power_county_daily.f32"),
        Path("data/processed/weather/power_county_daily.json"),
        FEATURES, ROOT / "cybench_canopy.f32", ROOT / "cybench_canopy.json",
        ROOT / "fallback_static.parquet", ROOT / "planting_dates.parquet",
        ROOT / "county_states.parquet",
    ]
    for name in ("data/processed/eo/eo_county_8day.f32", "data/processed/eo/eo_county_8day.json",
                  "data/processed/static/county_static.parquet", "data/processed/eo/ndvi_adjustment.parquet"):
        if Path(name).exists():
            files.append(Path(name))
    manifest = {"inputs": [], "source": "source.zip",
                "privacy": "Private research only; source licenses retained, not relicensed.",
                "calibration": "Outer/inner year design lives in EXPERIMENTS.md."}
    for number, path in enumerate(files):
        destination = f"input_{number:02d}_{path.name}"
        shutil.copyfile(path, directory / destination)
        digest = hashlib.sha256()
        with (directory / destination).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest["inputs"].append({"file": destination, "path": str(path),
                                    "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    for name in ("weather/power_county_daily", "eo/eo_county_8day"):
        original = Path("data/processed") / name
        selected = {Path(entry["path"]).suffix: entry for entry in manifest["inputs"]
                    if Path(entry["path"]).with_suffix("") == original}
        if not selected:
            continue
        meta = json.loads((directory / selected[".json"]["file"]).read_text())
        expected = 4
        for dimension in meta["shape"]:
            expected *= dimension
        if (directory / selected[".f32"]["file"]).stat().st_size != expected:
            raise RuntimeError(f"Input changed while staging: {original}; preserve snapshot for diagnosis")
    with zipfile.ZipFile(directory / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path("src").rglob("*.py")):
            archive.write(path, path)
        for path in (Path("EXPERIMENTS.md"), Path("pyproject.toml")):
            archive.write(path, path)
    manifest["source_sha256"] = hashlib.sha256((directory / "source.zip").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "dataset-metadata.json").write_text(json.dumps({
        "title": "FieldShift recursive private inputs 1981-2025",
        "id": f"{owner}/{dataset_slug}",
        "licenses": [{"name": "other"}], "is_private": True, "enable_gpu": True,
    }, indent=2))
    return manifest


def kernel(directory, owner, dataset_owner, folds, suffix, dataset_slug="fieldshift-recursive-r3-20261114"):
    """Generate a CUDA/T4-only runner; account/config authentication remains external."""
    directory.mkdir(parents=True, exist_ok=True)
    patch = io.BytesIO()
    with zipfile.ZipFile(patch, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("recursive.py", "recursive_data.py", "recursive_train.py", "expanded.py"):
            path = Path("src/fieldshift/model") / name
            archive.write(path, path)
        archive.write("EXPERIMENTS.md", "EXPERIMENTS.md")
        design = json.loads((BASELINE / "metrics.json").read_text())["design"]["folds"]
        archive.writestr("data/processed/recursive/r2_fold_design.json", json.dumps(design))
    encoded = base64.b64encode(patch.getvalue()).decode("ascii")
    patch_sha = hashlib.sha256(patch.getvalue()).hexdigest()
    runner = '''import base64, hashlib, io, json, os, pathlib, shutil, subprocess, sys, zipfile
import torch
if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
    raise RuntimeError("CUDA probe failed: a Tesla T4 is mandatory; P100/CPU is invalid")
print("CUDA_PROBE", torch.__version__, torch.cuda.get_device_name(0), flush=True)
manifests = list(pathlib.Path("/kaggle/input").rglob("manifest.json"))
matches = [p for p in manifests if "inputs" in json.loads(p.read_text())]
if len(matches) != 1:
    raise RuntimeError(f"Expected one private data manifest; got {matches}")
mount = matches[0].parent
manifest = json.loads(matches[0].read_text())
work = pathlib.Path("/kaggle/temp/fieldshift")
work.mkdir(parents=True, exist_ok=True)
source = mount / manifest["source"]
if source.exists():
    with zipfile.ZipFile(source) as archive:
        archive.extractall(work)
else:
    candidates = [p.parent for p in mount.rglob("fieldshift") if p.is_dir() and p.parent.name == "src"]
    if len(candidates) != 1:
        raise RuntimeError("Missing frozen Python source")
    shutil.copytree(candidates[0], work / "src")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(__PATCH_BASE64_LITERAL__))) as archive:
    archive.extractall(work)
for entry in manifest["inputs"]:
    destination = work / entry["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(mount / entry["file"])
os.chdir(work)
environment = dict(os.environ)
environment["PYTHONPATH"] = str(work / "src")
environment["FIELDSHIFT_SOURCE_OVERLAY_SHA256"] = __PATCH_SHA_LITERAL__
environment["FIELDSHIFT_DATA_MANIFEST_SHA256"] = hashlib.sha256(matches[0].read_bytes()).hexdigest()
environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
print("FROZEN_MANIFEST", manifest, flush=True)
for fold in FOLDS:
    processes = []
    if fold == "final":
        subprocess.run([sys.executable, "-m", "fieldshift.model.recursive_train", "--final",
                        "--require-t4", "--output", "/kaggle/working/r3"], env=environment, check=True)
        continue
    for held in (False, True):
        command = [sys.executable, "-m", "fieldshift.model.recursive_train",
                   "--folds", str(fold), "--require-t4",
                   "--output", "/kaggle/working/r3"]
        if held:
            command.append("--held")
        child_environment = dict(environment)
        child_environment["CUDA_VISIBLE_DEVICES"] = str(int(held) if torch.cuda.device_count() > 1 else 0)
        if torch.cuda.device_count() > 1:
            processes.append(subprocess.Popen(command, env=child_environment))
        else:
            subprocess.run(command, env=child_environment, check=True)
    statuses = [process.wait() for process in processes]
    if any(statuses):
        raise RuntimeError(f"Fold {fold} child failures: {statuses}; other work was allowed to finish")
'''
    runner = runner.replace("FOLDS", repr(folds))
    runner = runner.replace("__PATCH_BASE64_LITERAL__", repr(encoded)).replace("__PATCH_SHA_LITERAL__", repr(patch_sha))
    (directory / "run.py").write_text(runner)
    experiment = "r3b" if "r3b" in dataset_slug else "r3"
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/fieldshift-recursive-{experiment}-{suffix}",
        "title": f"FieldShift recursive {experiment.upper()} {suffix}",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{dataset_owner}/{dataset_slug}"],
        "competition_sources": [], "kernel_sources": [],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--kernel", action="store_true", help="write a runner instead of staging data")
    parser.add_argument("--dataset-owner")
    parser.add_argument("--folds", default="0", help="comma list; 'final' = deployment fit")
    parser.add_argument("--suffix")
    parser.add_argument("--dataset-slug", default="fieldshift-recursive-r3-20261114")
    args = parser.parse_args()
    if args.kernel:
        folds = [f if f == "final" else int(f) for f in args.folds.split(",")]
        kernel(args.directory, args.owner, args.dataset_owner or args.owner, folds,
               args.suffix or "fold" + "".join(map(str, folds)), args.dataset_slug)
    else:
        print(json.dumps(stage(args.directory, args.owner, args.dataset_slug), indent=2))
