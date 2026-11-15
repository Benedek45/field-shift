"""Train/evaluate one preregistered recursive configuration on resident tensors."""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from fieldshift.model.expanded import crop_groups, fold_masks, summary, target_for_fold
from fieldshift.model.neural import DeviceAdamW
from fieldshift.model.probabilistic import SEEDS, synchronize
from fieldshift.model.recursive import RecursiveField, save_recursive
from fieldshift.model.recursive_data import ROOT, SLOTS, FieldData

FEATURES = Path("data/processed/yield_expanded/power_8e14f4f0968b/features.parquet")
BASELINE = FEATURES.parent / "evaluation_5fold_40epoch_seq0_1981_2025"
OUTPUT = ROOT / "r3"


class FieldRows:
    def __init__(self, data, field, target):
        self.frame, self.field = data, field
        device = field.device
        self.county = torch.tensor(pd.Index(field.meta["counties"]).get_indexer(data.adm_id),
                                   dtype=torch.long, device=device)
        self.crop = torch.tensor(pd.Index(field.crops).get_indexer(data.crop),
                                 dtype=torch.long, device=device)
        self.year = torch.tensor(data.year.to_numpy() - 1981, device=device)
        self.target = torch.tensor(target, dtype=torch.float32, device=device)
        self.share = torch.tensor(data.irrigated_share.to_numpy(dtype=np.float32), device=device)
        size = data.groupby(["adm_id", "crop", "year"]).year.transform("size")
        self.weight = torch.tensor((1 / size).to_numpy(dtype=np.float32), device=device)
        self.primary = torch.tensor(data.primary.to_numpy(), device=device)

    def indices(self, mask):
        return torch.tensor(np.flatnonzero(mask), dtype=torch.long, device=self.field.device)

    def static(self, county, crop, share):
        field = self.field
        return torch.cat([field.static[county], field.traits[crop], share[:, None]], -1)

    def batch(self, indices, climatology, years=2, crop_masks=True, fit_years=None):
        county, crop, year = self.county[indices], self.crop[indices], self.year[indices]
        return self.make_batch(county, crop, year, self.share[indices], climatology, years,
                               crop_masks, fit_years) | {
            "target": self.target[indices], "weight": self.weight[indices],
        }

    def make_batch(self, county, crop, year, share, climatology, years, crop_masks, fit_years):
        field = self.field
        drivers, season, position = field.drivers(county, crop, year, years)
        positions = torch.cat([position, position[:, -1:] + 1], 1)
        truth = field.targets(county, crop, positions, crop_masks)
        auxiliary = field.auxiliary[county[:, None], positions]
        initial = torch.where(torch.isfinite(truth[:, 0]), truth[:, 0],
                              climatology[county, positions[:, 0] % 46])
        if fit_years is not None:
            # No next-Jan-1 test-year observation enters a fit-year state loss.
            permitted = torch.isin(positions // 46, fit_years)
            truth = torch.where(permitted[..., None], truth, torch.nan)
            auxiliary = torch.where(permitted[..., None], auxiliary, torch.nan)
        return {"initial": initial, "drivers": drivers, "season": season,
                "observations": truth, "static": self.static(county, crop, share),
                "capacity": field.capacities[county], "share": share,
                "auxiliary": auxiliary, "porosity": field.porosity[county],
                "county": county, "crop": crop, "year": year, "positions": positions}


def run_batch(model, batch, teacher_probability=0.):
    rollout = model.rollout(batch["initial"], batch["drivers"], batch["static"],
                            batch["capacity"], batch["share"], batch["observations"],
                            teacher_probability=teacher_probability)
    return rollout, model.yield_distribution(rollout, batch["drivers"], batch["season"])


def losses(model, batch, rollout, distribution, warmup=False):
    state = rollout["states"][:, :, 1:]
    truth = batch["observations"][:, 1:]
    observed = torch.isfinite(truth)
    clean = torch.nan_to_num(truth)
    weight = observed.float()
    water_weight = 1 - torch.nan_to_num(batch["share"], nan=0.)
    weight[..., :3] *= water_weight[:, None, None]
    squared = ((state - clean[None]) / model.state_scale).square()
    # Independent sensors retain their units; do not equate m3/m3 with MERRA
    # wetness. This operational theta = porosity * saturation operator is a proxy.
    auxiliary = batch["auxiliary"][:, 1:]
    auxiliary_mask = torch.isfinite(auxiliary)
    wet_mean = (rollout["states"][:, :, :-1, :3] + state[..., :3]) / 2
    theta = wet_mean[..., [0, 1, 0]] * batch["porosity"][None, :, None, None]
    auxiliary_error = ((theta - torch.nan_to_num(auxiliary)[None]) / .06).square()
    numerator = (squared * weight[None]).sum() + .25 * (auxiliary_error * auxiliary_mask[None]).sum()
    denominator = len(SEEDS) * (weight.sum() + .25 * auxiliary_mask.float().sum()).clamp_min(1)
    state_loss = numerator / denominator
    mean, _std, members, variance = distribution
    valid = torch.isfinite(batch["target"]) & batch["season"].sum(1).gt(0)
    target = torch.nan_to_num(batch["target"])
    w = batch["weight"] * valid
    error = (members - target[None]).square()
    likelihood = error if warmup else .5 * (variance.log() + error / variance)
    yield_loss = (likelihood * w[None]).sum() / (len(SEEDS) * w.sum().clamp_min(1))
    total = state_loss + .5 * yield_loss
    return total, torch.stack([state_loss.detach(), yield_loss.detach(),
                               ((mean - target).square() * w).sum() / w.sum().clamp_min(1)])


def train(rows, fit_mask, fit_years, epochs=24, batches=128, batch_size=96, held=False):
    field, device = rows.field, rows.field.device
    fit = rows.indices(fit_mask)
    adjacent = fit_mask & rows.frame.year.sub(1).isin(fit_years).to_numpy()
    two_year = rows.indices(adjacent)
    if not len(two_year):
        raise ValueError("No adjacent fit-year windows for continuous two-year training")
    allowed_years = torch.tensor(np.asarray(fit_years) - 1981, device=device)
    adjacent_years = torch.tensor([y - 1981 for y in fit_years if y - 1 in fit_years], device=device)
    allowed_crops = torch.unique(rows.crop[fit])
    climatology = field.climatology(fit_years, crop_masks=not held)
    model = RecursiveField(field.static.shape[1] + field.traits.shape[1] + 1)
    model.fit_scaler(rows.static(rows.county[fit], rows.crop[fit], rows.share[fit]))
    if device.type == "cuda":
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01,
                                      fused=True, capturable=True)
        scaler = torch.amp.GradScaler("cuda")
    else:
        optimizer = DeviceAdamW(model.parameters(), lr=.0003, weight_decay=.01)
        scaler = None
    history, epoch_times = [], []
    labelled = batch_size * 3 // 4
    unlabelled = batch_size - labelled
    torch.manual_seed(20261114)
    for epoch in range(epochs):
        synchronize()
        started = time.monotonic()
        model.train()
        probability = 1. if epoch < 4 else max(0., (7 - epoch) / 4)
        epoch_loss = torch.zeros(3, device=device)
        for iteration in range(batches):
            years = 1 + ((epoch * batches + iteration) % 2)
            candidates = fit if years == 1 else two_year
            selection = candidates[torch.randint(len(candidates), (labelled,), device=device)]
            batch = rows.batch(selection, climatology, years, not held, allowed_years)
            year_pool = allowed_years if years == 1 else adjacent_years
            county = torch.randint(field.n_counties, (unlabelled,), device=device)
            crop = allowed_crops[torch.randint(len(allowed_crops), (unlabelled,), device=device)]
            year = year_pool[torch.randint(len(year_pool), (unlabelled,), device=device)]
            share = torch.full((unlabelled,), torch.nan, device=device)
            extra = rows.make_batch(county, crop, year, share, climatology, years, not held, allowed_years)
            extra["target"] = torch.full((unlabelled,), torch.nan, device=device)
            extra["weight"] = torch.zeros(unlabelled, device=device)
            batch = {name: torch.cat([value, extra[name]]) for name, value in batch.items()}
            if device.type == "cuda":
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad()
            context = (torch.autocast("cuda", dtype=torch.float16)
                       if device.type == "cuda" else contextlib.nullcontext())
            with context:
                rollout, output = run_batch(model, batch, probability)
                loss, detail = losses(model, batch, rollout, output, warmup=epoch < 4)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            epoch_loss += detail
        synchronize()
        elapsed = time.monotonic() - started
        values = (epoch_loss / batches).tolist()
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Recursive training diverged at epoch {epoch}: {values}")
        history.append(values)
        epoch_times.append(elapsed)
        print(json.dumps({"epoch": epoch + 1, "teacher_probability": probability,
                          "state_loss": values[0], "yield_loss": values[1],
                          "yield_mse": values[2], "seconds": elapsed}), flush=True)
    # Fused AdamW step/moments must also be accelerator resident.
    for state in optimizer.state.values() if scaler is not None else []:
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.device != model.center.device:
                raise RuntimeError("Optimizer tensor escaped the accelerator")
    return model.eval(), climatology, {
        "epoch_losses": history, "epoch_seconds": epoch_times,
        "parameters": sum(p.numel() for p in model.parameters()),
        "members": len(SEEDS), "device": str(device), "epochs": epochs, "batches": batches,
        "source_overlay_sha256": os.environ.get("FIELDSHIFT_SOURCE_OVERLAY_SHA256"),
        "data_manifest_sha256": os.environ.get("FIELDSHIFT_DATA_MANIFEST_SHA256"),
    }


@torch.inference_mode()
def predict(model, rows, indices, climatology, teacher=False, batch_size=128, state_scores=False,
            crop_masks=True, score_years=None, window_years=None):
    means, stds = [], []
    device = rows.field.device
    errors = torch.zeros(3, 92, 8, device=device)
    counts = torch.zeros(92, 8, device=device)
    for start in range(0, len(indices), batch_size):
        selection = indices[start:start + batch_size]
        # 1981 cannot have a 1980 initial observation; retain these rows with a
        # one-year window rather than fabricating pre-archive states/weather.
        for years in ((1, 2) if window_years is None else (window_years,)):
            if window_years is None:
                keep = rows.year[selection].eq(0) if years == 1 else rows.year[selection].ne(0)
            else:
                keep = torch.ones(len(selection), dtype=torch.bool, device=device)
                if years == 2:
                    keep &= rows.year[selection].ne(0)
            subset = selection[keep]
            if not len(subset):
                continue
            batch = rows.batch(subset, climatology, years, crop_masks=crop_masks)
            context = (torch.autocast("cuda", dtype=torch.float16)
                       if device.type == "cuda" else contextlib.nullcontext())
            with context:
                rollout, output = run_batch(model, batch, float(teacher))
            mean, std = output[:2]
            # Return row indices so splitting archive-edge windows preserves order.
            means.append(torch.stack([subset, mean]))
            stds.append(torch.stack([subset, std]))
            if state_scores:
                truth = batch["observations"][:, 1:]
                present = torch.isfinite(truth)
                if score_years is not None:
                    allowed = torch.tensor(np.asarray(score_years) - 1981, device=device)
                    present &= torch.isin(batch["positions"][:, 1:] // 46, allowed)[..., None]
                target = torch.nan_to_num(truth)
                predicted = rollout["states"][:, :, 1:].mean(0)
                persistence = batch["initial"][:, None].expand_as(predicted)
                climo = climatology[batch["county"][:, None], batch["positions"][:, 1:] % 46]
                n = years * 46
                for k, estimate in enumerate((predicted, persistence, climo)):
                    errors[k, :n] += ((estimate - target).square() * present).sum(0)
                counts[:n] += present.float().sum(0)
    return torch.cat(means, 1), torch.cat(stds, 1), errors, counts


def align_predictions(output, indices):
    order = output[0].argsort()
    sorted_output = output[:, order]
    position = torch.searchsorted(sorted_output[0], indices)
    return sorted_output[1, position]


@torch.inference_mode()
def lead_scores(model, rows, indices, climatology, score_years, crop_masks=True):
    """Yield predictions plus lead errors: leads 1-46 from Jan-1 one-year windows.

    The merge happens inside inference mode; `predict` returns inference tensors,
    and in-place writes to them outside inference mode raise a RuntimeError.
    """
    mean, std, errors, counts = predict(model, rows, indices, climatology, state_scores=True,
                                        crop_masks=crop_masks, score_years=score_years)
    _, _, short_errors, short_counts = predict(
        model, rows, indices, climatology, state_scores=True, crop_masks=crop_masks,
        score_years=score_years, window_years=1)
    errors[:, :46], counts[:46] = short_errors[:, :46], short_counts[:46]
    return mean, std, errors, counts


def calibrate(model, rows, indices, climatology, calibration_years, crop_masks=True):
    mean, std, errors, counts = lead_scores(model, rows, indices, climatology,
                                            calibration_years, crop_masks)
    predicted, spread = align_predictions(mean, indices), align_predictions(std, indices)
    residual = (predicted - rows.target[indices]).abs() / spread.clamp_min(1e-6)
    quantiles = torch.quantile(residual, torch.tensor([.68, .95], device=residual.device))
    multiplier = (quantiles.prod() / 2).sqrt().clamp(.25, 4)
    model.yield_calibration.copy_(multiplier)
    model.state_error.copy_(torch.where(counts > 0, (errors[0] / counts.clamp_min(1)).sqrt(), torch.nan))
    return {"inner_year_sd_multiplier": multiplier.item(), "rows": len(indices)}


def evaluate_fold(data, field, fold, output=OUTPUT, epochs=24, batches=128, held=False):
    output.mkdir(parents=True, exist_ok=True)
    design_path = ROOT / "r2_fold_design.json"
    if design_path.exists():
        expected_design = json.loads(design_path.read_text())[fold]
        test_years = expected_design["test_years"]
    else:
        expected_design = None
        _, test_indices = list(GroupKFold(n_splits=5).split(data, groups=data.year))[fold]
        test_years = sorted(data.iloc[test_indices].year.unique().tolist())
    fit, calibration, test, fit_years, calibration_years = fold_masks(data, test_years)
    if expected_design is not None and (
        fit_years != expected_design["fit_years"] or calibration_years != expected_design["calibration_years"]
    ):
        raise ValueError("Dataset/fold mismatch against the frozen R2-D year design")
    target, _trend = target_for_fold(data, fit)
    group = [c for i, g in enumerate(crop_groups(data.crop.unique())) if i % 5 == fold for c in g]
    if held:
        keep = ~data.crop.isin(group).to_numpy()
        fit, calibration, test = fit & keep, calibration & keep, test & ~keep
    rows = FieldRows(data, field, target)
    started = time.monotonic()
    model, climatology, timing = train(rows, fit, fit_years, epochs, batches, held=held)
    stem = f"{'held' if held else 'fold'}_{fold}"
    # Uncalibrated weights survive any later evaluation failure (re-calibratable).
    save_recursive(model, output / f"{stem}_trained.pt", {
        "fold": fold, "held": held, "fit_years": fit_years, "calibration_years": calibration_years,
        "test_years": test_years, "timing": timing, "calibrated": False,
        "static_names": field.static_names, "trait_names": field.trait_names})
    calibration_indices = rows.indices(calibration & data.primary.to_numpy())
    record = calibrate(model, rows, calibration_indices, climatology, calibration_years,
                       crop_masks=not held)
    test_index = rows.indices(test)
    predicted, spread, errors, counts = lead_scores(model, rows, test_index, climatology, test_years)
    forced, forced_std, _, _ = predict(model, rows, test_index, climatology, teacher=True)
    result = data.loc[test, [
        "crop", "adm_id", "year", "series_id", "state_alpha", "harvested_acres",
        "irrigated_share", "irrigation_source", "primary", "prodn_practice_desc",
    ]].copy()
    result["target"] = np.asarray(rows.target[test_index].tolist())
    for name, value in (("recursive", predicted), ("recursive_std", spread),
                        ("teacher_forced", forced), ("teacher_forced_std", forced_std)):
        result[name] = np.asarray(align_predictions(value, test_index).tolist())
    result.to_parquet(output / f"{stem}.parquet", index=False)
    yield_report = summary(result.loc[result.primary])
    rmse = (errors / counts[None].clamp_min(1)).sqrt()
    report = {"fold": fold, "held": held, "held_crops": group if held else [],
              "test_years": test_years, "fit_years": fit_years,
              "calibration_years": calibration_years, "calibration": record,
              "yield": yield_report,
              "timing": timing, "seconds": time.monotonic() - started,
              "state_rmse": rmse.tolist(), "state_counts": counts.tolist(),
              "sources": field.sources, "state_slots": list(SLOTS),
              "state_forecast_design": (
                  "Leads 1-46 start Jan 1 of the held year; leads 47-92 start Jan 1 "
                  "of its preceding year. Only held-year outcomes are scored, never fitted context-year states.")}
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2))
    save_recursive(model, output / f"{stem}.pt", report | {
        "static_names": field.static_names, "trait_names": field.trait_names,
        "state_spread": "Inner-year lead-specific RMSE, including process/model error, not a causal interval.",
    })
    print(f"Completed {stem}; saved {output / f'{stem}.json'}", flush=True)
    print(json.dumps({"completed": stem, "macro_r2": yield_report["macro_r2"],
                      "seconds": report["seconds"]}), flush=True)
    return result, report


def final_fit(data, field, output=OUTPUT, epochs=24, batches=128):
    """Deployment fit: every year except the inner calibration years; no test score."""
    output.mkdir(parents=True, exist_ok=True)
    fit, calibration, _test, fit_years, calibration_years = fold_masks(data, [])
    target, _trend = target_for_fold(data, fit)
    rows = FieldRows(data, field, target)
    started = time.monotonic()
    model, climatology, timing = train(rows, fit, fit_years, epochs, batches)
    save_recursive(model, output / "final_trained.pt", {
        "fold": "final", "fit_years": fit_years, "calibration_years": calibration_years,
        "timing": timing, "calibrated": False,
        "static_names": field.static_names, "trait_names": field.trait_names})
    record = calibrate(model, rows, rows.indices(calibration & data.primary.to_numpy()),
                       climatology, calibration_years)
    report = {"fold": "final", "held": False, "test_years": [], "fit_years": fit_years,
              "calibration_years": calibration_years, "calibration": record, "timing": timing,
              "seconds": time.monotonic() - started, "sources": field.sources,
              "state_slots": list(SLOTS)}
    (output / "final.json").write_text(json.dumps(report, indent=2))
    save_recursive(model, output / "final.pt", report | {
        "static_names": field.static_names, "trait_names": field.trait_names,
        "state_spread": "Inner-year lead-specific RMSE, including process/model error, not a causal interval.",
    })
    print(json.dumps({"completed": "final", "seconds": report["seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--folds", default="0")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--held", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--require-t4", action="store_true")
    args = parser.parse_args()
    field = FieldData()
    if args.require_t4 and (field.device.type != "cuda" or "T4" not in torch.cuda.get_device_name()):
        raise RuntimeError("CUDA probe failed: this experiment requires a Tesla T4, never P100/CPU")
    print({"device": str(field.device), "weather": list(field.daily.shape), "sources": field.sources}, flush=True)
    data = pd.read_parquet(args.features)
    if args.final:
        final_fit(data, field, args.output, args.epochs, args.batches)
        return
    for fold in map(int, args.folds.split(",")):
        evaluate_fold(data, field, fold, args.output, args.epochs, args.batches, args.held)


if __name__ == "__main__":
    main()
