"""Public multi-year recursive simulation for edited daily weather and crop plans."""

import contextlib

import numpy as np
import pandas as pd
import torch

from fieldshift.features.traits import CROPS, numeric_traits
from fieldshift.model.neural import accelerator_device
from fieldshift.model.probabilistic import SEEDS
from fieldshift.model.recursive import load_recursive
from fieldshift.model.recursive_data import DRIVERS, SLOTS, FieldData, calendar

WEATHER = ("tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "wind", "rh")
MODEL = "data/processed/models/recursive_field_r3.pt"


class Scenario(FieldData):
    """Same accelerator driver engine, but no archive IDs or future state products."""

    def __init__(self, weather, records, trait_names, soil):
        frame = weather.copy()
        if "date" in frame:
            frame = frame.set_index("date")
        frame.index = pd.DatetimeIndex(frame.index)
        expected = pd.date_range(frame.index.min(), frame.index.max())
        if (not frame.index.equals(expected) or frame.index[0].month != 1
                or frame.index[0].day != 1 or frame.index[-1].month != 12 or frame.index[-1].day != 31):
            raise ValueError("Weather must contain contiguous whole calendar years, January 1–December 31")
        missing = set(WEATHER) - set(frame)
        if missing or not np.isfinite(frame[list(WEATHER)].to_numpy(float)).all():
            raise ValueError(f"Finite daily forcing required, in model units: {WEATHER}; missing={sorted(missing)}")
        if (frame.tmin > frame.tmax).any() or frame.prec.lt(0).any() or frame.et0.lt(0).any():
            raise ValueError("Invalid weather: require Tmin <= Tmax, precipitation/ET0 >=0")
        latitude = soil.get("latitude", soil.get("centroid_lat"))
        if latitude is None or not np.isfinite(latitude) or abs(latitude) >= 90:
            raise ValueError("A finite field latitude in (-90,90) is required for day length")
        self.device = accelerator_device()
        self.crops = list(records)
        self.trait_names = list(trait_names)
        self.trait_index = {k: i for i, k in enumerate(trait_names)}
        self.traits = torch.tensor([[records[c][k] for k in trait_names] for c in self.crops],
                                   dtype=torch.float32, device=self.device)
        self.variables = {name: i for i, name in enumerate(WEATHER)}
        self.daily = torch.tensor(frame[list(WEATHER)].to_numpy(np.float32)[None],
                                   dtype=torch.float16, device=self.device)
        dates, step, first, counts = calendar(frame.index[0], frame.index[-1])
        self.n_steps, self.n_counties = len(first), 1
        self.step = torch.tensor(step, dtype=torch.long, device=self.device)
        self.counts = torch.tensor(counts, dtype=torch.float32, device=self.device)
        self.step_doy = torch.tensor(dates[first].day_of_year.to_numpy(), device=self.device)
        sums = torch.zeros(1, self.n_steps, len(WEATHER), device=self.device)
        sums.index_add_(1, self.step, self.daily.float())
        means = sums / self.counts[None, :, None]
        self.weather = torch.stack([
            (sums if name in {"prec", "rad", "et0"} else means)[..., i]
            for i, name in enumerate(WEATHER)
        ], -1)
        self.latitude = torch.tensor([latitude], dtype=torch.float32, device=self.device)
        self.start_year, self.end_year = dates[0].year, dates[-1].year
        self.first_dates = dates[first]
        self.date_tables = {}
        years = range(self.start_year - 1, self.end_year + 2)
        origin = dates[0].toordinal()
        for name, month, day in (("jan", 1, 1), ("feb", 2, 1), ("mar", 3, 1), ("jun", 6, 15),
                                 ("aug", 8, 1), ("aug_end", 8, 31), ("sep", 9, 1),
                                 ("oct_end", 10, 31), ("dec", 12, 1), ("dec_end", 12, 31)):
            self.date_tables[name] = torch.tensor(
                [pd.Timestamp(y, month, day).toordinal() - origin for y in years],
                dtype=torch.long, device=self.device)
        self.planting = torch.full((1, len(self.crops), len(years)), -1, dtype=torch.long, device=self.device)


def _numeric(record):
    if all(isinstance(v, (int, float, np.number)) for v in record.values()):
        return dict(record)
    return numeric_traits(record)


def _share(value):
    if value is None:
        return np.nan
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("irrigated_share must be None or a finite fraction in [0,1]")
    return float(value)


@torch.inference_mode()
def simulate(initial_state, drivers, crop_sequence, traits=None, soil=None, irrigated_share=None,
             model_path=MODEL):
    """Roll one field continuously; never accepts future wetness/canopy as forcing.

    `drivers`: daily DataFrame, whole years; C, mm/day, MJ/m²/day, kPa, m/s, %.
    `crop_sequence`: [{crop, year, sowing_date?, irrigated_share?}], one crop per
    MODIS period. Supply a prior weather year for the first winter crop.
    `traits`: optional crop->complete numeric/source-tagged trait record, including
    new crops. `soil` uses checkpoint static names and must include latitude.
    `initial_state`: named slot dictionary, or previous result's final_state.

    State spread is an inner-year lead-RMSE floor plus member disagreement, NOT a
    validated causal intervention interval. Yield log SD uses inner-year scaling.
    Beyond two years, calibration extrapolation is explicitly flagged.
    """
    if not crop_sequence:
        raise ValueError("At least one planned harvest season is required")
    soil = dict(soil or {})
    for name, alias in (("latitude", "centroid_lat"), ("longitude", "centroid_lon")):
        if name not in soil and alias in soil:
            soil[name] = soil[alias]
    model, metadata = load_recursive(model_path)
    device = model.center.device
    records = {name: _numeric(CROPS[name]) for name in CROPS}
    records.update({name: _numeric(value) for name, value in (traits or {}).items()})
    needed = {entry["crop"] for entry in crop_sequence}
    if not needed.issubset(records):
        raise ValueError(f"Unknown crops need trait records: {sorted(needed - records.keys())}")
    records = {name: records[name] for name in sorted(needed)}
    for name, values in records.items():
        missing = set(metadata["trait_names"]) - values.keys()
        if missing:
            raise ValueError(f"Crop {name} missing traits: {sorted(missing)}")
    scenario = Scenario(drivers, records, metadata["trait_names"], soil)
    years = scenario.end_year - scenario.start_year + 1
    end_year = torch.tensor([years - 1], dtype=torch.long, device=device)
    county = torch.zeros(1, dtype=torch.long, device=device)
    masks, crop_drivers = [], []
    origin = pd.Timestamp(scenario.start_year, 1, 1).toordinal()
    for entry in crop_sequence:
        crop, year = entry["crop"], int(entry["year"])
        if not scenario.start_year <= year <= scenario.end_year:
            raise ValueError(f"Harvest {year} is outside the weather window")
        if records[crop]["autumn_sown"] and year == scenario.start_year:
            raise ValueError("The first winter harvest needs a preceding weather year to calculate GDD")
        c = scenario.crops.index(crop)
        if "sowing_date" in entry:
            date = pd.Timestamp(entry["sowing_date"])
            if records[crop].get("perennial", 0):
                raise ValueError("Established forage has green-up, not a sowing date")
            expected_year = year - int(records[crop]["autumn_sown"])
            if date.year != expected_year:
                raise ValueError("Sowing date and crop habit disagree with the harvest year")
            scenario.planting[0, c, year - scenario.start_year + 1] = date.toordinal() - origin
        generated, active, _ = scenario.drivers(
            county, torch.tensor([c], device=device), end_year, years,
            harvest_only=torch.tensor([year - scenario.start_year], device=device))
        masks.append(active)
        crop_drivers.append(generated)
    occupancy = torch.stack([mask.gt(0) for mask in masks]).sum(0)
    if occupancy.gt(1).any().item():
        raise ValueError("Crop seasons overlap in an 8-day period; supply a feasible non-overlapping sequence")
    full = crop_drivers[0].clone()
    crop_columns = list(range(9, 22)) + list(range(26, 29))
    full[..., crop_columns] = sum(item[..., crop_columns] for item in crop_drivers)
    # Exogenous soil/traits can change with a crop plan without resetting water or
    # the recurrent latent. Off-season uses that calendar year's planned crop.
    static_names = metadata["static_names"]
    static = torch.empty(1, years * 46, model.static_features, device=device)
    share = torch.full((1, years * 46), _share(irrigated_share), device=device)
    sequence = sorted(zip(crop_sequence, masks), key=lambda item: item[0]["year"])
    default_entry = sequence[0][0]
    for calendar_year in range(scenario.start_year, scenario.end_year + 1):
        candidates = [entry for entry, _ in sequence if entry["year"] == calendar_year]
        entry = candidates[0] if candidates else default_entry
        values = [soil.get(name, np.nan) for name in static_names]
        values += [records[entry["crop"]][name] for name in metadata["trait_names"]]
        values += [_share(entry.get("irrigated_share", irrigated_share))]
        k = (calendar_year - scenario.start_year) * 46
        static[:, k:k + 46] = torch.tensor(values, dtype=torch.float32, device=device)
        share[:, k:k + 46] = values[-1]
        default_entry = entry
    for entry, mask in sequence:
        values = [soil.get(name, np.nan) for name in static_names]
        values += [records[entry["crop"]][name] for name in metadata["trait_names"]]
        values += [_share(entry.get("irrigated_share", irrigated_share))]
        active = mask.gt(0)
        static[active] = torch.tensor(values, dtype=torch.float32, device=device)
        share[active] = values[-1]
    values = initial_state.get("values", initial_state)
    if set(SLOTS) - values.keys():
        raise ValueError(f"Initial values must include all named slots: {SLOTS}")
    initial = torch.tensor([[values[name] for name in SLOTS]], dtype=torch.float32, device=device)
    if not torch.isfinite(initial).all().item() or initial[:, :3].lt(0).any().item() or initial[:, :3].gt(1).any().item():
        raise ValueError("Initial state must be finite, with all wetness fractions in [0,1]")
    if "members" in initial_state:
        initial = torch.tensor(initial_state["members"], dtype=torch.float32, device=device)[:, None]
        if initial.shape != (len(SEEDS), 1, len(SLOTS)):
            raise ValueError("Initial member-state shape is invalid")
    hidden = None
    if "latent" in initial_state:
        hidden = torch.tensor(initial_state["latent"], dtype=torch.float32, device=device)
        if hidden.shape != (len(SEEDS), model.layers, 1, model.hidden):
            raise ValueError("Initial recurrent latent shape is invalid")
    capacity = torch.tensor([[180., 40., 400.]], device=device)
    context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
    with context:
        rollout = model.rollout(initial, full, static, capacity, share, hidden=hidden)
        states = rollout["states"][:, 0]
        mean_state, epistemic = states.mean(0), states.std(0, correction=0)
        lead = torch.arange(years * 46 + 1, device=device)
        floor = model.state_error[(lead - 1).clamp(0, 91)]
        spread = torch.maximum(epistemic, torch.nan_to_num(floor))
        spread[0] = epistemic[0]
        n = len(masks)
        readout = {
            "states": rollout["states"].expand(-1, n, -1, -1),
            "context": rollout["context"].expand(-1, n, -1, -1),
            "static": rollout["static"].expand(n, -1, -1),
        }
        season = torch.cat(masks)
        mean, sd = model.yield_distribution(readout, full.expand(n, -1, -1), season)[:2]
        variance = sd.square()
        arithmetic_mean = (mean + variance / 2).exp()
        arithmetic_sd = arithmetic_mean * variance.expm1().clamp_min(0).sqrt()
        last = torch.where(season > 0, lead[None, :-1], 0).amax(1)
        yield_values = torch.stack([arithmetic_mean, arithmetic_sd, mean, sd, last.lt(92)], -1)
    yields = [dict(entry) | dict(zip(
        ("relative_yield_mean", "relative_yield_std", "log_ratio_mean", "log_ratio_std",
         "within_calibration_horizon"), values))
        for entry, values in zip(crop_sequence, yield_values.tolist())]
    for entry in yields:
        entry["within_calibration_horizon"] = bool(entry["within_calibration_horizon"])
    dates = list(scenario.first_dates) + [pd.Timestamp(scenario.end_year + 1, 1, 1)]
    output = {
        "states": pd.DataFrame(mean_state.tolist(), index=dates, columns=SLOTS),
        "state_std": pd.DataFrame(spread.tolist(), index=dates, columns=SLOTS),
        "state_calibration_available": (torch.isfinite(floor) & lead[:, None].gt(0)
                                         & lead[:, None].le(92)).tolist(),
        "yields": yields,
        "final_state": {"values": dict(zip(SLOTS, mean_state[-1].tolist())),
                        "members": states[:, -1].tolist(), "latent": rollout["hidden"].tolist()},
        "water_fluxes": pd.DataFrame(rollout["fluxes"][:, 0].mean(0).tolist(),
                                     index=dates[:-1], columns=[
                                         "prec_mm", "irrigation_mm", "et_mm", "runoff_mm",
                                         "drainage_mm", "balance_residual_mm"]),
        "development_drivers": pd.DataFrame(full[0].tolist(), index=dates[:-1], columns=DRIVERS),
        "limitations": (
            "Effective soil reservoirs, county-trained proxies; automatic irrigation policy; "
            "not causal validation. Calibration was evaluated on at most two-year rollouts. "
            "Whole-year weather required; first winter crop needs a preceding weather year."
        ),
    }
    return output
