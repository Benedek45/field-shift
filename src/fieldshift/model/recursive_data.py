"""Resident daily/8-day observations and causal, crop-specific development drivers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from fieldshift.features.traits import CROPS, numeric_traits
from fieldshift.model.dataset import soil_table
from fieldshift.model.expanded_data import weather_spec
from fieldshift.model.neural import accelerator_device

ROOT = Path("data/processed/recursive")
SLOTS = ("root_wetness", "surface_wetness", "profile_wetness", "ndvi", "lai",
         "fpar", "et_actual_mm_day", "surface_air_temperature_proxy")
AUXILIARY = ("smap_rootzone", "smap_surface", "gldas_rootzone")
STATE_SCALES = (.15, .20, .15, .15, 1., .15, 2., 3.)
DEFAULT_STATE = (.5, .5, .5, .3, 1., .4, 2., 0.)
DRIVERS = (
    "tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "wind", "rh",
    "gdd_increment", "development_fraction", "heat_critical", "heat_30", "heat_35",
    "frost_days", "dry_days", "establishment", "vegetative", "flowering", "grain_fill",
    "forage_regrowth", "crop_present", "day_length", "period_days", "doy_sin", "doy_cos",
    "flowering_heat_critical", "flowering_heat_30", "flowering_heat_35",
)


def calendar(start="1981-01-01", end="2025-12-31"):
    dates = pd.date_range(start, end)
    year = dates.year.to_numpy()
    doy = dates.day_of_year.to_numpy()
    step = (year - year.min()) * 46 + (doy - 1) // 8
    first = np.flatnonzero(np.r_[True, np.diff(step) != 0])
    count = np.bincount(step)
    return dates, step, first, count


def prepare_auxiliary(root=ROOT):
    """Ingest crop-mask observations once; all numerical aggregation is on-device."""
    root.mkdir(parents=True, exist_ok=True)
    path, meta = weather_spec("power")
    del path
    dates, step, _, _ = calendar(meta["start_date"], meta.get("end_date", "2025-12-31"))
    out = root / "cybench_canopy.f32"
    shape = (len(meta["counties"]), int(step[-1]) + 2, 4)
    if not out.exists():
        device = accelerator_device()
        sums = torch.zeros(np.prod(shape), device=device)
        counts = torch.zeros_like(sums)
        index = pd.Index(meta["counties"])
        audit = {}
        for crop_number, crop in enumerate(("maize", "wheat")):
            for variable_number, variable in enumerate(("ndvi", "fpar")):
                source = Path(f"data/raw/cybench/cybench-data/{crop}/US/{variable}_{crop}_US.csv")
                channel = crop_number * 2 + variable_number
                retained = 0
                for frame in pd.read_csv(source, chunksize=500_000, usecols=["adm_id", "date", variable]):
                    day = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
                    k = ((day.dt.year - dates[0].year) * 46 + (day.dt.day_of_year - 1) // 8).to_numpy()
                    c = index.get_indexer(frame.adm_id)
                    v = frame[variable].to_numpy(dtype=np.float32)
                    valid = ((c >= 0) & (k >= 0) & (k < shape[1] - 1) & np.isfinite(v)
                             & (v >= (0 if variable == "fpar" else -.1))
                             & (v <= (100 if variable == "fpar" else 1)))
                    # CY-Bench FPAR is percent, unlike the model's fractional FPAR.
                    values = torch.tensor(v[valid], device=device) / (100 if variable == "fpar" else 1)
                    flat = (c[valid] * shape[1] + k[valid] + 1) * 4 + channel
                    position = torch.tensor(flat, dtype=torch.long, device=device)
                    sums.index_add_(0, position, values)
                    counts.index_add_(0, position, torch.ones_like(values))
                    retained += int(valid.sum())
                audit[f"{crop}_{variable}"] = {"source": str(source), "retained": retained}
        result = torch.where(counts > 0, sums / counts.clamp_min(1), torch.nan).reshape(shape)
        result.cpu().numpy().astype("float32").tofile(out)
        out.with_suffix(".json").write_text(json.dumps({"shape": shape, "sources": audit}, indent=2))
        del result, counts, sums
    locations = pd.concat([
        pd.read_csv(f"data/raw/cybench/cybench-data/{crop}/US/location_{crop}_US.csv")
        for crop in ("maize", "wheat")
    ]).drop_duplicates("adm_id").set_index("adm_id").reindex(meta["counties"])
    static = soil_table().reindex(meta["counties"]).join(locations[["latitude", "longitude"]])
    weights = pd.read_parquet("data/processed/weather/power_county_weights.parquet")
    for name, column, offset, scale in (("latitude", "cell_row", -90, .5),
                                        ("longitude", "cell_col", -180, .625)):
        value = (offset + scale * weights[column]) * weights.weight
        fallback = value.groupby(weights.adm_id).sum().reindex(meta["counties"])
        static[name] = static[name].fillna(fallback)
    static.to_parquet(root / "fallback_static.parquet")
    return root


def _variable_records(meta):
    variables = meta["variables"]
    if isinstance(variables, dict):
        return list(variables), variables
    if variables and isinstance(variables[0], dict):
        return [v["name"] for v in variables], {v["name"]: v for v in variables}
    return variables, {}


EO_BOUNDS = {
    # Operational physical bounds in canonical units. LAI matches the decoder's
    # maximum 8; it is deliberately stricter than the MODIS encoded range 0..10.
    "ndvi": (-1., 1.), "ndvi_source": (0., 1.), "lai": (0., 8.), "fpar": (0., 1.),
    "et_actual": (0., 25.), "lst_day": (-100., 100.), "lst_night": (-100., 100.),
    "smap_rootzone": (0., 1.), "smap_surface": (0., 1.), "gldas_rootzone": (0., 1.),
}


def validate_eo_values(variable, values):
    """NaN is missing; infinity or values outside stated canonical bounds are faults."""
    if variable not in EO_BOUNDS:
        if torch.isinf(values).any():
            raise ValueError(f"EO {variable} contains infinity")
        return
    low, high = EO_BOUNDS[variable]
    invalid = torch.isinf(values) | (values < low) | (values > high)
    if invalid.any():
        raise ValueError(f"EO {variable} outside physical bounds [{low}, {high}]: "
                         f"{int(invalid.sum())} cells; fix the source product, do not clip it")


class FieldData:
    """Daily weather stays on the accelerator; observations are never future drivers."""

    def __init__(self, root=ROOT):
        self.device = accelerator_device()
        self.path, self.meta = weather_spec("power")
        meta = self.meta
        self.crops = sorted(CROPS)
        self.trait_names = sorted(numeric_traits(CROPS[self.crops[0]]))
        self.traits = torch.tensor(
            [[numeric_traits(CROPS[c])[k] for k in self.trait_names] for c in self.crops],
            dtype=torch.float32, device=self.device)
        self.trait_index = {k: i for i, k in enumerate(self.trait_names)}
        self.daily = torch.tensor(np.memmap(self.path, mode="r", dtype="float32",
                                            shape=tuple(meta["shape"])),
                                  dtype=torch.float16, device=self.device)
        self.variables = {k: i for i, k in enumerate(meta["variables"])}
        dates, steps, first, counts = calendar(meta["start_date"], meta.get("end_date", "2025-12-31"))
        self.step_dates = dates[first]
        self.n_steps = len(first)
        self.n_counties = len(meta["counties"])
        self.step = torch.tensor(steps, dtype=torch.long, device=self.device)
        self.counts = torch.tensor(counts, dtype=torch.float32, device=self.device)
        self.first = torch.tensor(first, dtype=torch.long, device=self.device)
        self.step_doy = torch.tensor(dates[first].day_of_year.to_numpy(), device=self.device)
        self.year_steps = torch.tensor(dates[first].year.to_numpy(), device=self.device)
        self.day_doy = torch.tensor(dates.day_of_year.to_numpy(), device=self.device)
        sums = torch.zeros(self.n_counties, self.n_steps, self.daily.shape[-1], device=self.device)
        sums.index_add_(1, self.step, self.daily.float())
        means = sums / self.counts[None, :, None]
        selected = []
        for name in ("tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "wind", "rh"):
            selected.append((sums if name in {"prec", "rad", "et0"} else means)[..., self.variables[name]])
        self.weather = torch.stack(selected, -1)
        del sums, means
        self.observations = torch.full((self.n_counties, self.n_steps + 1, len(SLOTS)), torch.nan,
                                       device=self.device)
        wet = [self.variables[k] for k in ("gwetroot", "gwettop", "gwetprof")]
        self.observations[:, :self.n_steps, :3] = self.daily[:, self.first][:, :, wet].float()
        cy_meta = json.loads((root / "cybench_canopy.json").read_text())
        self.canopy = torch.tensor(np.memmap(root / "cybench_canopy.f32", mode="r", dtype="float32",
                                            shape=tuple(cy_meta["shape"])), device=self.device)
        generic_ndvi = torch.where(torch.isfinite(self.canopy[..., 0]), self.canopy[..., 0], self.canopy[..., 2])
        generic_fpar = torch.where(torch.isfinite(self.canopy[..., 1]), self.canopy[..., 1], self.canopy[..., 3])
        self.observations[..., 3], self.observations[..., 5] = generic_ndvi, generic_fpar
        self.generic_is_cy = torch.isfinite(torch.stack([generic_ndvi, generic_fpar], -1))
        self.auxiliary = torch.full((self.n_counties, self.n_steps + 1, len(AUXILIARY)),
                                    torch.nan, device=self.device)
        self.sources = {"wetness": "POWER/MERRA-2, period-start observations",
                        "canopy": str(root / "cybench_canopy.json")}
        eo_path = Path("data/processed/eo/eo_county_8day.f32")
        if eo_path.exists() and eo_path.with_suffix(".json").exists():
            self._eo(eo_path)
        static_path = Path("data/processed/static/county_static.parquet")
        static = pd.read_parquet(root / "fallback_static.parquet")
        if static_path.exists():
            extra = pd.read_parquet(static_path)
            if "adm_id" in extra:
                extra = extra.set_index("adm_id")
            extra = extra.reindex(meta["counties"])
            for column in extra.select_dtypes(include="number"):
                if column.startswith("grace_") or column in {"n_cells", "centroid_lat", "centroid_lon"}:
                    continue  # Whole-history trends are not time-invariant statics.
                static[column] = extra[column].to_numpy()
            for old, new in (("latitude", "centroid_lat"), ("longitude", "centroid_lon")):
                if new in extra:
                    static[old] = extra[new].fillna(static[old])
            self.sources["static"] = str(static_path)
        self.static_names = [c for c in static.select_dtypes(include="number")
                             if c not in {"fips", "index", "year", "county_id", "state_id"}
                             and not c.endswith(("_fips", "_code", "_id"))]
        static = static.reindex(meta["counties"]) if static.index.name == "adm_id" else static
        self.static = torch.tensor(static[self.static_names].to_numpy(dtype=np.float32), device=self.device)
        density = (static["bulk_density_0_100"].fillna(static.bulk_density)
                   if "bulk_density_0_100" in static else static.bulk_density).to_numpy(dtype=float)
        density = np.where(density > 20, density / 1000, density)
        porosity = np.clip(1 - density / 2.65, .2, .8)
        self.porosity = torch.tensor(np.nan_to_num(porosity, nan=.45), dtype=torch.float32, device=self.device)
        latitude = static["latitude"].to_numpy(dtype=float)
        if not np.isfinite(latitude).all():
            raise ValueError("Latitude unavailable for some POWER counties; need the static data product")
        self.latitude = torch.tensor(latitude, dtype=torch.float32, device=self.device)
        # Fixed soil reservoirs, independent of crop changes: no water is created by
        # changing a rooting-depth trait at a year boundary. Effective proxy capacities.
        self.capacities = torch.tensor([180., 40., 400.], device=self.device).expand(self.n_counties, -1)
        self.date_tables = {}
        start_ordinal = dates[0].toordinal()
        years = range(dates[0].year - 1, dates[-1].year + 2)
        for name, month, day in (("jan", 1, 1), ("feb", 2, 1), ("mar", 3, 1), ("jun", 6, 15),
                                 ("aug", 8, 1), ("aug_end", 8, 31), ("sep", 9, 1),
                                 ("oct_end", 10, 31), ("dec", 12, 1), ("dec_end", 12, 31)):
            self.date_tables[name] = torch.tensor(
                [pd.Timestamp(y, month, day).toordinal() - start_ordinal for y in years],
                 dtype=torch.long, device=self.device)
        self.planting = torch.full((self.n_counties, len(self.crops), len(years)), -1,
                                   dtype=torch.long, device=self.device)
        planting_path = root / "planting_dates.parquet"
        if planting_path.exists():
            planting = pd.read_parquet(planting_path)
            states = pd.read_parquet(root / "county_states.parquet")
            planting = states.merge(planting, on="state_alpha")
            c = pd.Index(meta["counties"]).get_indexer(planting.adm_id)
            crop = pd.Index(self.crops).get_indexer(planting.crop)
            y = planting.year.to_numpy() - 1980
            valid = (c >= 0) & (crop >= 0) & (y >= 0) & (y < len(years))
            ordinal = pd.to_datetime(planting.date).map(pd.Timestamp.toordinal).to_numpy() - start_ordinal
            index = [torch.tensor(v[valid], device=self.device) for v in (c, crop, y)]
            self.planting[tuple(index)] = torch.tensor(ordinal[valid], dtype=torch.long, device=self.device)
            self.sources["planting"] = "Observed state-level 50% planting dates (given management, not forecast)"

    def _eo(self, path):
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["shape"][:2] != [self.n_counties, self.n_steps]:
            raise ValueError("EO archive must match POWER counties and 8-day steps")
        if "counties" in meta and meta["counties"] != self.meta["counties"]:
            raise ValueError("EO county order differs from POWER")
        names, records = _variable_records(meta)
        data = torch.tensor(np.memmap(path, mode="r", dtype="float32", shape=tuple(meta["shape"])),
                            device=self.device)
        for variable in names:
            values = data[..., names.index(variable)]
            unit = records.get(variable, {}).get("units", "")
            if variable == "fpar" and unit in {"%", "percent"}:
                values = values / 100
            if variable == "et_actual" and unit in {"mm", "mm/8day", "mm/period", "mm/8 days", "mm/8-day"}:
                values = values / self.counts[None, :]
            if variable.startswith("lst_") and unit in {"K", "kelvin"}:
                values = values - 273.15
            validate_eo_values(variable, values)
        quality_path = path.parent / "ndvi_adjustment.parquet"
        poor_ndvi = None
        if quality_path.exists():
            quality = pd.read_parquet(quality_path).set_index("adm_id").reindex(self.meta["counties"])
            poor_ndvi = torch.tensor(quality.r2.fillna(0).to_numpy() < .3, device=self.device)
            pre_modis = torch.tensor(
                pd.to_datetime(self.step_dates) < pd.Timestamp("2000-02-18"), device=self.device)
            self.sources["ndvi_quality_rule"] = (
                "Before MODIS (2000-02-18), mask adjusted AVHRR NDVI where harmonisation r2 < .3 or missing; "
                "quality fit uses the source product's full overlap, not yield labels.")
        for variable, slot in (("ndvi", 3), ("lai", 4), ("fpar", 5), ("et_actual", 6)):
            if variable not in names:
                continue
            values = data[..., names.index(variable)]
            record = records.get(variable, {})
            unit = record.get("units", "") if isinstance(record, dict) else str(record)
            if variable == "fpar" and unit in {"%", "percent"}:
                values = values / 100
            if variable == "ndvi" and poor_ndvi is not None:
                values = values.masked_fill(poor_ndvi[:, None] & pre_modis[None], torch.nan)
            if variable == "et_actual" and unit not in {"mm/day", "mm d-1"}:
                if unit not in {"mm", "mm/8day", "mm/period", "mm/8 days", "mm/8-day"}:
                    raise ValueError(f"Unrecognised EO ET units: {unit!r}")
                values = values / self.counts[None, :]
            old = self.observations[:, 1:, slot]
            self.observations[:, 1:, slot] = torch.where(torch.isfinite(values), values, old)
            if slot in (3, 5):
                self.generic_is_cy[:, 1:, 0 if slot == 3 else 1] &= ~torch.isfinite(values)
        if "lst_day" in names:
            record = records.get("lst_day", {})
            unit = record.get("units", "") if isinstance(record, dict) else str(record)
            lst = data[..., names.index("lst_day")]
            if unit in {"K", "kelvin"}:
                lst = lst - 273.15
            elif unit not in {"degC", "C", "°C"}:
                raise ValueError(f"Unrecognised LST units: {unit!r}")
            self.observations[:, 1:, 7] = lst - self.weather[..., 2]
        for channel, variable in enumerate(AUXILIARY):
            if variable not in names:
                continue
            record = records.get(variable, {})
            if not isinstance(record, dict) or record.get("units") != "m3/m3":
                raise ValueError(f"Unrecognised volumetric moisture units for {variable}")
            self.auxiliary[:, 1:, channel] = data[..., names.index(variable)]
        self.sources["eo"] = str(path.with_suffix(".json"))
        used = {"ndvi", "lai", "fpar", "et_actual", "lst_day", *AUXILIARY}
        self.sources["eo_additional_channels_not_assimilated"] = [n for n in names if n not in used]
        if any(v.get("adjustment") for v in meta["variables"] if isinstance(v, dict)):
            self.sources["ndvi_reference_warning"] = (
                "Source AVHRR-to-MODIS reference harmonisation uses their full overlap; "
                "the model's fits/climatologies still use fit years only.")

    def climatology(self, fit_years, crop_masks=True):
        valid_year = torch.isin(self.year_steps, torch.tensor(fit_years, device=self.device))
        observations = self.observations[:, :self.n_steps].view(self.n_counties, -1, 46, len(SLOTS))
        if not crop_masks:
            observations = observations.clone()
            source = self.generic_is_cy[:, :self.n_steps].view(self.n_counties, -1, 46, 2)
            for channel, slot in enumerate((3, 5)):
                observations[..., slot].masked_fill_(source[..., channel], torch.nan)
        mask = valid_year.view(-1, 46)[None, :, :, None] & torch.isfinite(observations)
        sums = torch.where(mask, observations, 0.).sum(1)
        count = mask.float().sum(1)
        default = torch.tensor(DEFAULT_STATE, device=self.device)
        return torch.where(count > 0, sums / count.clamp_min(1), default)

    def targets(self, county, crop, positions, crop_masks=True):
        observed = self.observations[county[:, None], positions].clone()
        if not crop_masks:
            source = self.generic_is_cy[county[:, None], positions]
            for channel, slot in enumerate((3, 5)):
                observed[..., slot].masked_fill_(source[..., channel], torch.nan)
            return observed
        for names, channel in ((("corn", "corn_silage"), 0),
                               (("winter_wheat", "spring_wheat", "durum"), 2)):
            match = torch.isin(crop, torch.tensor([self.crops.index(n) for n in names], device=self.device))
            specific = self.canopy[county[:, None], positions, channel:channel + 2]
            for source, target in ((0, 3), (1, 5)):
                values = specific[..., source]
                observed[..., target] = torch.where(match[:, None] & torch.isfinite(values),
                                                     values, observed[..., target])
        return observed

    def drivers(self, county, crop, target_year, years=2, harvest_only=None, daily_transform=None,
                return_dates=False):
        """Daily GDD/stages -> MODIS 8-day drivers, with no future state observations."""
        b, device, steps = len(county), self.device, years * 46
        first_year = target_year - (years - 1)  # zero = 1981
        base_day = self.date_tables["jan"][first_year + 1]
        offset = torch.arange(160 + 366 * years, device=device) - 160
        day = base_day[:, None] + offset[None, :]
        valid = day.ge(0) & day.lt(self.daily.shape[1])
        safe = day.clamp(0, self.daily.shape[1] - 1)
        raw = self.daily[county[:, None], safe].float()
        if daily_transform is not None:
            raw = daily_transform(raw, day)
        t = self.traits[crop]
        get = lambda name: t[:, self.trait_index[name]]
        tmin, tmax, tavg = (raw[..., self.variables[k]] for k in ("tmin", "tmax", "tavg"))
        rolling = F.avg_pool1d(F.pad(tavg[:, None], (9, 0)), 10, stride=1).squeeze(1)
        lower, upper = get("base_c")[:, None], get("upper_c")[:, None]
        daily_gdd = ((torch.minimum(torch.maximum(tmin, lower), upper)
                      + torch.minimum(torch.maximum(tmax, lower), upper)) / 2 - lower).clamp_min(0)
        derived = torch.zeros(b, len(offset), 18, device=device)
        autumn, perennial = get("autumn_sown").gt(.5), get("perennial").gt(.5)
        for season in range(years + 1):
            harvest = first_year + season + 1  # +1 indexes the 1980-based date tables
            sow_year = harvest - autumn.long()
            start = torch.where(autumn, self.date_tables["sep"][sow_year],
                                torch.where(perennial, self.date_tables["feb"][sow_year],
                                            self.date_tables["mar"][sow_year]))
            stop = torch.where(autumn, self.date_tables["dec"][sow_year],
                               self.date_tables["jun"][sow_year])
            warm = rolling.ge(get("sowing_temp_c")[:, None])
            suitable = torch.where(autumn[:, None], rolling.le(get("sowing_temp_c")[:, None]), warm)
            candidate = valid & day.ge(start[:, None]) & day.le(stop[:, None]) & suitable
            sow = torch.where(candidate, day, 10 ** 8).amin(1)
            sow = torch.where(sow.lt(10 ** 8), sow, stop)
            managed = self.planting[county, crop, harvest]
            sow = torch.where(managed.ge(0) & ~perennial, managed, sow)
            end = torch.where(perennial, self.date_tables["dec_end"][harvest],
                              torch.where(autumn, self.date_tables["aug_end"][harvest],
                                          self.date_tables["oct_end"][harvest]))
            active = valid & day.ge(sow[:, None]) & day.le(end[:, None])
            if harvest_only is not None:
                active &= harvest.eq(harvest_only + 1)[:, None]
            gdd = torch.where(active, daily_gdd, 0.)
            chill = active & tavg.ge(get("vernalization_min_c")[:, None]) & tavg.le(get("vernalization_max_c")[:, None])
            unlocked = (chill.float().cumsum(1).ge(get("vernalization_days")[:, None])
                        & day.ge(self.date_tables["jan"][harvest, None]))
            dormant = torch.where(tavg.lt(get("dormancy_c")[:, None]), 0., gdd)
            locked = torch.minimum(torch.where(unlocked, 0., dormant).cumsum(1),
                                   get("pre_vernalization_gdd")[:, None])
            increments = torch.diff(locked, dim=1, prepend=torch.zeros_like(locked[:, :1]))
            gdd = torch.where(autumn[:, None], torch.where(unlocked, dormant, increments), gdd)
            before = gdd.cumsum(1) - gdd
            maturity = get("maturity_gdd")
            horizon = torch.where(perennial, 6 * get("cut_interval_gdd"), maturity)
            active &= perennial[:, None] | before.lt(maturity[:, None])
            frost_candidate = active & day.ge(self.date_tables["aug"][harvest, None]) & tmin.lt(get("frost_c")[:, None])
            first_frost = torch.where(frost_candidate, day, 10 ** 8).amin(1)
            active &= ~perennial[:, None] | day.lt(first_frost[:, None])
            boundaries = torch.stack([get("emergence_gdd"), get("flowering_gdd"),
                                      get("flowering_gdd") + get("flowering_duration_gdd")], -1)
            stage = (before[..., None] >= boundaries[:, None]).sum(-1)
            stage = torch.where(perennial[:, None], 4, stage)
            phase = F.one_hot(stage, 5).float() * active[..., None]
            if return_dates and season == years - 1:
                flower_day = torch.where(active & stage.eq(2), day, 10 ** 8).amin(1)
                mature_day = torch.where(
                    valid & day.ge(sow[:, None]) & day.le(end[:, None])
                    & (before + gdd).ge(maturity[:, None]), day, 10 ** 8).amin(1)
                stage_dates = torch.stack([sow, flower_day, mature_day], -1).float()
                stage_dates[:, 1:] = torch.where(
                    stage_dates[:, 1:].ge(10 ** 8) | perennial[:, None], torch.nan, stage_dates[:, 1:])
            heat = (tmax - get("heat_critical_c")[:, None]).clamp_min(0)
            heat30, heat35 = (tmax - 30).clamp_min(0), (tmax - 35).clamp_min(0)
            frost_limit = torch.where(stage.lt(2), get("vegetative_frost_c")[:, None], get("frost_c")[:, None])
            flower = phase[..., 2]
            columns = [
                gdd * active, (before / horizon[:, None]) * active, heat * active,
                heat30 * active, heat35 * active, (tmin < frost_limit).float() * active,
                raw[..., self.variables["prec"]].lt(1).float() * active,
                *phase.unbind(-1), active.float(), heat * flower, heat30 * flower, heat35 * flower,
                active.float() * ((season == years - 1) if harvest_only is None
                                  else harvest.eq(harvest_only + 1)[:, None]),
                active.float() * get("perennial")[:, None],
            ]
            # Missing annual thresholds are intentionally NaN for forage; no such
            # threshold is allowed into an arithmetic input.
            derived += torch.nan_to_num(torch.stack(columns, -1))
        local_step = self.step[safe] - first_year[:, None] * 46
        inside = valid & local_step.ge(0) & local_step.lt(steps)
        flat = torch.where(inside, torch.arange(b, device=device)[:, None] * steps
                           + local_step.clamp(0, steps - 1), b * steps).flatten()
        sums = torch.zeros(b * steps + 1, 18, device=device).index_add_(
            0, flat, torch.where(inside[..., None], derived, 0.).reshape(-1, 18))[:-1].view(b, steps, 18)
        position = first_year[:, None] * 46 + torch.arange(steps, device=device)[None]
        weather = self.weather[county[:, None], position]
        if daily_transform is not None:
            variables = [self.variables[k] for k in
                         ("tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "wind", "rh")]
            transformed = torch.zeros(b * steps + 1, 9, device=device).index_add_(
                0, flat, torch.where(inside[..., None], raw[..., variables], 0.).reshape(-1, 9)
            )[:-1].view(b, steps, 9)
            denominator = self.counts[position][..., None].expand(-1, -1, 9).clone()
            denominator[..., [3, 4, 6]] = 1.
            weather = transformed / denominator
        days = self.counts[position]
        doy = self.step_doy[position].float() + (days - 1) / 2
        latitude = torch.deg2rad(self.latitude[county])[:, None]
        declination = .409 * torch.sin(2 * torch.pi * doy / 365 - 1.39)
        day_length = 24 / torch.pi * torch.acos((-latitude.tan() * declination.tan()).clamp(-1, 1))
        scales = torch.tensor([30, 30, 30, 40, 120, 2, 30, 5, 100], device=device)
        count = sums[..., 12].clamp_min(1)
        driver = torch.cat([
            weather / scales, sums[..., :1] / 120,
            (sums[..., 1] / count)[..., None],
            sums[..., 2:5] / torch.tensor([20, 40, 20], device=device),
            sums[..., 5:7] / days[..., None], sums[..., 7:12] / days[..., None],
            (sums[..., 12] / days)[..., None], (day_length / 12)[..., None],
            (days / 8)[..., None], torch.sin(2 * torch.pi * doy / 365)[..., None],
            torch.cos(2 * torch.pi * doy / 365)[..., None],
            sums[..., 13:16] / torch.tensor([20, 40, 20], device=device),
        ], -1)
        result = driver, sums[..., 16] / days, position
        return (*result, stage_dates) if return_dates else result


if __name__ == "__main__":
    prepare_auxiliary()
