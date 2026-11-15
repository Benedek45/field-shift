"""Pure ledgers. Inputs/outputs remain torch tensors on the caller's device."""

from dataclasses import dataclass

import torch

from fieldshift.device import runtime_device


def accelerator():
    return runtime_device()


@dataclass
class FieldState:
    water_mm: torch.Tensor
    mineral_n_kg_ha: torch.Tensor
    organic_c_t_ha: torch.Tensor
    organic_n_kg_ha: torch.Tensor
    family_pressure: torch.Tensor
    erosion_t_ha: torch.Tensor
    residue_n_kg_ha: torch.Tensor
    residue_release_fraction: torch.Tensor


def water_stress(storage, capacity, p):
    """FAO-56 Eq 84; storage is water ABOVE permanent wilting point."""
    return (storage / ((1 - p) * capacity).clamp_min(1e-6)).clamp(0, 1)


def kc_curve(progress, initial, middle, end):
    """FAO-56 piecewise curve, normalized stage lengths 20/30/30/20%."""
    development = initial + (middle - initial) * ((progress - .2) / .3).clamp(0, 1)
    return torch.where(progress < .8, development,
                       middle + (end - middle) * ((progress - .8) / .2).clamp(0, 1))


def water_step(storage, rain, et0, kc, capacity, p, irrigated=False, active=True, irrigation_remaining=None):
    """FAO-56 Eq 85, early-day rain, no runoff/capillary rise, mm/day.

    Irrigation is ideal net application to field capacity when RAW is exceeded.
    Water exceeding field capacity drains after daily ET (Eq 88).
    """
    potential = et0.clamp_min(0) * kc
    adjusted_p = (p + .04 * (5 - potential)).clamp(.1, .8)
    wet = storage + rain.clamp_min(0)
    irrigation = torch.where((wet < (1 - adjusted_p) * capacity) & active,
                              (capacity - wet).clamp_min(0), 0) if irrigated else torch.zeros_like(wet)
    if irrigation_remaining is not None:
        irrigation = torch.minimum(irrigation, irrigation_remaining.clamp_min(0))
    wet = wet + irrigation
    ks = water_stress(wet, capacity, adjusted_p)
    actual = torch.minimum(wet, potential * ks)
    drainage = (wet - actual - capacity).clamp_min(0)
    new = wet - actual - drainage
    return new, {"et_mm": actual, "potential_et_mm": potential, "drainage_mm": drainage,
                 "irrigation_mm": irrigation, "ks": ks}


def nitrogen_step(mineral, organic, fixation, fertilizer, residue_n, harvest_n,
                  drainage_mm, capacity_mm, decay=.02):
    """Conservative annual two-store N ledger, kg N/ha.

    Fixation enters crop biomass; harvested N leaves; returned residue enters
    organic store. Non-fixed crop N is explicitly taken from mineral store.
    Missing N is reported, not created. Mixing leaching = 1-exp(-drainage/TAW).
    """
    mineralized = organic * decay
    organic_after = organic - mineralized
    available = mineral + fertilizer + mineralized
    demand = (harvest_n + residue_n - fixation).clamp_min(0)
    uptake = torch.minimum(demand, available)
    shortage = demand - uptake
    # When demand exceeds supply, scale biomass N flows, preserving both stores.
    flow_scale = ((uptake + fixation) / (harvest_n + residue_n).clamp_min(1e-6)).clamp(max=1)
    returned = residue_n * flow_scale
    removed = harvest_n * flow_scale
    unused_fixed = (fixation - removed - returned).clamp_min(0)
    mineral_before_leach = available - uptake + unused_fixed
    leached = mineral_before_leach * (1 - torch.exp(-drainage_mm / capacity_mm.clamp_min(1)))
    return mineral_before_leach - leached, organic_after + returned, {
        "fixation": fixation, "fertilizer": fertilizer, "mineralization": mineralized,
        "harvest_removal": removed, "residue_return": returned, "leaching": leached,
        "shortage": shortage,
    }


def carbon_step(carbon_t_ha, above_c_t_ha, below_c_t_ha, humification_above=.15,
                humification_below=.35, decay=.02):
    """One slow pool. Inputs are plant C (t C/ha), split as in Bolinder et al. 2007.

    Root-derived C is humified about 2.3 times more efficiently than above-ground
    residue C (Kätterer et al. 2011, Ultuna, h 0.35 vs ~0.15); decay is the RothC
    HUM rate 0.02/year. Combining the two is a low-confidence simplification.
    """
    retained = above_c_t_ha * humification_above + below_c_t_ha * humification_below
    decay_loss = carbon_t_ha * decay
    return carbon_t_ha + retained - decay_loss, {"retained_c": retained, "decay_c": decay_loss}


def plant_carbon(product_t_ha, dry_matter, harvest_index, shoot_root, extra_root, residue_return=1.):
    """Bolinder et al. 2007 Eqs 2-5 on dry matter, 0.45 g C/g: returns
    (above-ground residue C, root + extra-root C) in t C/ha."""
    dry = product_t_ha * dry_matter
    hi = harvest_index.clamp(.05, 1)
    above = dry * (1 - hi) / hi * .45 * residue_return
    root = dry / (shoot_root * hi) * .45
    return above, root * (1 + extra_root)


def renard_freimund_r(annual_rain_mm):
    """Annual erosivity from annual rain (Renard & Freimund 1994), MJ mm/(ha h yr)."""
    p = annual_rain_mm
    return torch.where(p < 850, .0483 * p.pow(1.61), 587.8 - 1.219 * p + .004105 * p.square()).clamp_min(0)


def rusle(r, k, slope_pct, slope_length_m, cover, practice=1.):
    """RUSLE SI: R MJ mm/(ha h year), K t ha h/(ha MJ mm); output t/ha/year."""
    theta = torch.atan(slope_pct / 100)
    ls = (slope_length_m / 22.13) ** .5 * (
        65.41 * torch.sin(theta) ** 2 + 4.56 * torch.sin(theta) + .065
    )
    return r * k * ls * cover * practice


def pest_penalty(consecutive, maximum, plateau, family_pressure, family_prior=.015):
    """Bounded build-up; a year away halves generic family pressure."""
    direct = torch.where(consecutive > 0,
                         maximum * (1 + .5 * ((consecutive - 1) / plateau).clamp(0, 1)), 0)
    return (direct + family_prior * family_pressure).clamp(0, .5)


def fao_relative_yield(stage_features, traits, soil=None, irrigated_share=0):
    """FAO-33 stage response product; SD is unavailable, not an invented CI."""
    potential = stage_features["potential_et"]
    # A stage with no simulated days (perennial stand, truncated season) carries no deficit.
    deficit = torch.where(potential > 0, 1 - stage_features["actual_et"] / potential.clamp_min(1e-6), 0)
    response = (1 - stage_features["ky"] * deficit.clamp(0, 1)).clamp(0, 1).prod(-1)
    heat = stage_features["flowering_heat_dd"]
    frost = stage_features["reproductive_frost_days"]
    # Smooth stress multiplier: explicit low-confidence response priors.
    response = response * torch.exp(-.02 * heat - .05 * frost)
    return response, torch.full_like(response, float("nan"))
