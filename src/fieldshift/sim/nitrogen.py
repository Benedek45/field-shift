"""Monthly conservative N stores, including a living winter-cover store.

The mineralisation timing (Q10 2, 2%/yr at 20 C) and complete-mixing leaching
remain low-confidence process scenarios, not a tile-drain calibration. Residue
is a separate store, avoiding mineralising the same N both as SOM and residue.
"""

import torch


def monthly_nitrogen(state, water, t, fertilizer, fixed, harvest_n, residue_n, residue_rate,
                      cover_cn, cover_root_factor, cover_fixed_fraction, capacity, diagnostics=False,
                      store_state=False):
    mineral, organic, fast, fast_rate_mass, live = state
    zero = torch.zeros_like(mineral)
    flows = {k: zero.clone() for k in ("mineralization", "leaching", "shortage", "harvest_removal",
                                      "residue_return", "cover_uptake", "cover_fixed")}
    before = mineral + organic + fast + live
    weights = water["uptake_month"][:, t]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    # The N in roots is included in cover uptake; fixation enters living
    # biomass, never directly into the soluble mineral pool.
    growth_n = water["cover_growth_month"][:, t] * .45 * 1000 * cover_root_factor[:, None] / cover_cn[:, None]
    cash_live = zero.clone()
    fixation_total = zero.clone()
    log = {k: [] for k in ("mineral_start", "mineralized", "applied", "demand", "uptake", "unmet",
                            "drainage", "leach_fraction", "leached")} if diagnostics else None
    if diagnostics and store_state:
        log.update({k: [] for k in ("state_mineral", "state_organic", "state_fast",
                                    "state_fast_rate_mass", "state_live_cover")})
    for month in range(12):
        mineral_start = mineral.clone() if diagnostics else None
        activity = water["temperature_month"][:, t, month]
        slow_release = organic * (1 - torch.exp(-.02 * activity))
        rate = fast_rate_mass / fast.clamp_min(1e-6)
        release = fast * (1 - torch.exp(-rate * activity))
        organic = organic - slow_release
        fast = fast - release
        fast_rate_mass = (fast_rate_mass - release * rate).clamp_min(0)
        mineral = mineral + slow_release + release
        flows["mineralization"] += slow_release + release
        demand_cover = growth_n[:, month]
        cv_fixed = demand_cover * cover_fixed_fraction
        cv_take = torch.minimum(mineral, (demand_cover - cv_fixed).clamp_min(0))
        mineral = mineral - cv_take
        live = live + cv_take + cv_fixed
        flows["cover_uptake"] += cv_take
        flows["cover_fixed"] += cv_fixed
        # At termination the overwinter cover becomes residue, still protected
        # from leaching until it mineralises. It is not new N at this point.
        term = water["termination_month"][:, t] == month
        returned_cover = torch.where(term, live, 0)
        live = live - returned_cover
        cv_fraction = (12.5 / cover_cn).clamp(.1, .8)
        cv_rate = -torch.log1p(-cv_fraction)
        fast = fast + returned_cover
        fast_rate_mass = fast_rate_mass + returned_cover * cv_rate
        applied = torch.where(water["plant_month"][:, t] == month, fertilizer, 0)
        if "month_active" in water:
            applied *= water["month_active"][:, t, month]
        mineral = mineral + applied
        # Whole-crop uptake follows the cash crop's potential-transpiration
        # season, not the December mineral balance.
        demand = (harvest_n + residue_n) * weights[:, month]
        fixation = torch.minimum(fixed * weights[:, month], demand)
        take = torch.minimum(mineral, (demand - fixation).clamp_min(0))
        mineral = mineral - take
        cash_live += take + fixation
        fixation_total += fixation
        flows["shortage"] += (demand - take - fixation).clamp_min(0)
        # Deposit current main-crop residues only after uptake/harvest.
        # Autumn-sown crops can have uptake after calendar harvest; any late
        # uptake is returned in December rather than silently lost.
        harvest = (water["harvest_month"][:, t] == month) | (month == 11)
        scale = (cash_live / (harvest_n + residue_n).clamp_min(1e-6)).clamp(max=1)
        removed = torch.where(harvest, harvest_n * scale, 0)
        returned = torch.where(harvest, cash_live - removed, 0)
        cash_live = torch.where(harvest, 0, cash_live)
        fast = fast + returned
        fast_rate_mass = fast_rate_mass + returned * residue_rate
        flows["harvest_removal"] += removed
        flows["residue_return"] += returned
        leach = mineral * (1 - torch.exp(-water["drain_month"][:, t, month] / capacity.clamp_min(1)))
        mineral -= leach
        flows["leaching"] += leach
        if diagnostics:
            for key, value in {
                "mineral_start": mineral_start, "mineralized": slow_release + release,
                "applied": applied, "demand": demand, "uptake": take + fixation,
                "unmet": (demand - take - fixation).clamp_min(0),
                "drainage": water["drain_month"][:, t, month],
                "leach_fraction": 1 - torch.exp(-water["drain_month"][:, t, month] / capacity.clamp_min(1)),
                "leached": leach,
            }.items():
                log[key].append(value.clone())
            if store_state:
                for key, value in zip(("state_mineral", "state_organic", "state_fast",
                                        "state_fast_rate_mass", "state_live_cover"),
                                       (mineral, organic, fast, fast_rate_mass, live), strict=True):
                    log[key].append(value.clone())
    after = mineral + organic + fast + live
    flows["balance_error"] = (after - before - fertilizer - fixation_total - flows["cover_fixed"]
                               + flows["harvest_removal"] + flows["leaching"])
    if diagnostics:
        flows["monthly"] = {k: torch.stack(v, -1) for k, v in log.items()}
    return (mineral, organic, fast, fast_rate_mass, live), flows
