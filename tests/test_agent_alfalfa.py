"""Stand accounting contracts; synthetic weather is not field validation."""
import numpy as np
import pandas as pd
import pytest
import torch

from fieldshift.planner import data
from fieldshift.planner.search import market_support
from fieldshift.sim.alfalfa import schedule
from fieldshift.sim.components import accelerator
from fieldshift.sim.economics import ACRES_HA, ECON_FIELDS, context, reprice
from fieldshift.sim.engine import ledger_pass, prepare, trajectories, water_pass
from fieldshift.sim.traits import CATALOG

gpu = pytest.mark.skipif(not (torch.backends.mps.is_available() or torch.cuda.is_available()),
                         reason="No CPU simulator fallback")


def test_census_practice_short_tons_and_recent_suppressed_fallback(monkeypatch):
    rows = []
    for state, year, practice, stat, unit, value in (
        ("48",2022,"IRRIGATED, ENTIRE CROP","YIELD","TONS / ACRE",4),
        ("48",2022,"IRRIGATED, NONE OF CROP","YIELD","TONS / ACRE",1.5),
        ("28",2017,"IRRIGATED, ENTIRE CROP","YIELD","TONS / ACRE",7.5),
        ("28",2022,"ALL PRODUCTION PRACTICES","PRODUCTION","TONS",164),
        ("28",2022,"ALL PRODUCTION PRACTICES","AREA HARVESTED","ACRES",115),
    ):
        rows.append(dict(zip(("state_ansi","year","prodn_practice_desc","statisticcat_desc","unit_desc","value"),
                             (state,year,practice,stat,unit,value),strict=True)) |
                    {"source_desc":"CENSUS","commodity_desc":"HAY","class_desc":"ALFALFA"})
    monkeypatch.setattr(data,"state_yields",lambda:pd.DataFrame(rows))
    # Published NASS2022 TX practice yields4/1.5 US-ton/ac; MS total164/115.
    assert data.alfalfa_normal("48",True)[0] == pytest.approx(4*2.2417)
    assert data.alfalfa_normal("48",False)[0] == pytest.approx(1.5*2.2417)
    value, source = data.alfalfa_normal("28",True)
    assert value == pytest.approx(164/115*2.2417)
    assert "ALL PRODUCTION" in source and "suppressed" in source and "LOW" in source


def test_market_is_area_depth_not_binary_presence():
    assert market_support("alfalfa",{"areas":{"alfalfa":1,"corn":99}}) == .01
    assert market_support("alfalfa",{"areas":{}}) == 0


@gpu
def test_one_stand_cash_event_not_production_plus_establishment():
    ctx = context({"adm_id":"US-48-303"},[CATALOG["alfalfa"]],[],None,True)
    fields = {k:torch.zeros(1,device=accelerator()) for k in ECON_FIELDS}
    fields["economic_establishment"][:] = 1
    fields["economic_establishment_fall"][:] = 1
    fields["n_fertilizer_kg_ha"][:] = 40*1.12085
    fields["irrigation_mm"][:] = 6*25.4
    # Reproduce TAMU2026 D3 budget at ITS6.75$/acre-inch, not D2's10$ default.
    op, listed, irr, _ = reprice(fields,ctx,{"irrigation_energy_usd_per_mm_ha":6.75*ACRES_HA/25.4})
    assert float(op[0]) == pytest.approx(-386.53*ACRES_HA,abs=.002)
    assert float(listed[0]) == pytest.approx(-(386.53+49.92)*ACRES_HA,abs=.002)
    assert float(irr[0]) == pytest.approx((40.50+27.38)*ACRES_HA,abs=.002)
    quoted, _, _, q = reprice(fields,ctx,{"crop_cost_usd_ha":{"alfalfa":{"seed":500}}})
    native, _, _, _ = reprice(fields,ctx)
    assert float(quoted[0]-native[0]) == pytest.approx(178*ACRES_HA-500,abs=.002)
    assert q["crops"][0]["establishment"]["fall"]["parts"]["seed"]["source"] == "farmer"


@gpu
def test_stand_lifecycle_cap_and_nitrogen_credit():
    # Deliberately controlled warm/dry weather, not an invented county observation.
    dates = pd.date_range("1998-01-01","2009-12-31")
    daily = pd.DataFrame({"date":dates,"tavg":20.,"tmin":14.,"tmax":26.,
                          "prec":1.,"et0":4.,"vpd":1.,"rad":15.,"gwetroot":.6,"rsm":.6})
    soil = {"awc_mm_per_m":150.,"profile_depth_m":2.,"bulk_density_kg_m3":1300.,
            "organic_matter_pct":3.,"rusle_k":.03,"slope_pct":2.,"slope_length_m":100.}
    site = {"adm_id":"US-48-303","daily":daily,"soil":soil,"planting":{}}
    crops = [CATALOG[k] for k in ("alfalfa","corn","spring_wheat")]
    p = prepare([site],crops,[],None,learned=False,irrigated=True)
    rotations = [((0,0,0,0),0,False),((0,0,1,1),0,False),((0,0,2,2),0,False)]
    tr = trajectories(rotations,crops,12,starts=[0])
    stand = schedule(tr,crops,["48"],p["device"])
    uniform = torch.as_tensor(tr["rotation"] == 0,device=p["device"])
    assert bool((stand["establish"][uniform].sum(-1) == 2).all())  #8years/two stands
    assert bool((stand["harvest_fraction"][uniform] == 0).sum(-1).eq(2).all())
    unlimited = water_pass(p,[soil],crops,[],None,tr,True)
    limited = water_pass(p,[soil | {"irrigation_capacity_mm_yr":100}],crops,[],None,tr,True)
    assert float(limited["irrigation_mm"].max()) <= 100.001
    assert bool((limited["irrigation_mm"][uniform,-1] > 0).all())
    # A fixed learned mean of1 must NOT erase capped/uncapped FAO deficits.
    p["learned"] = torch.ones((1,3,12,2),device=p["device"])
    norms = np.array([[9.,10.,3.]],dtype=np.float32)
    a = ledger_pass(p,unlimited,[soil],norms,crops,[],None,tr,True)
    b = ledger_pass(p,limited,[soil],norms,crops,[],None,tr,True,
                    base_fao={"fao_main":unlimited["fao_main"]})
    seed = stand["establish"]
    assert bool((a["harvest_t_ha"][seed] == 0).all())
    assert float(b["yield_relative"][uniform].mean()) < float(a["yield_relative"][uniform].mean())
    assert float(a["n_balance_error_kg_ha"].abs().max()) < .005
    c = torch.as_tensor(tr["crop"],device=p["device"])
    after = c[:,:-2].eq(0)
    corn = c[:,1:-1].eq(1) & after
    wheat = c[:,1:-1].eq(2) & after
    scored = torch.arange(8,device=p["device"])[None] >= 4
    assert torch.allclose(a["n_fertilizer_kg_ha"][corn & scored],
                          torch.full_like(a["n_fertilizer_kg_ha"][corn & scored],30*1.12085))
    assert bool((a["n_fertilizer_kg_ha"][wheat & scored] > 0).all())  #old min(rate,0) bug
