"""K-State 2026 FMG (dated Nov-01-2025), middle-yield budgets, USD/acre.

Numbers transcribed from the abstract/summary/detail versions in each exact PDF.
They are projected input costs, NOT observed KFMA profits or our yield forecasts.
Non-alfalfa hay uses MF2143 brome (2015), not annual forage sorghum.
"""
from urllib.parse import quote

ROOT = "https://www.agmanager.info/sites/default/files/wysiwyg/"
FILES = {
    "wheat": "Wheat (Rotation) Cost-Return Budget in South Central KS__Nov-01-2025.pdf",
    "soybean": "Soybean Cost-Return Budget in South Central Kansas__Nov-01-2025.pdf",
    "corn": "Corn Cost-Return Budget in South Central Kansas__Nov-01-2025.pdf",
    "sorghum": "Grain Sorghum Cost-Return Budget in South Central Kansas__Nov-01-2025.pdf",
    "alfalfa": "Alfalfa Cost-Return Budget in South Central and Southeast Kansas__Nov-01-2025.pdf",
    "double": "Double-Crop Soybeans Cost-Return Budget_Southeast KS and adjacent areas_Nov-01-2025.pdf",
    "irr_corn": "Irrigated Corn (center-pivot) Cost-Return Budget_South Central Kansas _Nov-01-2025.pdf",
    "irr_soybean": "Irrigated Soybeans (center-pivot) Cost-Return Budget_South Central KS_Nov-01-2025.pdf",
    "irr_sorghum": "Irrigated Grain Sorghum (center-pivot) Cost-Return Budget_South Central KS_Nov-01-2025.pdf",
    "irr_wheat": "Irrigated Wheat Cost-Return Budget in Western KS_Limited irrigation_Nov-01-2025.pdf",
    "irr_alfalfa": "Irrigated Alfalfa (Center-Pivot) Cost-Return Budget_Western Kansas_Nov-01-2025.pdf",
}
# direct, capital recovery (NO cash rent), seed, fertilizer total, N bill, N lb,
# chemicals, additional/operator/irrigation labour bills, diesel+repairs,
# custom operations, pumping bill, acre-inches, baling bill, hay US tons.
ROWS = {
    "wheat": (206.19,53.82,31.50,102.86,61.30,101,10.79,1.62,8.45,0,26.76,0,0,0,0,0),
    "soybean": (182.13,52.68,45.50,41.10,0,6,29.58,1.67,7.86,0,26.81,0,0,0,0,0),
    "corn": (312.07,61.16,67.20,97.55,47.60,100,50.63,1.18,10.21,0,31.52,17.69,0,0,0,0),
    "sorghum": (206.48,55.68,11.66,90.54,46.33,98,20.07,.92,9.21,0,27.38,17.69,0,0,0,0),
    "alfalfa": (310.77,13.45,28.20,92.63,0,11,16.83,.85,2.67,0,7.34,145.40,0,0,84.59,3),
    "double": (114.67,40.85,52.50,12.95,0,2,9.11,1.37,5.85,0,20.30,0,0,0,0,0),
    "irr_corn": (525.64,228.27,108,182.73,96.90,204,74.15,1.27,10.60,12.76,40.12,27.28,24.73,14.5,0,0),
    "irr_soybean": (262.37,209.76,49,63.93,0,12,58.26,1.72,7.29,7.92,29.08,0,15.35,9,0,0),
    "irr_sorghum": (316.53,211.61,18.66,123.95,68.03,142,38.73,1.36,8.50,8.80,30.01,17.69,17.06,10,0,0),
    "irr_wheat": (256.94,270.13,22.75,77.04,46.75,97,36.03,1.57,10.05,5.28,32.14,17.69,18.06,6,0,0),
    "irr_alfalfa": (467.58,220.60,28.20,70.83,0,17,40.15,1.81,4.05,16.72,15.84,175.94,57.19,19,94.86,5),
}
BROME = "https://www.coffey.k-state.edu/farm-management/Brome%20Hay%20Cost%20Return%20Budget.pdf"


def budget(crop, irrigated):
    """Return cited native cost parts; conversions to hectares happen in context."""
    key = {"winter_wheat": "wheat", "double_crop_soybean": "double"}.get(crop, crop)
    if key == "hay":
        # MF2143 April2015 p2, middle2.5US-ton yield. Custom fees are inseparable
        # operating services even though the contractor recovers ownership.
        return {
            "source": BROME + " MF2143 April2015 p2, brome proxy for mixed perennial hay; nominal2015, no inflation adjustment; custom fees unsplit operating",
            "parts": {"seed":1.90, "fertilizer_other":90-44.80, "chemicals":6.72,
                      "labour":16.88, "harvest":3.33*18.83,
                      "other_operating":228.39-1.90-90-6.72-16.88-3.33*18.83},
            "n_price":44.80/(80*.45359237), "hours":16.88/15,
            "harvest_ref":2.5, "native_labour":True, "region_transfer":True,
            "energy":None, "direct":228.39, "reference_n":80*.45359237,
        }
    if irrigated and "irr_"+key in ROWS:
        key = "irr_"+key
    if key not in ROWS:
        return None
    direct, fixed, seed, fert, nbill, nlb, chem, additional, operator, irr_lab, fuel, _custom, pump, inches, baling, tons = ROWS[key]
    parts = {"seed":seed, "fertilizer_other":fert-nbill, "chemicals":chem,
             "labour":additional+operator+irr_lab, "fuel_repairs":fuel,
             "fixed_overhead":fixed, "harvest":baling}
    # Remainder includes the explicitly operating insurance, working-capital
    # interest, misc, consulting, and non-baling custom services. No land charged.
    parts["other_operating"] = direct-pump-nbill-sum(v for k,v in parts.items() if k != "fixed_overhead")
    return {
        "source": ROOT+quote(FILES[key],safe="") + " 2026 FMG pp2–4, middle yield; direct versus capital recovery; excludes cash rent",
        "parts":parts, "n_price":nbill/(nlb*.45359237),
        "hours":(additional+irr_lab)/22+operator/24,
        "harvest_ref":tons, "native_labour":True,
        "region_transfer":key in ("double","irr_wheat","irr_alfalfa") or (irrigated and not key.startswith("irr_")),
        "energy":pump/inches if inches else None, "direct":direct,
        "reference_n":nlb*.45359237,
    }
