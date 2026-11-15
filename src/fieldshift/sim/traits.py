"""Ledger traits: representative reference scenarios, not local quotations.

All mass units are dry metric tonnes or kg N per hectare; money is USD/ha.
Low-confidence economic assumptions are surfaced, never called predictions.
"""

from dataclasses import dataclass

from fieldshift.features.traits import CROPS, Trait, load_traits

FAO56 = "https://www.fao.org/4/x0490e/x0490e0b.htm; Tables 11,12"
FAO22 = "https://www.fao.org/4/x0490e/x0490e0e.htm; Tables 22,24"
FAO33 = "https://www.fao.org/4/x0490e/x0490e0e.htm; FAO-33 water-production function"
NUTRIENTS = "https://bookstore.ksre.ksu.edu/pubs/mf2586.pdf; crop nutrient removal"
SARE = "https://www.sare.org/resources/managing-cover-crops-profitably-3rd-edition/"
ERS = "https://www.ers.usda.gov/data-products/commodity-costs-and-returns"
PRICES = "https://www.nass.usda.gov/Publications/Todays_Reports/reports/agpran25.pdf"
SEIFERT = "https://doi.org/10.1088/1748-9326/aa5444; Seifert et al. 2017"
CARBON = "https://www.rothamsted.ac.uk/rothamsted-carbon-model-rothc; HUM k=0.02/year"
RUSLE = "https://www.ars.usda.gov/ARSUserFiles/64080530/RUSLE/AH_703.pdf"
BOLINDER = ("https://doi.org/10.1016/j.agee.2006.05.013; Bolinder et al. 2007 Tables 3-4 and Eqs 2-5 "
            "(S:R small grains 7.4, corn 5.6, soybean 5.2, forages 1.6; extra-root C = 0.65 x root C)")
MOISTURE = "https://www.ams.usda.gov/sites/default/files/media/GrainInspectionHandbookII.pdf; market moisture bases"
AQUACROP = ("https://www.fao.org/aquacrop/resources/referencemanuals/en/; AquaCrop normalized water productivity "
            "WP* (C3 15-20 g/m2) and cold stress on transpiration (AquaCrop-OS default 12 GDD/day for full transpiration)")
TYNDALL = ("https://bmpcosttools.nrem.iastate.edu/files/inline-files/2016%20Cost%20Sheet%20for%20Cereal%20Rye%20Cover%20Crop.pdf; "
           "Tyndall & Bowman 2016, Iowa cereal rye cover ~$63/acre seed+seeding+termination (range $57-68)")
NICHOLS = ("https://doi.org/10.3389/fsufs.2023.1040927; Nichols & MacKenzie 2023 (rye needs ~100 GDD base 0 C before "
            "1 December is a fall-growth guideline, NOT a binary mortality threshold here; corn planted <14 days after rye termination risks yield loss)")
RYE_THERMAL = ("https://www.sare.org/publications/managing-cover-crops-profitably/nonlegume-cover-crops/cereal-rye/; "
               "germination >=34 F (1.11 C), vegetative growth >=38 F (3.33 C); late autumn sowing can overwinter "
               "with little autumn growth and resume in spring. Air temperature is a soil-temperature proxy.")
DAVIS = ("https://doi.org/10.1371/journal.pone.0047149; Davis et al. 2012 Boone IA 2003–2011: "
         "diverse small-grain/forage-legume systems corn +4%, soy +9%; manufactured N 80/16/11 kg/ha/yr; "
         "established-phase margins $953/965/913. Included manure, understory legumes and extra labor; transfer is LOW confidence.")
ALFALFA_N = ("https://www.agronext.iastate.edu/soilfertility/info/07CAScornsheet.pdf; ISU: corn after "
             "established alfalfa most often needs 0–30 lb N/acre; use 30 as a scenario, not a field recommendation")
BENDER_N = ("https://nutrien-ekonomics.com/wp-content/uploads/2019/11/Corn_Hybrid_Research.pdf; "
            "Bender et al., Modern Corn Hybrids' Nutrient Uptake Patterns: two-thirds of N acquired by VT/R1; "
            "256 lb/ac whole-plant and 148 lb/ac grain N at 230 bu/ac. Smooth within-stage interpolation is a low-confidence prior.")
PERRONE = "https://doi.org/10.1007/s10705-020-10055-z; Perrone et al. 2020, 59-75% of hairy vetch N from fixation (Minnesota)"
MARCILLO = ("https://doi.org/10.2489/jswc.72.3.226; Marcillo & Miguez 2017 meta-analysis: grass cover crops leave corn "
            "yield unchanged (response ratio 1.00, 0.98-1.02); legume covers +30% only at low N fertilizer")
CTIC = ("https://ctic.org/national-cover-crop-survey/; SARE/CTIC/ASTA farmer surveys, self-reported yield after covers: "
        "corn +1.9% / soybean +2.8% (2015-16), corn +2% / soybean +5% (2019-20)")
DOUBLE = ("https://ilagronomy.info/publications/crop-management/double-crop-soybeans/; Illinois double-crop soybean "
          "40-60% of full-season yield; Piatt County 2016-2023 trial 41 vs 83 bu/acre")
N_RATES = ("https://crops.extension.iastate.edu/encyclopedia/dont-double-dip-soybean; historical Iowa MRTN "
           "example at price ratio 0.10: corn after corn 181, after soy 124 lb N/acre; scenario, not current advice")
VETCH_CREDIT = ("https://www.sare.org/publications/managing-cover-crops-profitably/legume-cover-crops/hairy-vetch/; "
                "available N approx half total top-growth N; cap at 88 lb/acre from UNL 2026 wheat/legume/corn trial "
                "https://cropwatch.unl.edu/legume-cover-crops-after-winter-wheat-provide-meaningful-nitrogen-credits-corn/")


@dataclass(frozen=True)
class Crop:
    key: str
    traits: dict

    def v(self, name):
        return self.traits[name].value

    def override(self, **values):
        return Crop(self.key, self.traits | {
            k: Trait(v, self.traits[k].source + "; sensitivity scenario", "low")
            for k, v in values.items()
        })


# Kc initial, mid, end; depletion p; seasonal Ky; family; price $/t;
# operating cost $/ha (excludes land); harvest N kg/t; harvest index; normal t/ha.
# FAO table entries are direct; prices/costs are deliberately rounded reference
# scenarios, to be replaced by the user's bid and enterprise budget.
TABLE = {
    "corn": (.3, 1.2, .35, .55, 1.25, 1, 165, 1600, 12, .5, 10),
    "soybean": (.4, 1.15, .5, .5, .85, 2, 370, 1000, 60, .45, 3),
    "winter_wheat": (.7, 1.15, .25, .55, 1.05, 1, 210, 850, 23, .45, 3.5),
    "spring_wheat": (.3, 1.15, .25, .55, 1.15, 1, 210, 850, 23, .45, 3),
    "barley": (.3, 1.15, .25, .55, 1, 1, 210, 800, 20, .45, 3),
    "oats": (.3, 1.15, .25, .55, 1, 1, 220, 800, 20, .45, 2.5),
    "rye": (.7, 1.15, .25, .55, 1, 1, 220, 750, 20, .4, 2.5),
    "durum": (.3, 1.15, .25, .55, 1.15, 1, 260, 900, 23, .45, 3),
    "sorghum": (.3, 1.05, .55, .55, .9, 1, 160, 1000, 18, .45, 5),
    "sunflower": (.35, 1.15, .35, .45, .95, 3, 420, 1050, 30, .35, 2),
    "cotton": (.35, 1.15, .5, .65, .85, 4, 1600, 2400, 35, .2, 1),
    "rice": (1.05, 1.2, .9, .2, 1.1, 1, 300, 2200, 15, .5, 7),
    "potatoes": (.5, 1.15, .75, .35, 1.1, 5, 230, 8000, 4, .75, 35),
    "sugarbeets": (.35, 1.2, .7, .55, 1, 6, 55, 2500, 2, .7, 55),
    "dry_beans": (.4, 1.15, .35, .45, 1.15, 2, 750, 1500, 35, .4, 2),
    "peanuts": (.4, 1.15, .6, .5, .7, 2, 550, 1800, 45, .4, 3),
    "flaxseed": (.35, 1.1, .25, .5, 1, 7, 500, 850, 35, .3, 1.5),
    "alfalfa": (.4, .95, .9, .55, 1.1, 2, 180, 1100, 28, .85, 8),
    "hay": (.4, .95, .85, .6, 1, 1, 140, 750, 20, .85, 6),
    "corn_silage": (.3, 1.2, .6, .55, 1.25, 1, 50, 1900, 4, .85, 35),
}


# FAO-56 Table 22 upper Zr: footnote 1 says larger values suit rainfed stress modelling.
ROOT_MAX = {"corn": 1.7, "corn_silage": 1.7, "soybean": 1.3, "winter_wheat": 1.8, "spring_wheat": 1.5,
            "durum": 1.5, "barley": 1.5, "oats": 1.5, "sorghum": 2.0, "sunflower": 1.5, "cotton": 1.7,
            "rice": 1.0, "potatoes": .6, "sugarbeets": 1.2, "dry_beans": .9, "peanuts": 1.0,
            "flaxseed": 1.5, "alfalfa": 2.0, "hay": 1.5}


# Bolinder 2007 Tables 3-4; sorghum and other cereals take the closest cereal value.
SHOOT_ROOT = {"corn": 5.6, "corn_silage": 5.6, "sorghum": 5.6, "soybean": 5.2, "winter_wheat": 7.4,
              "spring_wheat": 7.4, "durum": 7.4, "barley": 7.4, "oats": 7.4, "rye": 7.4,
              "alfalfa": 1.6, "hay": 1.6}
# Market moisture bases (dry-matter share of harvested product).
DRY_MATTER = {"corn": .845, "sorghum": .86, "soybean": .87, "winter_wheat": .865, "spring_wheat": .865,
              "durum": .865, "barley": .855, "oats": .86, "rye": .86, "sunflower": .9, "flaxseed": .91,
              "rice": .86, "alfalfa": .88, "hay": .88, "corn_silage": .35}


def _extension(key, row):
    ini, mid, end, p, ky, family, price, cost, n, hi, normal = row
    legume = CROPS[key]["legume"].value
    perennial = CROPS[key]["perennial"].value
    values = {
        "kc_initial": (ini, FAO56, "medium"),
        "n_fraction_at_flowering": (2 / 3 if key == "corn" else -1, BENDER_N if key == "corn"
                                    else "No crop-specific N uptake curve; potential-ET timing proxy", "low"),
        "small_grain": (float(key in {"winter_wheat", "spring_wheat", "barley", "oats", "rye", "durum"}),
                        "Cool-season small-grain crop functional type", "high"),
        "diverse_rotation_yield_gain": (.04 if key == "corn" else .09 if key == "soybean" else 0., DAVIS, "low"),
        "after_perennial_legume_n_kg_ha": (30 * 1.12085 if key == "corn" else 0., ALFALFA_N, "low"),
        "recommended_n_kg_ha": (181 * 1.12085 if key == "corn" else
                                (0 if legume else normal * n), N_RATES if key == "corn" else
                                "Reference crop removal rate; no local N recommendation", "low"),
        "previous_legume_credit_kg_ha": (57 * 1.12085 if key == "corn" else 0, N_RATES, "low"),
        "fertilizer_price_usd_kg": (.35 / .45359237, N_RATES + "; historical $0.35/lb N scenario", "low"),
        "kc_mid": (mid, FAO56, "medium"),
        "kc_end": (end, FAO56, "medium"),
        "depletion_fraction": (p, FAO22, "medium"),
        "bucket_root_depth_m": (ROOT_MAX.get(key, CROPS[key]["root_depth_m"].value),
                                FAO22 + " upper Zr, footnote 1 (rainfed modelling)" if key in ROOT_MAX
                                else "parent root_depth_m", "medium" if key in ROOT_MAX else "low"),
        "ky": (ky, FAO22, "medium" if key not in {"oats", "rye", "flaxseed", "barley"} else "low"),
        "ky_establishment": (.2 if key in {"corn", "corn_silage"} else ky * .5 / 3.8,
                             FAO33 + "; maize early vegetative split / other-crop seasonal allocation proxy", "low"),
        "ky_vegetative": (.4 if key in {"corn", "corn_silage"} else ky * .8 / 3.8,
                          "https://www.fao.org/4/i2800e/i2800e02.pdf; maize .4 / other crop allocation proxy", "medium" if key in {"corn", "corn_silage"} else "low"),
        "ky_flowering": (1.5 if key in {"corn", "corn_silage"} else ky * 1.5 / 3.8,
                         "https://www.fao.org/4/i2800e/i2800e02.pdf; maize 1.5 / other crop allocation proxy", "medium" if key in {"corn", "corn_silage"} else "low"),
        "ky_grain_fill": (.5 if key in {"corn", "corn_silage"} else ky / 3.8,
                          "https://www.fao.org/4/i2800e/i2800e02.pdf; maize .5 / other crop allocation proxy", "medium" if key in {"corn", "corn_silage"} else "low"),
        "family": (family, "USDA PLANTS taxonomy; numeric family identifier", "high"),
        "price_usd_t": (price, PRICES + "; rounded reference scenario, NOT a local quote", "low"),
        "cost_usd_ha": (cost, ERS + "; rounded operating-budget scenario, excludes land", "low"),
        "harvest_n_kg_t": (n, NUTRIENTS + "; representative removal coefficient", "low"),
        "harvest_index": (hi, "https://www.fao.org/4/i2800e/i2800e.pdf; representative AquaCrop HI", "low"),
        "normal_t_ha": (normal, "USDA NASS reference yield scenario; local normal overrides", "low"),
        "fixed_n_kg_ha": (200 * legume, SARE + "; representative legume range 100–300 kg N/ha", "low"),
        "residue_cn": (25 if legume else 60, SARE + "; legume 15–30, cereal 40–80", "low"),
        "shoot_root_ratio": (SHOOT_ROOT.get(key, 5.0), BOLINDER if key in SHOOT_ROOT
                             else BOLINDER + "; no crop-specific value, annual-crop mean of about 5",
                             "medium" if key in SHOOT_ROOT else "low"),
        "extra_root_fraction": (.65, BOLINDER, "low"),
        "dry_matter_fraction": (DRY_MATTER.get(key, .87),
                                "https://efotg.sc.egov.usda.gov/references/public/UT/AlfalfaIrrigationFacts2013Final.pdf; "
                                "12% moisture example used as an explicit hay carbon/reference-mass convention, NOT measured NASS moisture"
                                if key == "alfalfa" else MOISTURE if key in DRY_MATTER else MOISTURE + "; generic grain",
                                "low" if key == "alfalfa" else "medium" if key in DRY_MATTER else "low"),
        "cover_termination_gap_days": (14 if family == 1 else 3, NICHOLS if family == 1 else
                                       SARE + "; non-grass crops are often planted into living covers", "medium" if family == 1 else "low"),
        "water_productivity_g_m2": (33.7 if CROPS[key]["c4"].value >= .5 else 15., AQUACROP, "medium"),
        "cold_transpiration_gdd": (12., AQUACROP, "low"),
        "cover_establishment_gdd": (100., NICHOLS, "low"),
        "germination_min_c": (0., "Parent temperature-rule proxy; not measured soil temperature", "low"),
        "fixed_n_fraction": (0., SARE + "; set for legume cover records only", "medium"),
        "double_crop_factor": (0., DOUBLE + "; set for the double-crop record only", "medium"),
        "cover_months": (10 if perennial else 5, FAO56 + "; representative seasonal cover", "low"),
        "forage_value": (float(perennial), "harvestable forage trait, not a livestock ration", "medium"),
        "relative_water_demand": (mid, FAO56 + "; midseason Kc proxy", "medium"),
        "requires_flooding": (int(key == "rice"), FAO22 + "; paddy hydrology not supported by dryland bucket", "high"),
        "same_crop_penalty": (.043 if key == "corn" else (.103 if key == "soybean" else .03),
                              SEIFERT if key in {"corn", "soybean"} else SEIFERT + "; unvalidated transfer prior", "medium" if key in {"corn", "soybean"} else "low"),
        "penalty_years_to_plateau": (3 if key == "corn" else 5, SEIFERT + "; qualitative build-up approximation", "low"),
        "family_penalty": (.015, SEIFERT + "; unvalidated same-family transfer, not a measured magnitude", "low"),
        "cover_crop": (0, SARE, "high"),
        "fallow": (0, FAO56, "high"),
        "fertilizer_replacement": (0. if legume else 1., ("https://doi.org/10.2134/jeq1973.00472425000200020001x; "
                                   "Stanford 1973 mass-balance rate: N applied = N removed at the county normal yield "
                                   "(an accounting scenario, not a recommendation)"), "low"),
    }
    return {k: Trait(*v) for k, v in values.items()}


CATALOG = {key: Crop(key, load_traits(key) | _extension(key, row)) for key, row in TABLE.items()}
# ISU A1-20 (2025) machinery + seed/chemical/fertilizer/insurance + labor, land
# excluded, $/acre x 2.47105: corn after soybean 595.03, corn after corn 640.75,
# soybean after corn 382.89, established alfalfa 558.52.
ISU = "https://www.extension.iastate.edu/agdm/crops/pdf/a1-20-2025.pdf; Iowa average, excludes land"
for _key, _cost, _extra in (("corn", 595.03, 45.72), ("soybean", 382.89, 0), ("alfalfa", 558.52, 0)):
    CATALOG[_key].traits["cost_usd_ha"] = Trait(_cost * 2.47105, ISU, "medium")
    CATALOG[_key].traits["same_crop_extra_cost_usd_ha"] = Trait(
        _extra * 2.47105, ISU + "; corn-after-corn minus corn-after-soybean budget", "medium")
CATALOG["alfalfa"].traits["cost_usd_ha"] = Trait(
    558.52 * 2.47105, ISU + "; budget assumes 6 tons/acre (13.4 t/ha), well above the 8 t/ha reference "
    "yield used without a county alfalfa series, so alfalfa margins are pessimistic", "low")
for _crop in CATALOG.values():
    _crop.traits.setdefault("same_crop_extra_cost_usd_ha", Trait(
        0, ISU + "; no published same-crop budget difference for this crop", "low"))
# Winter cover records. Growth follows the parent's GDD phenology and FAO-56 Kc;
# biomass comes from simulated transpiration (AquaCrop WP*), so it depends on
# establishment date, temperature and soil water rather than a fixed tonnage.
COVER_RECORDS = {
    "cereal_rye_cover": ("rye", {
        "residue_cn": (30, SARE + "; rye C:N rises from about 25 (vegetative) to 40+ (flowering)", "low"),
        "cost_usd_ha": ((24.30 + 22 + 17.60) * 2.47105,
                        ("https://agorocarbonalliance.com/knowledgehub/real-life-economics-of-cover-crops-in-iowa/; "
                        "2025 Iowa example: 90 lb seed $24.30 + drill $22 + glyphosate/application $17.60 per acre"), "medium"),
        "seed_cost_usd_ha": (24.30 * 2.47105, "Agoro Iowa 2025, 90 lb rye at $0.27/lb", "medium"),
        "termination_cost_usd_ha": (17.60 * 2.47105, "Agoro Iowa 2025 glyphosate $10.60 + application $7/acre", "medium"),
        "minimum_frost_free_days": (0, RYE_THERMAL, "medium"),
        "cover_establishment_gdd": (0, RYE_THERMAL + "; no autumn GDD mortality cutoff", "medium"),
        "germination_min_c": ((34 - 32) * 5 / 9, RYE_THERMAL, "medium"),
        "base_c": ((38 - 32) * 5 / 9, RYE_THERMAL, "medium"),
        "winter_min_c": ((-30 - 32) * 5 / 9, ("https://www.uvm.edu/vtvegandberry/factsheets/winterrye.html; "
                         "established rye tolerates -30 F"), "medium"),
        "fixed_n_fraction": (0, SARE, "high"),
    }),
    "hairy_vetch_cover": ("rye", {
        "residue_cn": (12, SARE + "; hairy vetch C:N about 8-15", "low"),
        "cost_usd_ha": ((50 + 22 + 10) * 2.47105,
                        ("https://covercrop.org/cover_crops/hairy-vetch/; 2024 $2.50/lb x 20 lb/ac; drill $22/ac "
                        "Agoro 2025; termination $10/ac NRCS Missouri rye/vetch case study "
                        "https://efotg.sc.egov.usda.gov/references/public/MO/Seed_Production_CaseStudy3.pdf; "
                        "mixed-date transferred budget, local quote needed"), "low"),
        "seed_cost_usd_ha": (50 * 2.47105, "https://covercrop.org/cover_crops/hairy-vetch/; 2024 $2.50/lb x20", "medium"),
        "termination_cost_usd_ha": (10 * 2.47105, "NRCS Missouri Seed_Production_CaseStudy3.pdf; rye/vetch mixture", "low"),
        "minimum_frost_free_days": (30, ("https://www.sare.org/publications/managing-cover-crops-profitably/"
                                    "legume-cover-crops/hairy-vetch/; plant 30-45 days before killing frost"), "medium"),
        "winter_min_c": ((-20 - 32) * 5 / 9, ("https://www.nrcs.usda.gov/sites/default/files/2024-06/"
                         "Hairy%20Vetch%20Crop%20Fact%20Sheet.pdf; Iowa cold tolerance -20 F; air-temperature proxy"), "medium"),
        "fixed_n_fraction": (.65, PERRONE, "medium"),
        "legume": (1, SARE, "high"),
        "family": (2, "USDA PLANTS taxonomy; Fabaceae", "high"),
        "shoot_root_ratio": (5.2, BOLINDER + "; soybean value transferred to a legume cover", "low"),
        "cover_establishment_gdd": (100, NICHOLS + "; rye threshold transferred to vetch", "low"),
    }),
}
for _key, (_parent, _changes) in COVER_RECORDS.items():
    _t = CATALOG[_parent].traits.copy()
    for _name, _value in {
        "cover_crop": 1, "fixed_n_kg_ha": 0, "normal_t_ha": 0, "price_usd_t": 0, "harvest_index": 1,
        "fertilizer_replacement": 0, "forage_value": 0, "same_crop_penalty": 0, "family_penalty": 0,
        "same_crop_extra_cost_usd_ha": 0, "harvest_n_kg_t": 0,
    }.items():
        _t[_name] = Trait(_value, SARE + "; cover crop is terminated, not harvested", "high")
    _t["bucket_root_depth_m"] = Trait(1.0, FAO22 + "; shallow autumn-to-spring rooting of a terminated cover", "low")
    _t["cover_yield_effect"] = Trait(0., MARCILLO + "; " + CTIC + " (survey values not used: self-reported)", "low")
    for _name, (_value, _source, _confidence) in _changes.items():
        _t[_name] = Trait(_value, _source, _confidence)
    CATALOG[_key] = Crop(_key, _t)
_t = CATALOG["soybean"].traits.copy()
_t["double_crop_factor"] = Trait(.49, DOUBLE, "medium")
_t["cost_usd_ha"] = Trait(CATALOG["soybean"].v("cost_usd_ha"), ISU + "; full-season soybean budget, double-crop "
                          "seed and drying differ", "low")
_t["sow_after_harvest_days"] = Trait(3, DOUBLE + "; planted as soon as possible after wheat harvest", "low")
CATALOG["double_crop_soybean"] = Crop("double_crop_soybean", _t)
_t = CATALOG["hay"].traits.copy()
for _name, _value in {
    "fallow": 1, "price_usd_t": 0, "cost_usd_ha": 0, "fixed_n_kg_ha": 0,
    "fertilizer_replacement": 0, "normal_t_ha": 0,
    "forage_value": 0, "kc_initial": .2, "kc_mid": .2, "kc_end": .2,
}.items():
    _t[_name] = Trait(_value, FAO56 + "; bare-soil operational scenario", "low")
_t["bucket_root_depth_m"] = Trait(.15, "https://www.fao.org/4/x0490e/x0490e0c.htm; FAO-56 Ch.7 "
                                  "evaporation layer Ze 0.10-0.15 m", "medium")
CATALOG["fallow"] = Crop("fallow", _t)
