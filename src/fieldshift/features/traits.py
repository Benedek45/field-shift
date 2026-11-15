"""Numeric crop records; identifiers are never model inputs."""

from dataclasses import dataclass

AQUA = "https://www.fao.org/fileadmin/user_upload/faowater/docs/AquaCropV40Annexes.pdf"
FAO56 = "https://www.fao.org/4/x0490e/x0490e0e.htm"
WHEAT = "https://www.fao.org/4/Y4011E/y4011e06.htm"
SOW = "https://www.fao.org/4/x0490e/x0490e0b.htm"
CORN_FROST = "https://extension.umn.edu/agriculture/crop-production/corn/early-fall-freeze-injury-in-corn"
SOY_FROST = "https://extension.umn.edu/agriculture/crop-production/soybean/spring-frost"


@dataclass(frozen=True)
class Trait:
    value: float
    source: str
    confidence: str


def _record(values):
    return {name: Trait(*entry) for name, entry in values.items()}


COMMON = {
    "vernalization_days": (0, WHEAT + "; zero for non-winter crops", "high"),
    "vernalization_min_c": (0, WHEAT, "medium"),
    "vernalization_max_c": (7, WHEAT, "medium"),
    "dormancy_c": (5, WHEAT + "; simplified operational cutoff", "low"),
    "pre_vernalization_gdd": (300, WHEAT + "; operational vegetative cap", "low"),
}

# Representative cultivars, not calibrated county-specific phenology.
CROPS = {
    "corn": _record(COMMON | {
        "base_c": (8, AQUA + " I.2", "high"),
        "upper_c": (30, AQUA + " I.2", "high"),
        "emergence_gdd": (80, AQUA + " I.2 midpoint of 60–100", "medium"),
        "flowering_gdd": (880, AQUA + " I.2 emergence + 800", "medium"),
        "flowering_duration_gdd": (180, AQUA + " I.2 range 150–200", "medium"),
        "maturity_gdd": (1700, AQUA + " I.2", "medium"),
        "heat_critical_c": (40, AQUA + " I.2 estimated pollination threshold", "medium"),
        "frost_c": (-2, CORN_FROST + "; rounded 28 F proxy; duration omitted", "medium"),
        "vegetative_frost_c": (-2, CORN_FROST + "; seedlings may survive protected growing points", "low"),
        "root_depth_m": (1.2, FAO56 + " table 22 maize 1.0–1.7 m", "medium"),
        "sowing_temp_c": (13, SOW + " table 11 footnote", "medium"),
        "autumn_sown": (0, SOW, "high"),
    }),
    "soybean": _record(COMMON | {
        "base_c": (5, AQUA + " I.6", "high"),
        "upper_c": (30, AQUA + " I.6", "high"),
        "emergence_gdd": (200, AQUA + " I.6 range 150–300", "medium"),
        "flowering_gdd": (1200, AQUA + " I.6 emergence + 1000", "medium"),
        "flowering_duration_gdd": (500, AQUA + " I.6 range 400–800", "medium"),
        "maturity_gdd": (2500, AQUA + " I.6 emergence + 2300", "medium"),
        "heat_critical_c": (40, AQUA + " I.6 estimated pollination threshold", "medium"),
        "frost_c": (-2, SOY_FROST + "; rounded 28–30 F proxy; duration omitted", "low"),
        "vegetative_frost_c": (-2, SOY_FROST + "; rounded 28–30 F proxy; duration omitted", "low"),
        "root_depth_m": (1.0, FAO56 + " table 22 soybean 0.6–1.3 m", "medium"),
        "sowing_temp_c": (15, SOW + "; conservative warm-soil proxy, not measured sowing", "low"),
        "autumn_sown": (0, SOW, "high"),
    }),
    "winter_wheat": _record(COMMON | {
        "base_c": (0, AQUA + " I.10; " + WHEAT, "high"),
        "upper_c": (26, AQUA + " I.10", "high"),
        "emergence_gdd": (150, AQUA + " I.10 range 100–250", "medium"),
        "flowering_gdd": (1300, AQUA + " I.10 emergence + 1150; winter extension assumed", "low"),
        "flowering_duration_gdd": (200, AQUA + " I.10 range 150–280", "medium"),
        "maturity_gdd": (2200, AQUA + " I.10 emergence + 2050", "low"),
        "heat_critical_c": (35, AQUA + " I.10 estimated pollination threshold", "medium"),
        "frost_c": (-2, WHEAT + "; heading sensitivity proxy, not canopy temperature", "low"),
        "vegetative_frost_c": (-20, WHEAT, "medium"),
        "root_depth_m": (1.5, FAO56 + " table 22 winter wheat 1.5–1.8 m", "medium"),
        "sowing_temp_c": (17, SOW + " table 11 footnote", "medium"),
        "autumn_sown": (1, WHEAT, "high"),
        "vernalization_days": (45, WHEAT + " midpoint of 30–60 days", "medium"),
    }),
}

OSPY = "https://github.com/aquacropos/aquacrop/blob/master/aquacrop/entities/crops/crop_params.py"
FAO66 = "https://www.fao.org/4/i2800e/i2800e.pdf"
# Extracted from shipped GDD defaults, not calendar-day defaults. Zero Flowering
# for roots is not interpreted as a real flowering period.
OSPY_GDD = {
    # base, upper, emergence, yield formation start, flowering length, maturity, root, heat, determinate
    "barley": (0, 15, 98, 867, 160, 1296, 1.3, 35, 1),
    "cotton": (12, 35, 12, 502, 709, 1956, 2, 43, 0),
    "dry_beans": (9, 30, 59, 556, 233, 1298, 1.7, 40, 0),
    "sorghum": (8, 30, 136, 1041, 306, 1760, 1.8, 40, 1),
    "rice": (8, 30, 50, 1150, 350, 1900, .5, 35, 1),
    "potatoes": (2, 26, 200, 550, 0, 1276, 1.5, -9, 0),
    "sunflower": (4, 30, 170, 1266, 350, 2400, 2, 40, 1),
    "sugarbeets": (5, 30, 23, 865, 0, 2203, 1, 40, 0),
    "spring_wheat": (0, 26, 150, 1250, 200, 2400, 1.5, 35, 1),
}
OSPY_NAMES = {
    "barley": "BarleyGDD", "cotton": "CottonGDD", "dry_beans": "DryBeanGDD",
    "sorghum": "SorghumGDD", "rice": "PaddyRiceGDD", "potatoes": "PotatoGDD",
    "sunflower": "SunflowerGDD", "sugarbeets": "SugarBeetGDD", "spring_wheat": "WheatGDD",
}
_PARAMETER_KEYS = (
    "base_c", "upper_c", "emergence_gdd", "flowering_gdd",
    "flowering_duration_gdd", "maturity_gdd", "root_depth_m",
    "heat_critical_c", "determinate",
)
for _crop, _values in OSPY_GDD.items():
    _record_values = {
        key: Trait(value, OSPY + f"; {OSPY_NAMES[_crop]} default", "medium")
        for key, value in zip(_PARAMETER_KEYS, _values)
    }
    CROPS[_crop] = CROPS["corn"].copy() | _record_values
    CROPS[_crop]["autumn_sown"] = Trait(0, FAO56, "high")
    CROPS[_crop]["sowing_temp_c"] = Trait(
        5 if _crop in {"barley", "spring_wheat"} else 12, SOW + "; air-temperature sowing proxy", "low"
    )
    CROPS[_crop]["frost_c"] = Trait(-2, FAO66 + "; generic reproductive frost proxy", "low")
    CROPS[_crop]["vegetative_frost_c"] = Trait(
        -8 if _crop in {"barley", "spring_wheat"} else -2,
        FAO66 + "; generic vegetative frost proxy", "low"
    )
    if _crop in {"potatoes", "sugarbeets"}:
        # Same tensor schema, but this interval represents tuber/root bulking,
        # explicitly signalled by reproductive_stage_is_flowering=0 below.
        CROPS[_crop]["flowering_duration_gdd"] = Trait(200, OSPY + "; operational bulking interval", "low")
    if _values[7] < 0:
        CROPS[_crop]["heat_critical_c"] = Trait(30, FAO66 + "; tuber heat-risk proxy", "low")

# Missing OSPy crop types use an explicitly low-confidence related-crop prior.
for _crop, _parent in {
    "oats": "barley", "durum": "spring_wheat", "rye": "winter_wheat",
    "flaxseed": "spring_wheat", "peanuts": "dry_beans",
    "alfalfa": "soybean", "hay": "barley",
}.items():
    CROPS[_crop] = {
        k: Trait(v.value, FAO66 + f"; {_parent} thermal proxy for {_crop}", "low")
        for k, v in CROPS[_parent].items()
    }
CROPS["peanuts"].update({
    "base_c": Trait(10, FAO66 + "; chosen warm-season legume prior, not extracted cultivar default", "low"),
    "maturity_gdd": Trait(2000, FAO66 + "; operational groundnut thermal-duration prior", "low"),
    "root_depth_m": Trait(.8, FAO56 + " table 22 groundnuts 0.5–1.0", "medium"),
})
CROPS["alfalfa"].update({
    "base_c": Trait(5, OSPY + "; AlfalfaGDD", "medium"),
    "upper_c": Trait(30, OSPY + "; AlfalfaGDD", "medium"),
    "root_depth_m": Trait(3, OSPY + "; AlfalfaGDD", "medium"),
})
CROPS["hay"]["base_c"] = Trait(4, FAO66 + "; mixed cool-season forage approximation", "low")
for _crop in ("alfalfa", "hay"):
    CROPS[_crop]["autumn_sown"] = Trait(0, FAO56, "high")
    CROPS[_crop]["sowing_temp_c"] = Trait(5, FAO56 + "; green-up threshold, NOT sowing", "low")
    CROPS[_crop]["frost_c"] = Trait(
        -4, "https://extension.umn.edu/prepare-your-alfalfa-for-winter"
        + "; rounded 24 F foliage-killing proxy; duration omitted; grass-hay transfer unverified", "low"
    )

_C4 = {"corn", "sorghum"}
_LEGUMES = {"soybean", "alfalfa", "dry_beans", "peanuts"}
_SHORT_DAY = {"soybean", "cotton", "rice", "sorghum"}
for _crop, _traits in CROPS.items():
    _root = _crop in {"potatoes", "sugarbeets"}
    _perennial = _crop in {"alfalfa", "hay"}
    _extras = {
        "c4": (int(_crop in _C4), "high"),
        "legume": (int(_crop in _LEGUMES), "high"),
        "harvest_whole_plant": (int(_perennial), "high"),
        "harvest_storage_root": (int(_root), "high"),
        "harvest_fiber": (int(_crop == "cotton"), "high"),
        "perennial": (int(_perennial), "high"),
        "photoperiod_short_day": (int(_crop in _SHORT_DAY), "medium"),
        "photoperiod_sensitivity": (.7 if _crop in _SHORT_DAY else .3, "low"),
        "determinate": (_traits.get("determinate", Trait(1, FAO66, "low")).value, "medium"),
        "reproductive_stage_is_flowering": (int(not _root and not _perennial), "high"),
        "cut_interval_gdd": (500 if _crop == "alfalfa" else (600 if _crop == "hay" else 0), "low"),
        "sowing_start_doy": (32 if _perennial else (244 if _traits["autumn_sown"].value else 60), "low"),
        "sowing_end_doy": (335 if _traits["autumn_sown"].value else 166, "low"),
        "season_end_month": (12 if _perennial else (8 if _traits["autumn_sown"].value else 10), "low"),
    }
    _traits.update({k: Trait(v, FAO66 + "; physical type / operational regional approximation", c)
                    for k, (v, c) in _extras.items()})
    if _perennial:
        for _key in ("emergence_gdd", "flowering_gdd", "flowering_duration_gdd", "maturity_gdd"):
            _traits[_key] = Trait(
                float("nan"), FAO66 + "; not applicable to established multi-cut season definition", "high"
            )
CROPS["hay"]["c4"] = Trait(.3, FAO66 + "; unknown mixed-hay C3/C4 composition proxy", "low")
CROPS["hay"]["legume"] = Trait(.1, FAO66 + "; unknown mixed-hay legume composition proxy", "low")
CROPS["hay"]["perennial"] = Trait(1, FAO66 + "; representative perennial mixture; label may include annual hay", "low")

VARIETIES = {
    "corn_silage": {
        "parent": "corn",
        "overrides": {
            "maturity_gdd": Trait(1400, OSPY + "; maize senescence, early whole-plant harvest proxy", "low"),
            "harvest_whole_plant": Trait(1, FAO66 + "; silage whole-plant harvest", "high"),
        },
    },
}
for _crop, _definition in VARIETIES.items():
    CROPS[_crop] = CROPS[_definition["parent"]].copy() | _definition["overrides"]


def load_traits(crop: str) -> dict[str, Trait]:
    if crop in VARIETIES:
        definition = VARIETIES[crop]
        return load_traits(definition["parent"]) | definition["overrides"]
    return CROPS[crop].copy()


def numeric_traits(traits) -> dict[str, float]:
    return {k: float(v.value if isinstance(v, Trait) else v) for k, v in traits.items()}
