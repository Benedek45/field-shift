"""Explicit, cited alfalfa stand scenarios, not fitted ranking penalties."""
import torch

UC = "https://coststudyfiles.ucdavis.edu/uploads/cs_public/1c/e2/1ce256d0-957e-4bd4-b17e-18fef4efcedd/16alfalfasjv300acfinal_41916.pdf"
TX = "https://agecoext.tamu.edu/wp-content/uploads/2026/02/2026D3AlfalfaIrrigatedEstablish.pdf"
NE = "https://cap.unl.edu/sites/unl.edu.ianr.agecon.center-for-ag-profitability/files/media/file/2026-nebraska-crop-budgets.pdf"
ISU = "https://www.extension.iastate.edu/agdm/crops/pdf/a1-20-2025.pdf"
AUTO = "https://extension.umn.edu/agriculture/crop-production/forages/reducing-autotoxicity-in-alfalfa"
N_CREDIT = "https://extension.umn.edu/agriculture/crop-production/corn/tools-to-predict-nitrogen-needs-in-corn-following-alfalfa"
YOUNG_N = "https://blog-crop-news.extension.umn.edu/2022/04/alfalfa-and-red-clover-can-supply-n-for.html"
WATER = "https://efotg.sc.egov.usda.gov/references/public/UT/AlfalfaIrrigationFacts2013Final.pdf"
# lb N/acre in the chosen SOURCE budgets. NE/CA/KS N is embedded in
# phosphate: do not purchase/bill that nitrogen again.
PRODUCTION_N = {"48":30., "31":75*.11, "06":200*.11, "20":100*.11}
STARTER_N = {"48":40., "31":100*.11, "06":200*.11}
POLICY = (
    "LOW management scenario: a uniform alfalfa enterprise has one fall-seeding/no-hay year "
    "and three production years (UC2016 pp3,8: three productive years). Terminate by the regional "
    "fall-seeding date in the last production year; reseed on that date next year, a full year later "
    "(UMN autotoxicity). Date: September15 TX/CA/MS; August15 elsewhere (UMN southern-MN window; LOW transfer). "
    "Thus A×4 is one stand lifecycle, NOT four full harvest years. Dates are modelling "
    "scenarios, not measured operations. Nonuniform stands establish "
    "in-year: TX/CA fall, elsewhere spring; no reseeding during production. "
    "No winterkill, density deterioration or individual cuttings are modelled. "
    + UC + "; " + AUTO
)


def schedule(traj, crops, states, device):
    """Per-trajectory/year stand events. All bulk arrays stay on the accelerator."""
    seq = torch.as_tensor(traj["crop"], device=device)
    site = torch.as_tensor(traj["site"], device=device)
    is_alf = torch.tensor([c.key == "alfalfa" for c in crops], device=device)
    current = is_alf[seq[:, 1:-1]]
    previous = is_alf[seq[:, :-2]]
    uniform = torch.as_tensor(traj["uniform"], device=device)[:, None] & current
    time = torch.arange(traj["horizon"], device=device)[None]
    # All four phases remain paired with the same weather windows. Short uniform
    # requests ALSO use four calendar years, never an annual reseeding fiction.
    phase = (torch.as_tensor(traj["year0"], device=device)[:, None] + time
             + torch.as_tensor(traj["position"], device=device)[:, :1]) % 4
    establish = current & (~previous | (uniform & (phase == 0)))
    terminate = uniform & (phase == 3)
    regional_fall = torch.tensor([s in ("48", "06") for s in states], device=device)[site, None]
    fall = regional_fall | uniform
    spring_fraction = torch.tensor([3.8/6.7 if s == "31" else 2.5/6 for s in states],
                                   device=device)[site, None]
    fall_day = torch.tensor([257 if s in ("48","06","28") else 226 for s in states], device=device)[site, None]
    harvest_fraction = torch.where(establish, torch.where(fall, 0., spring_fraction), 1.)
    return {"alfalfa": current, "establish": establish, "terminate": terminate,
            "fall": fall, "harvest_fraction": harvest_fraction, "fall_day":fall_day.expand_as(current)}


def establishment(state, fall, irrigated):
    """Cash operating and ownership lines, excluding separately priced water.

    Values are USD/acre, hours/acre and source starter-N lb/acre. Unsplit
    establishment cash lines remain an explicitly cited bundle.
    """
    if state == "06":
        return {"operating":667-87, "fixed":108, "irrigation_fixed":0, "labour":50, "wage":20.44, "seed":221,
                    "fertilizer_other":74, "chemicals":7, "fuel_repairs":48,
                    "nitrogen":22, "n_price":0, "source":UC+" Table1 pp10–11; operating667 minus water87; "
                    "ownership527 minus land488 plus cash overhead221 minus property tax152; "
                    "starter N included in phosphate blend, not billed again; nominal2016"}
    if state == "48":
        return {"operating":386.53-40.50-26.80, "fixed":139.92-90, "irrigation_fixed":9+18.38,
                    "labour":17.73, "wage":15, "seed":178, "nitrogen":40, "n_price":.67/.45359237,
                    "fertilizer_other":48.8+15.6+3.6, "chemicals":7+6.88,
                    "fuel_repairs":5.59+9+3.9+10+4.44+4.6,
                    "source":TX+" p1: variable386.53 less energy40.50 and N26.80; "
                    "fixed139.92 less land90; D3→D2 transfer LOW"}
    if state == "31":
        if fall:
            return {"operating":192.90, "fixed":87.06+(61.61 if irrigated else 0),
                        "irrigation_fixed":61.61 if irrigated else 0, "labour":14.97, "wage":27,
                        "seed":75, "fertilizer_other":46, "chemicals":24.58, "fuel_repairs":12.13+12.8,
                        "nitrogen":11, "n_price":0, "source":NE+" pp16–18 budget1 fall/dryland; "
                        "irrigated scenario retains budget10 system ownership61.61; "
                        "LOW irrigation transfer, actual water priced separately; hours use2025 wage27"}
        return {"operating":302.05-12.67, "fixed":627.47-122.22-261.90, "irrigation_fixed":43.09+24.89,
                    "labour":32.22, "wage":27, "seed":75, "nitrogen":11, "n_price":0,
                    "fertilizer_other":46, "chemicals":22.14, "fuel_repairs":35.94-12.67+45.63,
                    "source":NE+" pp31–33 budget6; operating302.05 minus pumping12.67; "
                    "ownership627.47 minus land tax122.22/opportunity261.90; hours use2025 wage27"}
    # ISU seeding is an ACTUAL cash event, not its annual one-third allocation.
    # Labour is operating even though printed in the fixed column (ISU p1).
    return {"operating":131.60+20.15+89.70+(0 if fall else 38.94+60.45),
                "fixed":35.30+(0 if fall else 67.38), "irrigation_fixed":0,
                "labour":20.15+(0 if fall else 60.45), "wage":20.15, "seed":62.70,
                "fertilizer_other":22+89.7, "chemicals":26.3, "fuel_repairs":20.6+(0 if fall else 38.94),
                "nitrogen":0, "n_price":0, "source":ISU+" p9 herbicide-assisted spring seeding: "
                "full cash establishment131.60+labour20.15; annual fertilizer89.70, "
                "harvest38.94+labour60.45; ownership35.30+67.38; LOW outside Iowa"}
