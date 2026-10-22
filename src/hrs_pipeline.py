"""Build an HRS incident-cancer survival cohort from the RAND HRS Longitudinal File.

This is the incidence upgrade the NHANES companion pointed to. The Health and
Retirement Study follows US adults aged 50+ biennially; the RAND longitudinal file
harmonises every wave. We reshape it to one survival record per respondent:

    entry   = first interviewed wave with no prior cancer and complete covariates
    event   = first later wave reporting a *new* cancer diagnosis (incidence)
    time    = age at exit - age at entry   (years; interview-interval censored)

so `nhanes_cox.fit_exposure` runs against it almost unchanged. The headline
exposure available in the RAND file itself is the **CES-D depression score** (the
HRS analogue of the PHQ-9 we used on NHANES); the perceived-stress scale lives in
the separate Psychosocial "Leave-Behind" file and can be merged on `hhidpn` later.

DATA IS NOT REDISTRIBUTABLE. The .dta lives in data/hrs/ (gitignored); this module
reads it locally and never writes derived microdata back to the repo.

Honest caveats:
  * Cancer is SELF-REPORTED (first report of ever-having-cancer), not registry-
    confirmed -> report/recall error, and non-melanoma skin cancers are mixed in.
  * Population is 50+; this is older-adult incidence, not lifetime.
  * 2-year interview spacing = interval-censored timing (we approximate with age).
  * Baseline (fixed) exposures; CES-D especially is prone to reverse causation.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
HRS_DTA = DATA / "hrs" / "randhrs1992_2022v1.dta"

WAVES = range(1, 17)                      # 1992 (w1) .. 2022 (w16)
FIXED = ["hhidpn", "ragender", "raracem", "rahispan", "raeduc", "rabyear", "radyear"]
WAVEVARS = ["iwstat", "agey_e", "cancre", "cancr", "cesd", "bmi", "smokev", "smoken",
            "wtresp", "drink", "drinkd", "drinkn"]
GRAMS_PER_DRINK = 14.0                     # US standard drink ≈ 14 g ethanol

# Exposures available directly in the RAND file. Diet nutrients (fat, sugar, ...)
# are NOT in HRS core -- they need the 2013 Health Care & Nutrition Study FFQ.
#   cesd      : CES-D depression score (0-8), waves 2-16
#   alcohol_g : usual ethanol g/day = days/week x drinks/day x 14 / 7, waves 3-16
EXPOSURES = {
    "cesd":      dict(label="Depression (CES-D)", per=1,  unit="points", transform="none"),
    "alcohol_g": dict(label="Alcohol",            per=10, unit="g/day",  transform="none"),
}


def _read_columns() -> pd.DataFrame:
    """Read only the needed columns from the 1.7 GB .dta (numeric codes).

    Some wave-vars don't exist in every wave (e.g. CES-D starts at wave 2), so we
    intersect the wish-list with the file's actual columns.
    """
    with pd.io.stata.StataReader(HRS_DTA) as rdr:
        available = set(rdr.read(nrows=1).columns)
    wish = FIXED + [f"r{w}{v}" for w in WAVES for v in WAVEVARS]
    cols = [c for c in wish if c in available]
    return pd.read_stata(HRS_DTA, columns=cols, convert_categoricals=False)


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide (one row/person) -> long (one row/person-wave)."""
    parts = []
    for w in WAVES:
        sub = df[FIXED].copy()
        sub["wave"] = w
        for v in WAVEVARS:
            col = f"r{w}{v}"
            sub[v] = df[col] if col in df.columns else np.nan
        parts.append(sub)
    long = pd.concat(parts, ignore_index=True)
    return long[long["iwstat"] == 1]          # keep completed respondent interviews


def _smoke(row) -> float:
    if row["smokev"] == 0:
        return 0.0                              # never
    if row["smokev"] == 1:
        return 2.0 if row["smoken"] == 1 else 1.0   # current / former
    return np.nan


def _alcohol_gday(long: pd.DataFrame) -> pd.Series:
    """Usual ethanol g/day from frequency x quantity; non-drinkers -> 0."""
    dr, dd, dn = long["drink"], long["drinkd"], long["drinkn"]
    g = (dd * dn / 7.0) * GRAMS_PER_DRINK
    g = g.where(dd.notna() & dn.notna())        # drinker with missing qty -> NaN
    g = g.mask(dr == 0, 0.0)                     # explicit non-drinker -> 0
    g = g.mask(dr.isna(), np.nan)
    return g


def build_cohort(exposures=("cesd", "alcohol_g"), min_age: int = 50) -> pd.DataFrame:
    """One survival record per respondent, ready for nhanes_cox.fit_exposure.

    `exposures` lists which exposure columns must be present at the entry wave
    (they are captured at baseline). Pass a single-item tuple to maximise the
    cohort for one exposure (e.g. alcohol needs waves >=3, CES-D >=2)."""
    long = _to_long(_read_columns())
    long["smoke"] = long.apply(_smoke, axis=1)
    long["alcohol_g"] = _alcohol_gday(long)

    # eligibility at a wave: interviewed, aged >= min_age, complete covariates
    # AND complete for every requested exposure (captured at baseline).
    need = ["agey_e", "bmi", "smoke", "raeduc"] + list(exposures)

    rows = []
    for pid, g in long.sort_values("wave").groupby("hhidpn", sort=False):
        g = g.reset_index(drop=True)
        elig = g[g[need].notna().all(axis=1) & (g["agey_e"] >= min_age)]
        # entry = first eligible wave with no cancer yet
        entry_cands = elig[elig["cancre"] == 0]
        if entry_cands.empty:
            continue
        base = entry_cands.iloc[0]
        entry_wave, entry_age = base["wave"], base["agey_e"]

        fu = g[g["wave"] >= entry_wave]
        # event = first wave at/after entry reporting ever-cancer == 1
        ev = fu[fu["cancre"] == 1]
        if not ev.empty:
            exit_age, event = ev.iloc[0]["agey_e"], 1
        else:
            exit_age, event = fu.iloc[-1]["agey_e"], 0
            # if died with no cancer report, censor at approx age of death
            if pd.notna(base["radyear"]) and base["radyear"] > 0:
                exit_age = max(exit_age, base["radyear"] - base["rabyear"])

        time = float(exit_age) - float(entry_age)
        if not np.isfinite(time) or time <= 0:
            continue

        # race: 1 NH-white, 2 NH-black, 3 Hispanic, 4 other
        race = (3 if base["rahispan"] == 1 else
                {1: 1, 2: 2, 3: 4}.get(base["raracem"], 4))
        rec = dict(
            hhidpn=pid, time_years=time, event_cancer=int(event),
            event_death=int(pd.notna(base["radyear"]) and base["radyear"] > 0),
            age=float(entry_age), female=int(base["ragender"] == 2),
            race=race, educ=float(base["raeduc"]), bmi=float(base["bmi"]),
            smoke=float(base["smoke"]),
            mec_weight=float(base["wtresp"]) if pd.notna(base["wtresp"]) else np.nan,
            entry_wave=int(entry_wave),
        )
        for k in EXPOSURES:
            rec[k] = float(base[k]) if pd.notna(base[k]) else np.nan
        rows.append(rec)
    coh = pd.DataFrame(rows)
    coh["race"] = coh["race"].astype("Int64")
    return coh


if __name__ == "__main__":
    import pandas as pd
    import nhanes_cox as C

    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)
    rows = []
    for key, spec in EXPOSURES.items():
        coh = build_cohort(exposures=(key,))          # own cohort per exposure
        print(f"{spec['label']:20s}: n = {len(coh):,}  incident cancers = "
              f"{coh.event_cancer.sum():,}  person-years = {coh.time_years.sum():,.0f}")
        for ev in ("event_cancer", "event_death"):
            rows.append(C.fit_exposure(coh, key, spec, event_col=ev))
    res = pd.DataFrame(rows)[C.COLS]
    res["outcome"] = res["outcome"].replace({"cancer death": "incident cancer"})
    print("\nHRS exposures -> incident cancer / all-cause death (adjusted Cox):\n")
    print(res.round(3).to_string(index=False))
