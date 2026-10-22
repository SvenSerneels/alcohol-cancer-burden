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
WAVEVARS = ["iwstat", "agey_e", "cancre", "cancr", "cesd", "bmi", "smokev", "smoken", "wtresp"]

# Exposure available directly in the RAND file (perceived stress needs the LB file).
EXPOSURES = {
    "cesd": dict(col="cesd", label="Depression (CES-D)", per=1, unit="points", transform="none"),
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


def build_cohort(min_age: int = 50) -> pd.DataFrame:
    """One survival record per respondent, ready for nhanes_cox.fit_exposure."""
    long = _to_long(_read_columns())

    # eligibility at a wave: interviewed, complete covariates, aged >= min_age
    long["smoke"] = long.apply(_smoke, axis=1)
    need = ["agey_e", "cesd", "bmi", "smoke", "raeduc"]

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
        rows.append(dict(
            hhidpn=pid, time_years=time, event_cancer=int(event),
            event_death=int(pd.notna(base["radyear"]) and base["radyear"] > 0),
            age=float(entry_age), female=int(base["ragender"] == 2),
            race=race, educ=float(base["raeduc"]), bmi=float(base["bmi"]),
            smoke=float(base["smoke"]), cesd=float(base["cesd"]),
            mec_weight=float(base["wtresp"]) if pd.notna(base["wtresp"]) else np.nan,
            entry_wave=int(entry_wave),
        ))
    coh = pd.DataFrame(rows)
    coh["race"] = coh["race"].astype("Int64")
    return coh


if __name__ == "__main__":
    import pandas as pd
    import nhanes_cox as C

    coh = build_cohort()
    print(f"HRS incident-cancer cohort: n = {len(coh):,}")
    print(f"  incident cancers = {coh.event_cancer.sum():,}")
    print(f"  median follow-up = {coh.time_years.median():.1f} years, "
          f"total person-years = {coh.time_years.sum():,.0f}")
    print(f"  mean entry age   = {coh.age.mean():.1f}\n")

    rows = [C.fit_exposure(coh, "cesd", EXPOSURES["cesd"], event_col=ev)
            for ev in ("event_cancer", "event_death")]
    res = pd.DataFrame(rows)[C.COLS]
    res["outcome"] = res["outcome"].replace({"cancer death": "incident cancer"})
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)
    print("CES-D depression (reusing nhanes_cox, confounder-adjusted Cox):\n")
    print(res.round(3).to_string(index=False))
