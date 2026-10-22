"""Cox proportional-hazards estimation of diet -> cancer-mortality HRs on NHANES.

For each continuous exposure (alcohol, total sugars, total/saturated fat) we fit a
cause-specific Cox model for time-to-cancer-death, energy-adjusting the exposure
(Willett residual method) and adjusting for the usual confounders the WCRF cohorts
used -- age, sex, race/ethnicity, education, smoking, BMI -- plus total energy.

HRs are reported per a natural increment (e.g. per 10 g/day alcohol) and per 1 SD of
the energy-adjusted exposure, so the four diet factors are comparable on one scale.

This mirrors, on open data and for mortality, the kind of dose-response estimation
that produced the WCRF per-10g relative risks the sister notebook imports. It is a
template, not a definitive estimate -- see the caveats in nhanes_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import nhanes_pipeline as N

# Confounders. race/smoke are categorical -> one-hot (drop first level).
CONFOUNDERS = ["age", "female", "bmi", "energy_kcal"]
CATEGORICAL = {"race": "race", "smoke": "smoke", "educ": "educ"}


def _design(df: pd.DataFrame, exposure_adj: str) -> pd.DataFrame:
    """Assemble the Cox design matrix: duration, event, exposure, covariates."""
    out = pd.DataFrame({
        "time_years": df["time_years"].to_numpy(float),
        "event_cancer": df["event_cancer"].to_numpy(int),
        exposure_adj: df[exposure_adj].to_numpy(float),
    }, index=df.index)
    for c in CONFOUNDERS:
        out[c] = df[c].to_numpy(float)
    for name in CATEGORICAL:
        dummies = pd.get_dummies(df[name].astype("category"), prefix=name, drop_first=True).astype(float)
        out = pd.concat([out, dummies.set_axis(out.index)], axis=1)
    return out


def fit_exposure(df: pd.DataFrame, key: str, weighted: bool = False) -> dict:
    """Fit the Cox model for one exposure and return HR per-increment and per-SD."""
    spec = N.EXPOSURES[key]
    adj = f"{key}_adj"
    work = df.copy()
    work[adj] = N.energy_adjust(work, key)

    design = _design(work, adj)
    cph = CoxPHFitter(penalizer=0.0)
    kwargs = {}
    if weighted:
        design["mec_weight"] = df["mec_weight"].to_numpy(float)
        kwargs = dict(weights_col="mec_weight", robust=True)
    cph.fit(design, duration_col="time_years", event_col="event_cancer",
            show_progress=False, **kwargs)

    beta = cph.params_[adj]
    se = cph.standard_errors_[adj]
    sd = work[adj].std()
    per = spec["per"]

    def hr(mult):
        return np.exp(beta * mult), np.exp((beta - 1.96 * se) * mult), np.exp((beta + 1.96 * se) * mult)

    hr_u, lo_u, hi_u = hr(per)
    hr_s, lo_s, hi_s = hr(sd)
    return {
        "exposure": spec["label"],
        "key": key,
        "per_unit": f"per {per} {spec['unit']}",
        "HR_per_unit": hr_u, "lci_unit": lo_u, "uci_unit": hi_u,
        "HR_per_sd": hr_s, "lci_sd": lo_s, "uci_sd": hi_s,
        "p": cph.summary.loc[adj, "p"],
        "sd_exposure": sd,
        "n": int(design.shape[0]),
        "events": int(design["event_cancer"].sum()),
        "weighted": weighted,
    }


def run_all(df: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    rows = [fit_exposure(df, k, weighted=weighted) for k in N.EXPOSURES]
    cols = ["exposure", "per_unit", "HR_per_unit", "lci_unit", "uci_unit",
            "HR_per_sd", "lci_sd", "uci_sd", "p", "n", "events", "weighted"]
    return pd.DataFrame(rows)[cols]


if __name__ == "__main__":
    coh = N.build_cohort()
    print("Unweighted, confounder-adjusted cause-specific Cox (cancer mortality):\n")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(run_all(coh).round(3).to_string(index=False))
