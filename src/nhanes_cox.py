"""Cox proportional-hazards estimation of exposure -> mortality HRs on NHANES.

Two families of exposure share one estimation routine:
  * diet (alcohol, total sugars, total/saturated fat) -- energy-adjusted;
  * environmental / psychosocial (PHQ-9 depression score; urinary TCPy, the
    chlorpyrifos pesticide metabolite) -- each from its own NHANES subsample.

For each we fit a cause-specific Cox model (default outcome: cancer death),
adjusting for the confounders the WCRF cohorts used -- age, sex, race/ethnicity,
education, smoking, BMI -- plus total energy for the diet exposures. HRs are
reported per a natural increment and per 1 SD of the (transformed) exposure.

A template, not a definitive estimate -- see the caveats in nhanes_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import nhanes_pipeline as N

CONFOUNDERS = ["age", "female", "bmi"]        # + energy_kcal for diet exposures
CATEGORICAL = ["race", "smoke", "educ"]


def _transform(df: pd.DataFrame, key: str, spec: dict) -> tuple[pd.Series, str]:
    """Apply the exposure's transform; return the modelled series and its label."""
    t = spec.get("transform", "none")
    if t == "energy_adj":
        return N.energy_adjust(df, key), f"{key}_adj"
    if t == "log_std":
        return df[key], key          # already log in the pipeline
    return df[key], key              # "none"


def _design(df: pd.DataFrame, expo_name: str, expo_vals: pd.Series,
            with_energy: bool) -> pd.DataFrame:
    out = pd.DataFrame({
        "time_years": df["time_years"].to_numpy(float),
        "_event": df["_event"].to_numpy(int),
        expo_name: expo_vals.to_numpy(float),
    }, index=df.index)
    covars = CONFOUNDERS + (["energy_kcal"] if with_energy else [])
    for c in covars:
        out[c] = df[c].to_numpy(float)
    for name in CATEGORICAL:
        d = pd.get_dummies(df[name].astype("category"), prefix=name, drop_first=True).astype(float)
        out = pd.concat([out, d.set_axis(out.index)], axis=1)
    return out


def fit_exposure(df: pd.DataFrame, key: str, spec: dict,
                 event_col: str = "event_cancer", weighted: bool = False) -> dict:
    """Fit one Cox model and return HR per-increment and per-SD."""
    work = df.copy()
    work["_event"] = work[event_col].to_numpy(int)
    vals, expo_name = _transform(work, key, spec)
    with_energy = spec.get("transform") == "energy_adj"

    design = _design(work, expo_name, vals, with_energy)
    kwargs = {}
    if weighted:
        design["mec_weight"] = df["mec_weight"].to_numpy(float)
        kwargs = dict(weights_col="mec_weight", robust=True)
    cph = CoxPHFitter()
    cph.fit(design, duration_col="time_years", event_col="_event",
            show_progress=False, **kwargs)

    beta, se = cph.params_[expo_name], cph.standard_errors_[expo_name]
    sd = float(np.nanstd(vals))
    per = spec["per"]

    def hr(mult):
        return (np.exp(beta * mult), np.exp((beta - 1.96 * se) * mult),
                np.exp((beta + 1.96 * se) * mult))

    hr_u, lo_u, hi_u = hr(per)
    hr_s, lo_s, hi_s = hr(sd)
    return {
        "exposure": spec["label"], "key": key,
        "outcome": "cancer death" if event_col == "event_cancer" else "all-cause death",
        "per_unit": f"per {per} {spec['unit']}",
        "HR_per_unit": hr_u, "lci_unit": lo_u, "uci_unit": hi_u,
        "HR_per_sd": hr_s, "lci_sd": lo_s, "uci_sd": hi_s,
        "p": cph.summary.loc[expo_name, "p"],
        "n": int(design.shape[0]), "events": int(design["_event"].sum()),
        "weighted": weighted,
    }


COLS = ["exposure", "outcome", "per_unit", "HR_per_unit", "lci_unit", "uci_unit",
        "HR_per_sd", "lci_sd", "uci_sd", "p", "n", "events", "weighted"]


def run_all(df: pd.DataFrame, weighted: bool = False,
            event_col: str = "event_cancer") -> pd.DataFrame:
    """Run every dietary exposure on the diet cohort `df`."""
    rows = [fit_exposure(df, k, s, event_col=event_col, weighted=weighted)
            for k, s in N.EXPOSURES.items()]
    return pd.DataFrame(rows)[COLS]


def run_extra(weighted: bool = False) -> pd.DataFrame:
    """Build each environmental/psychosocial cohort and fit its exposure, for
    both cancer-specific and all-cause mortality (the subsamples are smaller, so
    all-cause gives more events)."""
    builders = {"depression": N.build_depression_cohort,
                "pesticide": N.build_pesticide_cohort}
    cohorts = {name: b() for name, b in builders.items()}
    rows = []
    for key, spec in N.EXTRA_EXPOSURES.items():
        coh = cohorts[spec["cohort"]]
        for ev in ("event_cancer", "event_death"):
            rows.append(fit_exposure(coh, key, spec, event_col=ev, weighted=weighted))
    return pd.DataFrame(rows)[COLS]


if __name__ == "__main__":
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)
    coh = N.build_cohort()
    print("DIET exposures -> cancer mortality (unweighted, adjusted Cox):\n")
    print(run_all(coh).round(3).to_string(index=False))
    print("\nENVIRONMENTAL / PSYCHOSOCIAL exposures (own subsamples):\n")
    print(run_extra().round(3).to_string(index=False))
