"""Build an NHANES diet -> cancer-mortality analytic cohort (fully public data).

A companion to the alcohol/GLOBOCAN reconstruction. Where that notebook *imports*
WCRF relative risks, this pipeline shows how one would *estimate* diet-exposure
hazard ratios directly from an open prospective resource.

Data (no application, no login required):
  * NHANES continuous survey files (demographics, day-1 dietary recall, smoking,
    body measures), served as SAS-transport .xpt at
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{startyear}/DataFiles/{FILE}.xpt
  * NHANES Public-Use Linked Mortality Files (follow-up through 31 Dec 2019),
    fixed-width .dat at .../datalinkage/linked_mortality/NHANES_{s}_{e}_MORT_2019_PUBLIC.dat

Design: NHANES is cross-sectional at interview, but each participant is linked
forward to the National Death Index, which turns it into a retrospective cohort
for *mortality* outcomes. We model the cause-specific hazard of cancer death
(UCOD_LEADING == 2, malignant neoplasms) as a function of usual diet.

Honest caveats (see also the notebook):
  * Outcome is cancer MORTALITY, not incidence (WCRF/Rumgay use incidence).
  * Diet is a single 24-hour recall -> large within-person measurement error,
    which biases diet-mortality associations toward the null (regression dilution).
  * Cox fits here are unweighted by default; NHANES' complex survey design
    (WTMEC weights, SDMVPSU/SDMVSTRA) is handled only approximately in the
    weighted sensitivity fit. Population inference needs full Taylor linearization.
"""
from __future__ import annotations
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "nhanes"

XPT_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
MORT_BASE = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
             "linked_mortality/NHANES_{s}_{e}_MORT_2019_PUBLIC.dat")

# NHANES continuous cycles with both a day-1 dietary file and 2019 mortality linkage.
# (name_suffix maps the file stems, e.g. DEMO_C for 2003-2004.)
CYCLES = {
    "2003-2004": dict(year=2003, suffix="C", s=2003, e=2004),
    "2005-2006": dict(year=2005, suffix="D", s=2005, e=2006),
    "2007-2008": dict(year=2007, suffix="E", s=2007, e=2008),
    "2009-2010": dict(year=2009, suffix="F", s=2009, e=2010),
    "2011-2012": dict(year=2011, suffix="G", s=2011, e=2012),
    "2013-2014": dict(year=2013, suffix="H", s=2013, e=2014),
}
DEFAULT_CYCLES = list(CYCLES)

# Public-use Linked Mortality File fixed-width layout (0-indexed half-open spans),
# per the NCHS SAS/R/Stata input programs. Validated against the 2005-2006 file.
MORT_COLSPECS = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
MORT_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
              "DIABETES", "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]
UCOD_CANCER = 2  # malignant neoplasms in the "leading causes" recode

# Columns pulled from each NHANES component file.
DEMO_COLS = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
             "INDFMPIR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
DIET_COLS = ["SEQN", "DR1DRSTZ", "DR1TKCAL", "DR1TALCO",
             "DR1TSUGR", "DR1TTFAT", "DR1TSFAT"]
SMQ_COLS = ["SEQN", "SMQ020", "SMQ040"]
BMX_COLS = ["SEQN", "BMXBMI"]

# Dietary exposures (energy-adjusted downstream via the Willett residual method).
EXPOSURES = {
    "alcohol_g":  dict(col="DR1TALCO", label="Alcohol",       per=10,  unit="g/day", transform="energy_adj"),
    "sugar_g":    dict(col="DR1TSUGR", label="Total sugars",  per=50,  unit="g/day", transform="energy_adj"),
    "totfat_g":   dict(col="DR1TTFAT", label="Total fat",     per=25,  unit="g/day", transform="energy_adj"),
    "satfat_g":   dict(col="DR1TSFAT", label="Saturated fat", per=10,  unit="g/day", transform="energy_adj"),
}

# Environmental / psychosocial exposures, each from its own NHANES component and
# subsample (so they get their own analytic cohort, not merged into the diet one).
#   depression : PHQ-9 score (0-27), DPQ_* questionnaire, cycles 2005-2014
#   pesticide  : TCPy (urinary 3,5,6-trichloro-2-pyridinol, the chlorpyrifos
#                organophosphate metabolite), creatinine-corrected + log, UPHOPM_*
#                environmental subsample, cycles 2007-2014
EXTRA_EXPOSURES = {
    "phq9":      dict(label="Depression (PHQ-9)", per=5, unit="points",
                      transform="none", cohort="depression"),
    "tcpy_log":  dict(label="Pesticide (TCPy)",   per=1, unit="log µg/g creat (per SD)",
                      transform="log_std", cohort="pesticide"),
}
DEPRESSION_CYCLES = ["2005-2006", "2007-2008", "2009-2010", "2011-2012", "2013-2014"]
PESTICIDE_CYCLES = ["2007-2008", "2009-2010", "2011-2012", "2013-2014"]
DPQ_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]  # 9 PHQ-9 items (DPQ100 excluded)


# ------------------------------------------------------------------ downloads
def _download(url: str, dest: Path) -> Path:
    """Cache a file locally; re-running is a no-op once present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=240) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _xpt(cycle: str, stem: str) -> pd.DataFrame:
    c = CYCLES[cycle]
    name = f"{stem}_{c['suffix']}"
    path = _download(XPT_BASE.format(year=c["year"], name=name), CACHE / f"{name}.xpt")
    return pd.read_sas(path)


def _mortality(cycle: str) -> pd.DataFrame:
    c = CYCLES[cycle]
    fname = f"NHANES_{c['s']}_{c['e']}_MORT_2019_PUBLIC.dat"
    path = _download(MORT_BASE.format(s=c["s"], e=c["e"]), CACHE / fname)
    m = pd.read_fwf(path, colspecs=MORT_COLSPECS, names=MORT_NAMES, na_values=["."])
    return m


# --------------------------------------------------------------- cohort build
BASE_COVARS = ["age", "female", "race", "educ", "bmi", "smoke"]


def _norm0(s: pd.Series) -> pd.Series:
    """pandas read_sas encodes SAS zero as ~5.4e-79; snap those back to 0."""
    return s.mask(s.abs() < 1e-8, 0.0)


def _subset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out["SEQN"] = out["SEQN"].astype("int64")
    return out


def _smoking_status(row) -> float:
    """0 never, 1 former, 2 current (NaN if unknown)."""
    ever = row.get("SMQ020")
    now = row.get("SMQ040")
    if ever == 2:
        return 0.0
    if ever == 1:
        if now in (1.0, 2.0):
            return 2.0
        if now == 3.0:
            return 1.0
    return np.nan


def _base_frame(cycle: str) -> pd.DataFrame:
    """Demographics + smoking + body measures + mortality, with the shared
    outcomes and confounders. The common spine under every analytic cohort."""
    demo = _subset(_xpt(cycle, "DEMO"), DEMO_COLS)
    smq = _subset(_xpt(cycle, "SMQ"), SMQ_COLS)
    bmx = _subset(_xpt(cycle, "BMX"), BMX_COLS)
    mort = _mortality(cycle)
    mort["SEQN"] = mort["SEQN"].astype("int64")

    df = (demo.merge(smq, on="SEQN", how="left")
              .merge(bmx, on="SEQN", how="left")
              .merge(mort, on="SEQN", how="inner"))
    df["cycle"] = cycle
    df["time_years"] = df["PERMTH_EXM"] / 12.0
    df["event_cancer"] = ((df["MORTSTAT"] == 1) & (df["UCOD_LEADING"] == UCOD_CANCER)).astype(int)
    df["event_death"] = (df["MORTSTAT"] == 1).astype(int)
    df["age"] = df["RIDAGEYR"]
    df["female"] = (df["RIAGENDR"] == 2).astype(int)
    df["race"] = df["RIDRETH1"].astype("Int64")            # 1..5 categorical
    df["educ"] = df["DMDEDUC2"].where(df["DMDEDUC2"] < 7)   # 7/9 = refused/dk
    df["bmi"] = df["BMXBMI"]
    df["smoke"] = df.apply(_smoking_status, axis=1)
    return df


def _finalize(df: pd.DataFrame, cycles: list[str], exposures: list[str],
              extra_covars: list[str], weight_src: str, min_age: int) -> pd.DataFrame:
    """Shared tail: pool weight, apply eligibility + completeness filters."""
    df["n_cycles"] = len(cycles)
    df["mec_weight"] = df[weight_src] / df["n_cycles"]   # pooled survey weight
    need = BASE_COVARS + extra_covars + exposures
    keep = (
        (df["ELIGSTAT"] == 1)
        & (df["age"] >= min_age)
        & df["time_years"].gt(0)
        & df[need].notna().all(axis=1)
    )
    return df.loc[keep].reset_index(drop=True)


def build_cohort(cycles: list[str] | None = None, min_age: int = 20,
                 kcal_range: tuple[int, int] = (500, 5000)) -> pd.DataFrame:
    """Diet cohort: mortality-eligible adults with a reliable day-1 recall,
    plausible energy intake and complete confounders."""
    cycles = cycles or DEFAULT_CYCLES
    parts = []
    for c in cycles:
        df = _base_frame(c)
        diet = _subset(_xpt(c, "DR1TOT"), DIET_COLS)
        df = df.merge(diet, on="SEQN", how="inner")
        df["energy_kcal"] = df["DR1TKCAL"]
        for key, spec in EXPOSURES.items():
            df[key] = df[spec["col"]]
        df = df[(df["DR1DRSTZ"] == 1) & df["energy_kcal"].between(*kcal_range)]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return _finalize(df, cycles, list(EXPOSURES), ["energy_kcal"], "WTMEC2YR", min_age)


def build_depression_cohort(cycles: list[str] | None = None, min_age: int = 20) -> pd.DataFrame:
    """PHQ-9 depression cohort (DPQ_* questionnaire, cycles 2005-2014)."""
    cycles = cycles or DEPRESSION_CYCLES
    parts = []
    for c in cycles:
        df = _base_frame(c)
        dpq = _subset(_xpt(c, "DPQ"), ["SEQN"] + DPQ_ITEMS)
        for it in DPQ_ITEMS:
            dpq[it] = _norm0(dpq[it]).where(dpq[it] < 4)   # 0-3 valid; 7/9 -> NaN
        dpq["phq9"] = dpq[DPQ_ITEMS].sum(axis=1, min_count=len(DPQ_ITEMS))
        parts.append(df.merge(dpq[["SEQN", "phq9"]], on="SEQN", how="inner"))
    df = pd.concat(parts, ignore_index=True)
    return _finalize(df, cycles, ["phq9"], [], "WTMEC2YR", min_age)


def build_pesticide_cohort(cycles: list[str] | None = None, min_age: int = 20) -> pd.DataFrame:
    """Pesticide (TCPy) cohort: chlorpyrifos metabolite, creatinine-corrected and
    log-transformed, from the UPHOPM environmental subsample (cycles 2007-2014)."""
    cycles = cycles or PESTICIDE_CYCLES
    parts = []
    for c in cycles:
        df = _base_frame(c)
        up = _subset(_xpt(c, "UPHOPM"), ["SEQN", "URXTCC", "WTSC2YR"])
        cr = _subset(_xpt(c, "ALB_CR"), ["SEQN", "URXUCR"])
        up = up.merge(cr, on="SEQN", how="left")
        tcpy = _norm0(up["URXTCC"])                        # ng/mL
        creat = up["URXUCR"] / 100.0                       # mg/dL -> g/L (≈ g/L creatinine)
        # µg per gram creatinine; log for the right-skewed concentration
        up["tcpy_log"] = np.log((tcpy / creat).where(creat > 0))
        parts.append(df.merge(up[["SEQN", "tcpy_log", "WTSC2YR"]], on="SEQN", how="inner"))
    df = pd.concat(parts, ignore_index=True)
    df = df[np.isfinite(df["tcpy_log"])]
    return _finalize(df, cycles, ["tcpy_log"], [], "WTSC2YR", min_age)


# ------------------------------------------------------ energy adjustment
def energy_adjust(df: pd.DataFrame, nutrient: str, energy: str = "energy_kcal") -> pd.Series:
    """Willett residual method: nutrient regressed on energy; residual + overall
    mean gives an energy-adjusted intake uncorrelated with total energy."""
    x = df[energy].to_numpy(float)
    y = df[nutrient].to_numpy(float)
    b1, b0 = np.polyfit(x, y, 1)
    predicted = b0 + b1 * x
    return pd.Series(y - predicted + y.mean(), index=df.index)


if __name__ == "__main__":
    for name, builder in [("diet", build_cohort),
                          ("depression (PHQ-9)", build_depression_cohort),
                          ("pesticide (TCPy)", build_pesticide_cohort)]:
        coh = builder()
        print(f"{name:20s} n = {len(coh):6,}  cancer deaths = {coh.event_cancer.sum():4,}  "
              f"all-cause deaths = {coh.event_death.sum():5,}")
