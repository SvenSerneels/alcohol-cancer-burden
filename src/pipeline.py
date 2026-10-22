"""Reconstruction pipeline: PAF -> attributable cancer cases (Rumgay et al. 2021).

Combines the fetched GLOBOCAN 2020 incidence and WHO GISAH 2010 consumption with
the relative-risk parameters and the grid-based PAF model to reproduce the paper's
main results at country x sex x site granularity.

Modelling choices (documented as this reconstruction's simplifications):
  * All-ages adult granularity (no six-band age stratification): the public GHO
    API and the GLOBOCAN `data/rate` endpoint both expose adult-aggregate figures
    by sex. The 10-year latency is honoured by pairing 2010 exposure with 2020
    incidence at country level.
  * Histology: liver PAF uses the HCC dose-response applied to an HCC fraction of
    liver cancer; oesophagus uses the SCC dose-response applied to an SCC fraction.
    The paper used country-specific histology splits; we use documented global
    constants (HCC_FRACTION, SCC_FRACTION).
  * Breast: a single female RR blended from the pre-/post-menopausal estimates
    (age-specific menopausal status is unavailable at this granularity).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

import paf_model as m

DATA = Path(__file__).resolve().parent.parent / "data"

# ---- documented histology / breast constants -------------------------------
HCC_FRACTION = 0.80    # hepatocellular carcinoma as share of liver cancer (paper: ~80%)
SCC_FRACTION = 0.85    # oesophageal squamous-cell carcinoma share (global approx.)
BREAST_PRE_FRAC = 0.30  # rough share of breast cancer that is pre-menopausal
BREAST_POST_FRAC = 0.70

# ---- site -> incidence-source and RR spec ----------------------------------
# incidence_sites: which GLOBOCAN site column(s) supply the case denominator
# histology: multiplies incidence to get the at-risk histological subtype
SITE_SPEC = {
    "oral_cavity": dict(incidence=["oral_cavity"], rr="oral_cavity", histology=1.0, sex="both"),
    "pharynx":     dict(incidence=["oropharynx", "hypopharynx"], rr="pharynx", histology=1.0, sex="both"),
    "oesophagus":  dict(incidence=["oesophagus"], rr="__scc__", histology=SCC_FRACTION, sex="both"),
    "colon":       dict(incidence=["colon"], rr="colon", histology=1.0, sex="both"),
    "rectum":      dict(incidence=["rectum"], rr="rectum", histology=1.0, sex="both"),
    "liver":       dict(incidence=["liver"], rr="liver_hcc", histology=HCC_FRACTION, sex="both"),
    "larynx":      dict(incidence=["larynx"], rr="larynx", histology=1.0, sex="both"),
    "breast":      dict(incidence=["breast"], rr="__breast__", histology=1.0, sex="female"),
}
CAUSAL_SITES = list(SITE_SPEC)


# ---------------------------------------------------------------- load inputs
def load_inputs():
    rr_lin = pd.read_csv(DATA / "rr_linear.csv").set_index("site")
    scc = pd.read_csv(DATA / "rr_oesophagus_scc.csv").set_index("param")["value"]
    gisah = pd.read_csv(DATA / "gisah_consumption_2010.csv")
    glob = pd.read_csv(DATA / "globocan2020_incidence.csv")
    region = pd.read_csv(DATA / "country_region_hdi.csv")
    return rr_lin, scc, gisah, glob, region


def _breast_rr10(rr_lin) -> float:
    """Blend pre/post-menopausal betas by rough case shares -> one female RR/10g."""
    b = (BREAST_PRE_FRAC * np.log(rr_lin.loc["breast_premeno", "rr_per_10g"])
         + BREAST_POST_FRAC * np.log(rr_lin.loc["breast_postmeno", "rr_per_10g"]))
    return float(np.exp(b))


def _rr_grid(site: str, rr_lin, scc, x):
    """RR evaluated on the integration grid for a given modelling site."""
    spec = SITE_SPEC[site]
    if spec["rr"] == "__scc__":
        return m.rr_scc(x, float(scc["beta1"]), float(scc["beta2"]))
    if spec["rr"] == "__breast__":
        return m.rr_linear(x, _breast_rr10(rr_lin), 0.0)
    row = rr_lin.loc[spec["rr"]]
    return m.rr_linear(x, float(row["rr_per_10g"]), float(row["dose_threshold_gday"]))


def _rr_draw(site: str, rr_lin, scc) -> m.RRDraw:
    spec = SITE_SPEC[site]
    if spec["rr"] == "__scc__":
        cov = ((float(scc["var_b1"]), float(scc["cov_b1_b2"])),
               (float(scc["cov_b1_b2"]), float(scc["var_b2"])))
        return m.RRDraw(kind="scc", b1=float(scc["beta1"]), b2=float(scc["beta2"]), cov=cov)
    if spec["rr"] == "__breast__":
        v = float(rr_lin.loc["breast_postmeno", "var_logrr"])
        return m.RRDraw(kind="linear", rr_per_10g=_breast_rr10(rr_lin), var_logrr=v)
    row = rr_lin.loc[spec["rr"]]
    return m.RRDraw(kind="linear", rr_per_10g=float(row["rr_per_10g"]),
                    var_logrr=float(row["var_logrr"]),
                    threshold=float(row["dose_threshold_gday"]))


# --------------------------------------------------- core reconstruction
def build(gisah, glob, region, rr_lin, scc, step=0.1):
    """Return a tidy DataFrame: country x sex x site with PAF and attributable cases."""
    x = m.make_grid(step)

    # incidence wide table: index (country_code, sex), columns = GLOBOCAN sites
    inc = glob.pivot_table(index=["country_code", "iso3", "country", "sex"],
                           columns="site", values="cases", aggfunc="sum").reset_index()
    asr = glob.pivot_table(index=["country_code", "sex"],
                           columns="site", values="asr", aggfunc="sum")

    g = gisah[["iso3", "sex", "p_current_drinker", "mu_drinkers_gday", "sigma_gday"]]
    df = inc.merge(g, on=["iso3", "sex"], how="left")
    df = df.merge(region[["iso3", "region"]], on="iso3", how="left")
    df = df[df["mu_drinkers_gday"].notna() & df["region"].notna()].reset_index(drop=True)

    # gamma pdf matrix once per (country, sex) row -> reused across sites & MC draws
    pdf = m.gamma_pdf_matrix(df["mu_drinkers_gday"].to_numpy(),
                             df["sigma_gday"].to_numpy(), x)
    p_cd = df["p_current_drinker"].to_numpy()

    records = []
    for site, spec in SITE_SPEC.items():
        rr_grid = _rr_grid(site, rr_lin, scc, x)
        paf_curve = m.paf_from_grid(p_cd, pdf, rr_grid, x)          # PAF on the histological subtype
        eff_paf = paf_curve * spec["histology"]                    # effective PAF on all cases of the site
        # incidence of this site = sum of its GLOBOCAN source columns
        site_inc = np.zeros(len(df))
        for col in spec["incidence"]:
            if col in df:
                site_inc = site_inc + df[col].fillna(0).to_numpy()
        att = eff_paf * site_inc
        cat = {c: m.paf_grid_between(p_cd, pdf, rr_grid, x, lo, hi) * spec["histology"] * site_inc
               for c, (lo, hi) in m.CAT_BOUNDS.items()}
        for i in range(len(df)):
            if spec["sex"] == "female" and df.at[i, "sex"] != "female":
                continue
            records.append({
                "country_code": df.at[i, "country_code"], "iso3": df.at[i, "iso3"],
                "country": df.at[i, "country"], "region": df.at[i, "region"],
                "sex": df.at[i, "sex"], "site": site,
                "incidence": site_inc[i], "paf": eff_paf[i], "att_cases": att[i],
                "moderate": cat["moderate"][i], "risky": cat["risky"][i], "heavy": cat["heavy"][i],
            })
    out = pd.DataFrame.from_records(records)
    # attach all-sites incidence denominator per country x sex
    allsites = glob[glob.site == "all_sites"][["country_code", "sex", "cases"]] \
        .rename(columns={"cases": "all_sites_incidence"})
    out = out.merge(allsites, on=["country_code", "sex"], how="left")
    return out, df, pdf, x


# ------------------------------------------------------------- aggregations
def global_by_site(recon):
    g = recon.groupby(["site", "sex"]).agg(
        att_cases=("att_cases", "sum"), incidence=("incidence", "sum")).reset_index()
    piv = g.pivot(index="site", columns="sex", values="att_cases").fillna(0)
    inc = g.pivot(index="site", columns="sex", values="incidence").fillna(0)
    piv["total"] = piv.sum(axis=1)
    inc["total"] = inc.sum(axis=1)
    piv["paf_pct"] = 100 * piv["total"] / inc["total"].replace(0, np.nan)
    return piv, inc


def overall(recon):
    """Global attributable cases, total incidence and overall PAF, by sex + total."""
    att = recon.groupby("sex")["att_cases"].sum()
    # all-sites incidence per country x sex, de-duplicated
    denom = recon.drop_duplicates(["country_code", "sex"]).groupby("sex")["all_sites_incidence"].sum()
    res = {}
    for sex in ["male", "female"]:
        res[sex] = dict(att=att.get(sex, 0), inc=denom.get(sex, 0),
                        paf=100 * att.get(sex, 0) / denom.get(sex, 1))
    tot_att, tot_inc = att.sum(), denom.sum()
    res["total"] = dict(att=tot_att, inc=tot_inc, paf=100 * tot_att / tot_inc)
    return res


def by_region(recon):
    g = recon.groupby(["region", "sex"]).agg(att=("att_cases", "sum")).reset_index()
    denom = recon.drop_duplicates(["country_code", "sex"]).groupby(["region", "sex"])[
        "all_sites_incidence"].sum().reset_index()
    g = g.merge(denom, on=["region", "sex"], how="left")
    g["paf_pct"] = 100 * g["att"] / g["all_sites_incidence"]
    tot = recon.groupby("region")["att"] if False else None  # noqa
    piv_att = g.pivot(index="region", columns="sex", values="att").fillna(0)
    piv_paf = g.pivot(index="region", columns="sex", values="paf_pct").fillna(0)
    denom_r = denom.pivot(index="region", columns="sex", values="all_sites_incidence").fillna(0)
    piv_att["total"] = piv_att.sum(axis=1)
    piv_paf["total"] = 100 * piv_att["total"] / denom_r.sum(axis=1).replace(0, np.nan)
    return piv_att, piv_paf


def drinking_categories(recon):
    tot = recon[["moderate", "risky", "heavy"]].sum()
    total = tot.sum()
    return pd.DataFrame({
        "cases": tot,
        "pct": 100 * tot / total,
    })


def increments(recon_inputs, rr_lin, scc, step=0.1, band=10.0):
    """Attributable cases by 10 g/day band x sex, summed over sites/countries."""
    recon, df, pdf, x = recon_inputs
    p_cd = df["p_current_drinker"].to_numpy()
    edges = [m.LO] + list(np.arange(band, m.HI + band, band))
    bands = list(zip(edges[:-1], [min(e, m.HI) for e in edges[1:]]))
    # build per (country,sex) incidence per site quickly from recon
    result = {b: {"male": 0.0, "female": 0.0} for b in bands}
    # map row index in df to (country_code, sex)
    keyidx = {(r.country_code, r.sex): i for i, r in df.iterrows()}
    for site, spec in SITE_SPEC.items():
        rr_grid = _rr_grid(site, rr_lin, scc, x)
        # incidence vector aligned to df rows
        site_inc = np.zeros(len(df))
        for col in spec["incidence"]:
            if col in df:
                site_inc += df[col].fillna(0).to_numpy()
        for lo, hi in bands:
            paf_b = m.paf_grid_between(p_cd, pdf, rr_grid, x, lo, hi) * spec["histology"]
            att_b = paf_b * site_inc
            for i in range(len(df)):
                sex = df.at[i, "sex"]
                if spec["sex"] == "female" and sex != "female":
                    continue
                result[(lo, hi)][sex] += att_b[i]
    rows = []
    for (lo, hi) in bands:
        rows.append({"band_low": lo, "band_high": hi,
                     "male": result[(lo, hi)]["male"],
                     "female": result[(lo, hi)]["female"],
                     "total": result[(lo, hi)]["male"] + result[(lo, hi)]["female"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- Monte Carlo UI
def overall_uncertainty(recon_inputs, rr_lin, scc, n=1000, seed=1, rel_sd=0.10):
    """95% UI for global attributable cases & PAF via parameter Monte Carlo.

    Draws RR parameters (per site, from published (co)variances) and a per-draw
    consumption-mean jitter; recomputes the global total using the precomputed
    gamma-pdf matrix so each draw is a handful of matrix ops."""
    recon, df, pdf, x = recon_inputs
    rng = np.random.default_rng(seed)
    p_cd = df["p_current_drinker"].to_numpy()
    mu = df["mu_drinkers_gday"].to_numpy()
    sig = df["sigma_gday"].to_numpy()
    female = (df["sex"] == "female").to_numpy()

    inc_by_site = {}
    for site, spec in SITE_SPEC.items():
        v = np.zeros(len(df))
        for col in spec["incidence"]:
            if col in df:
                v += df[col].fillna(0).to_numpy()
        if spec["sex"] == "female":
            v = np.where(female, v, 0.0)
        inc_by_site[site] = v
    denom = recon.drop_duplicates(["country_code", "sex"])["all_sites_incidence"].sum()
    draws_att = np.empty(n)
    for d in range(n):
        mu_d = np.maximum(rng.normal(mu, rel_sd * mu), 1e-6)
        sig_d = sig / mu * mu_d
        pdf_d = m.gamma_pdf_matrix(mu_d, sig_d, x)
        total = 0.0
        for site, spec in SITE_SPEC.items():
            rr_grid = m.rr_grid_from_draw(_rr_draw(site, rr_lin, scc), x, rng)
            paf = m.paf_from_grid(p_cd, pdf_d, rr_grid, x) * spec["histology"]
            total += float(np.sum(paf * inc_by_site[site]))
        draws_att[d] = total
    lo, hi = np.percentile(draws_att, [2.5, 97.5])
    paf_draws = 100 * draws_att / denom
    return dict(att_lo=lo, att_hi=hi,
                paf_lo=float(np.percentile(paf_draws, 2.5)),
                paf_hi=float(np.percentile(paf_draws, 97.5)))
