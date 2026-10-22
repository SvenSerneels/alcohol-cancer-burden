"""Fetch WHO GISAH 2010 alcohol-consumption data via the GHO OData API.

Reconstruction of Rumgay et al. 2021 (Lancet Oncol). The paper drew 2010 adult
per-capita alcohol consumption and drinking-status prevalence from the WHO Global
Information System on Alcohol and Health (GISAH), disaggregated by country, sex and
six age bands, assuming a 10-year latency to 2020 cancer incidence.

The public GHO OData API (https://ghoapi.azureedge.net/api/) serves the same GISAH
indicators, sex-disaggregated (SEX_MLE / SEX_FMLE / SEX_BTSX), at the adult (15+)
aggregate level -- but NOT by the six age bands. We therefore reconstruct at the
country x sex (all-adult) level, consistent with the all-ages GLOBOCAN incidence we
use downstream. This is the study's main documented simplification.

Indicators used (year 2010, nearest available otherwise):
    SA_0000001403  Total per-capita (15+) consumption = recorded 3yr avg + unrecorded
                   (litres pure alcohol / year), by sex           -> mean exposure
    SA_0000001413  Consumers past 12 months (%), by sex           -> current-drinker prevalence
    SA_0000001409  Lifetime abstainers (%), by sex                -> (context)
    SA_0000001414  Former drinkers (%), by sex                    -> former-drinker prevalence (sensitivity)

Derivation, per country x sex:
    per_capita_gday = APC_litres_yr * 789 g/L / 365 d * 0.8      (0.8 = wastage/tourist correction)
    mu_drinkers_gday = per_capita_gday / P_current_drinker       (mean intake among drinkers)
    sigma = (1.171 + 0.087*is_female) * mu_drinkers_gday         (Kehoe et al. gamma SD relation)

Output: data/gisah_consumption_2010.csv (country x sex).
Countries missing consumption/prevalence are imputed from their region mean
(matching the paper's treatment of 10 missing-data countries).
"""
from __future__ import annotations
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import country_converter as coco

GHO = "https://ghoapi.azureedge.net/api"
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "gisah_consumption_2010.csv"

ETHANOL_G_PER_L = 789.0          # density of pure ethanol
CORRECTION = 0.8                 # per-capita wastage/tourist correction (paper)
TARGET_YEAR = 2010

SEX_MAP = {"SEX_MLE": "male", "SEX_FMLE": "female"}


def _get(indicator: str) -> list[dict]:
    url = f"{GHO}/{indicator}"
    req = urllib.request.Request(url, headers={"User-Agent": "recon/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["value"]


def _nearest_year(records: list[dict], iso3: str, sexcode: str) -> float | None:
    """Value for iso3/sex closest to TARGET_YEAR."""
    cand = [
        x for x in records
        if x.get("SpatialDim") == iso3 and x.get("Dim1") == sexcode
        and x.get("NumericValue") is not None
    ]
    if not cand:
        return None
    best = min(cand, key=lambda x: abs((x.get("TimeDim") or 0) - TARGET_YEAR))
    return float(best["NumericValue"])


def fetch(force: bool = False) -> Path:
    if OUT.exists() and not force:
        print(f"cache hit: {OUT} (use --force to refetch)")
        return OUT

    print("GHO/GISAH: downloading indicators ...")
    apc = _get("SA_0000001403")       # total per-capita consumption
    pcd = _get("SA_0000001413")       # consumers past 12 months (%)
    ab = _get("SA_0000001409")        # lifetime abstainers (%)
    former = _get("SA_0000001414")    # former drinkers (%)

    # universe of ISO3 codes appearing in the consumption indicator
    iso3s = sorted({x["SpatialDim"] for x in apc
                    if isinstance(x.get("SpatialDim"), str) and len(x["SpatialDim"]) == 3})

    cc = coco.CountryConverter()
    rows = []
    for iso3 in iso3s:
        name = cc.convert(iso3, src="ISO3", to="name_short", not_found=None)
        if name is None:
            continue
        for sexcode, sex in SEX_MAP.items():
            is_female = 1 if sex == "female" else 0
            apc_lyr = _nearest_year(apc, iso3, sexcode)
            p_drink = _nearest_year(pcd, iso3, sexcode)
            p_abst = _nearest_year(ab, iso3, sexcode)
            p_former = _nearest_year(former, iso3, sexcode)

            per_capita_gday = None
            mu = None
            if apc_lyr is not None:
                per_capita_gday = apc_lyr * ETHANOL_G_PER_L / 365.0 * CORRECTION
            if per_capita_gday is not None and p_drink and p_drink > 0:
                mu = per_capita_gday / (p_drink / 100.0)
            sigma = None
            if mu is not None:
                sigma = (1.171 + 0.087 * is_female) * mu

            rows.append({
                "iso3": iso3,
                "country": name,
                "sex": sex,
                "apc_litres_year": apc_lyr,
                "per_capita_gday": per_capita_gday,
                "p_current_drinker": (p_drink / 100.0) if p_drink is not None else None,
                "p_lifetime_abstainer": (p_abst / 100.0) if p_abst is not None else None,
                "p_former_drinker": (p_former / 100.0) if p_former is not None else None,
                "mu_drinkers_gday": mu,
                "sigma_gday": sigma,
                "imputed": 0,
            })

    _impute_missing(rows)

    DATA.mkdir(exist_ok=True)
    fields = ["iso3", "country", "sex", "apc_litres_year", "per_capita_gday",
              "p_current_drinker", "p_lifetime_abstainer", "p_former_drinker",
              "mu_drinkers_gday", "sigma_gday", "imputed"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["mu_drinkers_gday"] is not None)
    print(f"wrote {len(rows)} rows ({n_ok} with usable mu) -> {OUT}")
    return OUT


def _impute_missing(rows: list[dict]) -> None:
    """Fill missing mu/sigma/prevalence from the region x sex mean."""
    region_csv = DATA / "country_region_hdi.csv"
    iso2region = {}
    if region_csv.exists():
        with region_csv.open() as f:
            for r in csv.DictReader(f):
                iso2region[r["iso3"]] = r["region"]

    # accumulate region x sex means for the derivable fields
    acc: dict[tuple, list] = {}
    for r in rows:
        reg = iso2region.get(r["iso3"])
        if reg is None or r["mu_drinkers_gday"] is None:
            continue
        key = (reg, r["sex"])
        acc.setdefault(key, {"mu": [], "sig": [], "pcd": [], "pf": []})
        acc[key]["mu"].append(r["mu_drinkers_gday"])
        acc[key]["sig"].append(r["sigma_gday"])
        if r["p_current_drinker"] is not None:
            acc[key]["pcd"].append(r["p_current_drinker"])
        if r["p_former_drinker"] is not None:
            acc[key]["pf"].append(r["p_former_drinker"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    for r in rows:
        if r["mu_drinkers_gday"] is not None:
            continue
        reg = iso2region.get(r["iso3"])
        stats = acc.get((reg, r["sex"]))
        if not stats:
            continue
        r["mu_drinkers_gday"] = mean(stats["mu"])
        r["sigma_gday"] = mean(stats["sig"])
        if r["p_current_drinker"] is None:
            r["p_current_drinker"] = mean(stats["pcd"])
        if r["p_former_drinker"] is None:
            r["p_former_drinker"] = mean(stats["pf"])
        r["imputed"] = 1


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
