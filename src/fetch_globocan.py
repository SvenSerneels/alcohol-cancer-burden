"""Fetch GLOBOCAN 2020 cancer incidence (IARC Global Cancer Observatory).

Reconstruction of Rumgay et al. 2021 (Lancet Oncol). The paper used GLOBOCAN 2020
incidence by country x sex x site. The live GCO "Today" v3 JSON API
(https://gco-api.iarc.fr/api/globocan/v3/2020/) still serves the 2020 database via
its `data/rate` endpoint, returning all-ages case counts and age-standardised rates
per cancer for a given country and sex.

Endpoint (reverse-engineered from the GCO front-end bundle):
    GET v3/2020/data/rate/{type}/{sex}/{population}/{cancer_csv}/
      type       0 = incidence
      sex        1 = male, 2 = female  (0 = both)
      population UN M49 numeric country code (e.g. France = 250)
      cancer_csv underscore-joined GCO cancer codes

We pull the cancer sites causally linked to alcohol in the paper, plus the
denominators and the sensitivity sites (stomach, pancreas):

    1  Lip, oral cavity        -> oral_cavity
    3  Oropharynx  } paper "pharyngeal cancer" C09-10/C12-13 = oropharynx+hypopharynx
    5  Hypopharynx }             (nasopharynx C11 is excluded)
    6  Oesophagus              -> oesophagus (SCC fraction applied downstream)
    7  Stomach                 -> stomach (sensitivity)
    8  Colon                   -> colon
    9  Rectum                  -> rectum
    11 Liver                   -> liver (HCC fraction applied downstream)
    13 Pancreas                -> pancreas (sensitivity)
    14 Larynx                  -> larynx
    20 Breast                  -> breast (female only)
    40 All cancers excl. NMSC  -> all_sites (overall PAF denominator)

Output: data/globocan2020_incidence.csv  (one row per country x sex x site).
Cached: re-running is a no-op unless --force is passed.
"""
from __future__ import annotations
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import country_converter as coco

API = "https://gco-api.iarc.fr/api/globocan/v3/2020"
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "globocan2020_incidence.csv"

# GCO cancer code -> our site label
CANCER_CODES = {
    1: "oral_cavity",
    3: "oropharynx",
    5: "hypopharynx",
    6: "oesophagus",
    7: "stomach",
    8: "colon",
    9: "rectum",
    11: "liver",
    13: "pancreas",
    14: "larynx",
    20: "breast",
    40: "all_sites",
}
CANCER_CSV = "_".join(str(c) for c in CANCER_CODES)

# GCO uses a few population codes that differ from UN M49 / ISO-numeric and that
# country_converter cannot resolve. Most important: GCO codes China as 160.
GCO_ISO3_OVERRIDE = {160: ("CHN", "China")}


def _get(url: str, retries: int = 4, pause: float = 0.4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "recon/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed {url}: {last}")


def country_list() -> list[int]:
    """UN numeric codes of the individual countries in GLOBOCAN 2020."""
    meta = _get(f"{API}/meta/predefined_populations/all/")
    for grp in meta:
        if grp.get("short_label", "").lower().startswith("countries"):
            return [c for c in grp["country_code"] if c < 900]  # <900 = real country
    raise RuntimeError("country group not found in meta")


def fetch(force: bool = False) -> Path:
    if OUT.exists() and not force:
        print(f"cache hit: {OUT} (use --force to refetch)")
        return OUT

    cc = coco.CountryConverter()
    codes = country_list()
    # Precompute UN-numeric -> ISO3 / name maps once (per-call coco.convert is slow).
    iso3_map = dict(zip(codes, cc.convert(codes, src="ISOnumeric", to="ISO3", not_found=None)))
    name_map = dict(zip(codes, cc.convert(codes, src="ISOnumeric", to="name_short", not_found=None)))
    print(f"GLOBOCAN 2020: fetching {len(codes)} countries x 2 sexes ...", flush=True)

    rows = []
    for i, code in enumerate(codes):
        if code in GCO_ISO3_OVERRIDE:
            iso3, name = GCO_ISO3_OVERRIDE[code]
        else:
            iso3 = iso3_map.get(code)
            name = name_map.get(code)
        for sex in (1, 2):
            data = _get(f"{API}/data/rate/0/{sex}/{code}/{CANCER_CSV}/")
            by_site = {}
            for rec in data.get("dataset", []):
                site = CANCER_CODES.get(rec.get("cancer_code"))
                if site:
                    by_site[site] = rec
            for site, rec in by_site.items():
                rows.append(
                    {
                        "country_code": code,
                        "iso3": iso3,
                        "country": name,
                        "sex": "male" if sex == 1 else "female",
                        "site": site,
                        "cases": rec.get("total", 0),
                        "asr": rec.get("asr", 0.0),
                        "crude_rate": rec.get("crude_rate", 0.0),
                        "total_pop": rec.get("total_pop", 0),
                    }
                )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(codes)} countries", flush=True)
        time.sleep(0.10)  # be polite to the API

    DATA.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["country_code", "iso3", "country", "sex", "site",
                        "cases", "asr", "crude_rate", "total_pop"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    return OUT


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
