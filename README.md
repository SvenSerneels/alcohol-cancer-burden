# Reconstructing the global alcohol-attributable cancer burden (2020)

A from-scratch reconstruction of:

> Rumgay H, Shield K, Charvat H, Ferrari P, Sornpaisarn B, Obot I, Islami F, Lemmens VEPP,
> Rehm J, Soerjomataram I. **Global burden of cancer in 2020 attributable to alcohol
> consumption: a population-based study.** *Lancet Oncology* 2021; 22(8):1071–80.
> [doi:10.1016/S1470-2045(21)00279-5](https://doi.org/10.1016/S1470-2045(21)00279-5)

The pipeline fetches the **raw data the paper is based on**, re-implements the paper's
Levin-based population-attributable-fraction (PAF) model, and reproduces its main
conclusions — validating against the paper's own published tables.

## Headline result (reconstructed vs. paper)

| Quantity | Reconstructed | Paper |
|---|---|---|
| Total alcohol-attributable cancer cases, 2020 | **697,766** | 741,300 |
| Share of all cancers (PAF) | **3.9%** | 4.1% |
| Male share of the burden | **75.7%** | 76.7% |
| Oesophagus / Liver / Breast cases | 181,592 / 138,389 / 93,800 | 189,700 / 154,700 / 98,300 |
| Moderate-drinking (<20 g/day) cases | 97,619 (14.0%) | 103,100 (13.9%) |
| + stomach & pancreas (sensitivity) | 796,237 | 808,700 |
| Regional PAF correlation (17 regions) | **r = 0.95** | — |

Every qualitative conclusion of the paper is reproduced.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy matplotlib requests country_converter jupyter nbconvert ipykernel

# fetch raw data (writes to data/; cached, so this is a one-off)
.venv/bin/python src/fetch_gho.py         # WHO GISAH 2010 alcohol consumption
.venv/bin/python src/fetch_globocan.py    # GLOBOCAN 2020 cancer incidence (~7 min; API-latency bound)

# build + run the notebook
.venv/bin/python src/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace alcohol_cancer_burden_2020.ipynb
```

Then open `alcohol_cancer_burden_2020.ipynb`. It runs offline once the two CSVs are cached.

## Layout

```
alcohol_cancer_burden_2020.ipynb   the narrative reconstruction (start here)
src/
  fetch_gho.py        WHO GHO/GISAH OData API  -> data/gisah_consumption_2010.csv
  fetch_globocan.py   IARC GCO v3 API (2020)   -> data/globocan2020_incidence.csv
  paf_model.py        RR curves, gamma model, grid-based PAF integrals, Monte Carlo
  pipeline.py         combines everything -> country x sex x site PAFs & cases; aggregations
  build_notebook.py   assembles the notebook (kept for reproducibility)
data/
  rr_linear.csv, rr_oesophagus_scc.csv, rr_former.csv   RR parameters (appendix Table 2)
  country_region_hdi.csv                                 185 countries -> 17 UN regions
  gisah_consumption_2010.csv, globocan2020_incidence.csv (fetched)
  target_*.csv                                           paper's tables, for validation
```

## Data sources

- **Cancer incidence** — IARC Global Cancer Observatory, GLOBOCAN 2020, via the live
  `gco-api.iarc.fr/api/globocan/v3/2020/data/rate/` endpoint (all-ages counts + ASR by
  country × sex × site).
- **Alcohol consumption** — WHO Global Information System on Alcohol and Health (GISAH),
  2010, via the GHO OData API (`ghoapi.azureedge.net`): per-capita consumption, drinker
  prevalence, abstainers and former drinkers, by country × sex.
- **Relative risks** — WCRF Continuous Update Project dose–response estimates, transcribed
  from the paper's appendix Table 2.

## Method

For each country × sex we model daily intake among drinkers as a gamma distribution
(mean μ = per-capita consumption ÷ drinker prevalence; SD σ = (1.171 + 0.087·sex)·μ),
integrate the site-specific relative-risk curve over it to get a Levin-based PAF, and
multiply by GLOBOCAN incidence. Integration is on a fine fixed grid (0.1–150 g/day),
which makes the whole pipeline and the 1000-draw Monte-Carlo uncertainty intervals fast.

## Documented simplifications (why numbers differ slightly from the paper)

1. **All-ages adult granularity.** The public APIs expose adult-aggregate figures by sex,
   not the paper's six age bands. We work at country × sex level; the 10-year latency is
   honoured by pairing 2010 exposure with 2020 incidence.
2. **Global histology fractions.** Liver uses the HCC dose–response on an 80% HCC fraction;
   oesophagus uses the SCC dose–response on an ~85% SCC fraction. The paper used
   country-specific histology splits — this is the main reason our *eastern-Asia* and
   *oesophageal* estimates run a little low (China's true SCC fraction is ~90%).
3. **Breast** uses one female RR blended from the pre-/post-menopausal estimates.
4. The 0.8 exposure-correction factor, integration limits, and PAF formula are exactly as
   published.

## Companion: estimating diet–cancer HRs ourselves (NHANES)

Where the reconstruction *imports* the WCRF relative risks, the companion notebook
**estimates diet–cancer hazard ratios from scratch** on fully public data — showing the
machinery behind a number like "RR 1.15 per 10 g/day".

```bash
.venv/bin/pip install lifelines
.venv/bin/python src/build_nhanes_notebook.py   # downloads NHANES (cached to data/nhanes/) + builds
.venv/bin/jupyter nbconvert --to notebook --execute --inplace nhanes_diet_cancer_mortality.ipynb
```

- **Data (no login/application):** NHANES continuous 2003–2014 (demographics, day-1 dietary
  recall, smoking, body measures) + NHANES **Public-Use Linked Mortality Files** (NDI linkage
  through 2019). ~29k adults, ~1,000 cancer deaths.
- **Method:** cause-specific Cox PH for cancer mortality vs alcohol, total sugars, total fat,
  saturated fat — energy-adjusted (Willett residual method), confounder-adjusted (age, sex,
  race/ethnicity, education, smoking, BMI, energy), with a survey-weighted sensitivity fit.
- **Point of the exercise:** the method *and its limits* — a single 24-hour recall plus a
  mortality (not incidence) outcome attenuates associations toward the null, which is exactly
  why alcohol looks weak here despite WCRF's clear incidence signal.

```
nhanes_diet_cancer_mortality.ipynb   companion notebook
src/
  nhanes_pipeline.py   download + merge + cohort build + Willett energy adjustment
  nhanes_cox.py        cause-specific Cox estimation (per-unit & per-SD HRs)
  build_nhanes_notebook.py
```

*Independent reconstruction for educational purposes. Not affiliated with IARC, WHO, or the
study authors.*
