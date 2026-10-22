"""Population Attributable Fraction (PAF) model for alcohol-attributable cancer.

Re-implements the Levin-based PAF methodology of Rumgay et al. 2021 (Lancet Oncol),
using the relative-risk dose-response functions and the gamma consumption model
documented in the paper's appendix.

Core formula (current drinking), theoretical-minimum-risk exposure = lifetime abstention:

              integral_{0.1..150} P_CD * P(x) * [RR(x) - 1] dx
    PAF = ----------------------------------------------------------
          integral_{0.1..150} P_CD * P(x) * [RR(x) - 1] dx  +  1

where
    P_CD  = prevalence of current drinking (fraction of the population),
    P(x)  = gamma pdf of daily intake x (g/day) among current drinkers,
    RR(x) = relative risk at intake x for the cancer site,
    integration limits 0.1 -> 150 g/day.

A category PAF (moderate 0.1-20, risky 20-60, heavy >60 g/day) uses the same
denominator but integrates the numerator only over the category's [lo, hi].

RR forms:
    log-linear sites:  RR(x) = exp(beta * x / 10),  beta = ln(RR_per_10g),
                       optionally with a dose threshold below which RR = 1
                       (colon/rectum: 20 g/day; sensitivity sites: 45 g/day).
    oesophageal SCC:   RR(x) = exp(b1*x + b2*x*ln(x)).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.stats import gamma as gamma_dist

LO, HI = 0.1, 150.0            # integration limits (g/day)
CAT_BOUNDS = {                 # drinking categories (g/day)
    "moderate": (0.1, 20.0),
    "risky": (20.0, 60.0),
    "heavy": (60.0, 150.0),
}


# --------------------------------------------------------------------------- RR
def rr_linear(x, rr_per_10g: float, threshold: float = 0.0):
    """Log-linear RR per 10 g/day, flat at 1 below `threshold`."""
    x = np.asarray(x, dtype=float)
    beta = math.log(rr_per_10g)
    rr = np.exp(beta * x / 10.0)
    if threshold > 0:
        rr = np.where(x <= threshold, 1.0, rr)
    return rr


def rr_scc(x, b1: float, b2: float):
    """Non-linear RR for oesophageal squamous-cell carcinoma."""
    x = np.asarray(x, dtype=float)
    xs = np.clip(x, 1e-9, None)
    return np.exp(b1 * xs + b2 * xs * np.log(xs))


# ----------------------------------------------------------------- consumption
def gamma_params(mu: float, sigma: float):
    """Shape/scale of the gamma distribution matching mean mu and SD sigma."""
    shape = (mu * mu) / (sigma * sigma)
    scale = (sigma * sigma) / mu
    return shape, scale


def _integrand(rr_func, shape, scale):
    def f(x):
        return gamma_dist.pdf(x, a=shape, scale=scale) * (rr_func(x) - 1.0)
    return f


# ------------------------------------------------------------------------- PAF
def paf_current(p_cd: float, mu: float, sigma: float, rr_func, lo=LO, hi=HI) -> float:
    """PAF for current drinking given prevalence, mean/SD intake and an RR function."""
    if not (p_cd and mu and sigma) or mu <= 0 or sigma <= 0:
        return 0.0
    shape, scale = gamma_params(mu, sigma)
    num, _ = integrate.quad(_integrand(rr_func, shape, scale), lo, hi, limit=100)
    num *= p_cd
    return num / (num + 1.0)


def paf_by_category(p_cd: float, mu: float, sigma: float, rr_func) -> dict:
    """PAF attributable to each drinking category (shared full denominator)."""
    if not (p_cd and mu and sigma) or mu <= 0 or sigma <= 0:
        return {k: 0.0 for k in CAT_BOUNDS}
    shape, scale = gamma_params(mu, sigma)
    integ = _integrand(rr_func, shape, scale)
    denom_num, _ = integrate.quad(integ, LO, HI, limit=100)
    denom_num *= p_cd
    denom = denom_num + 1.0
    out = {}
    for cat, (lo, hi) in CAT_BOUNDS.items():
        part, _ = integrate.quad(integ, lo, hi, limit=100)
        out[cat] = (p_cd * part) / denom
    return out


def paf_by_increment(p_cd: float, mu: float, sigma: float, rr_func, step=10.0) -> dict:
    """PAF attributable to each 10 g/day band up to 150 g/day."""
    if not (p_cd and mu and sigma) or mu <= 0 or sigma <= 0:
        return {}
    shape, scale = gamma_params(mu, sigma)
    integ = _integrand(rr_func, shape, scale)
    denom_num, _ = integrate.quad(integ, LO, HI, limit=100)
    denom = p_cd * denom_num + 1.0
    edges = [LO] + list(np.arange(step, HI + step, step))
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = min(hi, HI)
        part, _ = integrate.quad(integ, lo, hi, limit=100)
        out[(lo, hi)] = (p_cd * part) / denom
    return out


# ------------------------------------------------------------- Monte-Carlo UIs
@dataclass
class RRDraw:
    """Sampler for a site's RR function under parameter uncertainty."""
    kind: str                       # "linear" or "scc"
    rr_per_10g: float = 1.0
    var_logrr: float = 0.0
    threshold: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    cov: tuple = ((0, 0), (0, 0))   # 2x2 var-cov for (b1, b2)

    def sample(self, rng: np.random.Generator):
        if self.kind == "linear":
            beta = math.log(self.rr_per_10g)
            beta_s = rng.normal(beta, math.sqrt(max(self.var_logrr, 0.0)))
            rr10 = math.exp(beta_s)
            return lambda x, r=rr10, t=self.threshold: rr_linear(x, r, t)
        b1s, b2s = rng.multivariate_normal([self.b1, self.b2], np.array(self.cov))
        return lambda x, a=b1s, b=b2s: rr_scc(x, a, b)


def monte_carlo_paf(p_cd, mu, sigma, rr_draw: RRDraw, n=1000, seed=0,
                    rel_sd_consumption=0.10):
    """Return (point PAF, lo95, hi95) via Monte-Carlo over RR and consumption.

    RR parameters are drawn from their published (co)variances; the consumption
    mean is jittered with a modest relative SD to reflect exposure uncertainty
    (the paper simulates drinking status, mean, SD and RRs; we keep a compact,
    reproducible analogue)."""
    rng = np.random.default_rng(seed)
    point = paf_current(p_cd, mu, sigma, lambda x: _point_rr(x, rr_draw))
    if not (p_cd and mu and sigma):
        return point, point, point
    draws = np.empty(n)
    for i in range(n):
        rr_func = rr_draw.sample(rng)
        mu_i = max(rng.normal(mu, rel_sd_consumption * mu), 1e-6)
        sig_i = (sigma / mu) * mu_i
        draws[i] = paf_current(p_cd, mu_i, sig_i, rr_func)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def _point_rr(x, rr_draw: RRDraw):
    if rr_draw.kind == "linear":
        return rr_linear(x, rr_draw.rr_per_10g, rr_draw.threshold)
    return rr_scc(x, rr_draw.b1, rr_draw.b2)


# ============================================================================
# Fast grid-based integration (vectorised over countries; enables Monte Carlo)
# ============================================================================
# scipy.quad is exact but far too slow to run for every country x site x draw.
# On a fine fixed grid the PAF numerator is a trapezoidal dot product, so we can
# integrate all countries at once and re-use a precomputed gamma-pdf matrix
# across Monte-Carlo draws (only the RR curve changes per draw).

def make_grid(step: float = 0.1):
    """x grid over the integration domain [0.1, 150] g/day."""
    return np.arange(LO, HI + step, step)


def gamma_pdf_matrix(mu, sigma, x):
    """(N, len(x)) gamma pdf for arrays of means/SDs, evaluated on grid x."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    shape = (mu * mu) / (sigma * sigma)          # (N,)
    scale = (sigma * sigma) / mu                 # (N,)
    ok = np.isfinite(shape) & np.isfinite(scale) & (shape > 0) & (scale > 0)
    shape = np.where(ok, shape, 1.0)
    scale = np.where(ok, scale, 1.0)
    pdf = gamma_dist.pdf(x[None, :], a=shape[:, None], scale=scale[:, None])
    pdf[~ok, :] = 0.0
    return pdf


def paf_from_grid(p_cd, pdf_matrix, rr_grid, x):
    """Vectorised current-drinking PAF for many countries on the shared grid.

    p_cd: (N,) drinker prevalence; pdf_matrix: (N,len(x)); rr_grid: (len(x),)."""
    p_cd = np.nan_to_num(np.asarray(p_cd, float))
    integrand = pdf_matrix * (rr_grid - 1.0)[None, :]         # (N, len(x))
    num = p_cd * np.trapezoid(integrand, x, axis=1)           # (N,)
    return num / (num + 1.0)


def paf_grid_between(p_cd, pdf_matrix, rr_grid, x, lo, hi):
    """Category PAF on grid: numerator restricted to [lo,hi], full denominator."""
    p_cd = np.nan_to_num(np.asarray(p_cd, float))
    integrand = pdf_matrix * (rr_grid - 1.0)[None, :]
    denom = p_cd * np.trapezoid(integrand, x, axis=1) + 1.0
    mask = (x >= lo) & (x < hi)
    part = p_cd * np.trapezoid(integrand[:, mask], x[mask], axis=1)
    return part / denom


def rr_grid_from_draw(rr_draw: RRDraw, x, rng: np.random.Generator | None = None):
    """RR evaluated on the grid; point estimate if rng is None, else one sample."""
    if rng is None:
        return _point_rr(x, rr_draw)
    func = rr_draw.sample(rng)
    return func(x)
