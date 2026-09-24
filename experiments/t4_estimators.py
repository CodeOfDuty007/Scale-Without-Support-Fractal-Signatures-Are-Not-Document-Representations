#!/usr/bin/env python3
"""
T4 -- alternative estimators (DFA, local Whittle), the module.

Per the plan's corrected-against-primary-source estimator selection
(Millan, Osorio-Comparan & Lefranc, arXiv:2103.02091): Whittle needs
N>=2^10=1024 for its reported reliable band (discussion text also says
"high precision for N>=2^8", internally inconsistent with the paper's own
conclusion -- documented, not resolved, by the plan); Abry-Veitch wavelet
(2^13) and periodogram (2^15) are both dropped as far above GAGLE's ~400-
token support. DFA has no literature-proposed minimum here either but is
the standard "does a different estimator family also fail" comparison.

DFA (Detrended Fluctuation Analysis), implemented directly (no external
dependency): given an increment series, build the integrated profile
(cumsum of mean-centered increments), divide into non-overlapping boxes of
size n for n in a scale list, locally detrend each box with a degree-1
polynomial fit, compute the RMS residual F(n), then fit log F(n) vs log n;
the slope IS the DFA scaling exponent (directly comparable to H: for
fractional Gaussian noise, DFA order-1 alpha equals H).

Local Whittle estimator, implemented directly (no external dependency):
semiparametric estimator of the long-memory parameter d from the low-
frequency periodogram (Kunsch 1987 / Robinson 1995's local Whittle),
minimizing R(d) = log(mean_{j<=m}(I(lambda_j) lambda_j^{2d})) -
2d/m * sum_{j<=m} log(lambda_j) over d via 1-D numerical search on a bounded
grid then refined with scipy's bounded minimizer. H = d + 0.5. Bandwidth m
= floor(n^0.65) (a standard choice; too small a bandwidth is high-variance,
too large brings in short-range-dependence contamination -- documented, not
tuned per-series).

Both are validated against synthetic fBm with known Hurst exponent (reusing
the `fbm` package + the same seeding convention as t2_step0/t2_step1)
BEFORE being run on real per-document GAGLE score sequences.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import linregress


def dfa(series, scales):
    """series: 1D array (the INCREMENT series, matching calc_H's own input
    convention -- fractal_math.calc_H takes `increment`, not the raw scores).
    scales: box sizes to sweep (same role as fractal_math's scale_param).
    Returns (alpha, r_squared, fit_dict)."""
    series = np.asarray(series, dtype=np.float64)
    x = series - series.mean()
    profile = np.cumsum(x)
    n = len(profile)
    used_scales, log_f = [], []
    for s in scales:
        if s < 4 or s > n:
            continue
        n_boxes = n // s
        if n_boxes < 1:
            continue
        f2_vals = []
        for i in range(n_boxes):
            seg = profile[i * s:(i + 1) * s]
            t = np.arange(s)
            # degree-1 detrend (standard DFA1)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            resid = seg - trend
            f2_vals.append(np.mean(resid ** 2))
        if f2_vals:
            f_s = np.sqrt(np.mean(f2_vals))
            if f_s > 0:
                used_scales.append(s)
                log_f.append(np.log(f_s))
    if len(used_scales) < 2:
        return float("nan"), 0.0, {"scales": [], "reason": "insufficient_scale_points"}
    log_n = np.log(used_scales)
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_f)
    return float(slope), float(r_value ** 2), {
        "scales": used_scales, "log_scale": log_n.tolist(), "log_f": log_f,
        "slope": float(slope), "intercept": float(intercept),
    }


def local_whittle(series, bandwidth_exp=0.65):
    """series: 1D array (use the SAME increment series as calc_H/dfa for a
    fair comparison -- local Whittle is standardly applied to a stationary
    series, and the increment process is GAGLE's own stationary-by-
    construction quantity, matching how calc_H is already applied here).
    Returns (H_hat, m_bandwidth)."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 16:
        return float("nan"), 0
    m = max(2, int(n ** bandwidth_exp))
    m = min(m, n // 2 - 1)
    if m < 2:
        return float("nan"), m

    fft = np.fft.rfft(x)
    periodogram = (np.abs(fft) ** 2) / (2 * np.pi * n)
    freqs = 2 * np.pi * np.arange(1, len(fft)) / n  # exclude the zero frequency
    I = periodogram[1:len(fft)]

    lam = freqs[:m]
    Ij = I[:m]
    log_lam = np.log(lam)

    def objective(d):
        weighted = Ij * lam ** (2 * d)
        mean_w = np.mean(weighted)
        if mean_w <= 0:
            return 1e10
        return np.log(mean_w) - (2 * d / m) * np.sum(log_lam)

    res = minimize_scalar(objective, bounds=(-0.49, 1.49), method="bounded",
                           options={"xatol": 1e-4})
    d_hat = res.x
    h_hat = d_hat + 0.5
    return float(h_hat), int(m)
