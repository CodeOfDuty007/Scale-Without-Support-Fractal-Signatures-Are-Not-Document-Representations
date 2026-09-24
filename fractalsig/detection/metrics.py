"""
Frozen evaluation harness for T1/T4 detection tasks (X-04).

Every function takes plain numpy arrays (y_true in {0,1}, y_score continuous)
so it has no dependency on the feature store, splits, or the fractal
whitelist -- this module is usable standalone and does not need to wait on
P1/P2 to be written or tested.

Metrics: AUROC, macro-F1 (at a chosen threshold), TPR@1%FPR, TPR@5%FPR, ECE,
and DeLong's test for comparing two correlated (same-sample) AUROCs -- the
paired significance test the plan uses for `All` vs `All-fractal`.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score


def auroc(y_true, y_score):
    """Area under the ROC curve. y_true in {0,1}, y_score continuous (higher = more positive)."""
    return float(roc_auc_score(y_true, y_score))


def macro_f1(y_true, y_score, threshold=0.5):
    """Macro-averaged F1 at a fixed decision threshold on y_score."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro"))


def tpr_at_fpr(y_true, y_score, target_fpr):
    """True positive rate at the smallest achieved FPR <= target_fpr.

    Uses sklearn's roc_curve thresholds (all achievable operating points for
    this exact sample) rather than interpolating, so the reported TPR is
    always one this classifier actually attains on this data.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def tpr_at_1pct_fpr(y_true, y_score):
    return tpr_at_fpr(y_true, y_score, 0.01)


def tpr_at_5pct_fpr(y_true, y_score):
    return tpr_at_fpr(y_true, y_score, 0.05)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Standard equal-width-bin ECE. y_prob must be a calibrated-scale
    probability in [0, 1] (NOT a raw score) -- callers must pass model
    probabilities, not logits or perplexity."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("expected_calibration_error requires y_prob in [0, 1]; "
                          "got values outside that range -- pass calibrated "
                          "probabilities, not raw scores")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        acc = y_true[in_bin].mean()
        conf = y_prob[in_bin].mean()
        ece += (n_in_bin / n) * abs(acc - conf)
    return float(ece)


def _compute_midrank(x):
    """Midranks, per Sun & Xu (2014) 'Fast Implementation of DeLong's
    Algorithm for Comparing the Areas Under Correlated ROC Curves'."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """predictions_sorted_transposed: (n_classifiers, n_examples), positives first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, y_score_a, y_score_b):
    """Paired DeLong test comparing two correlated AUROCs on the SAME sample
    (e.g. All vs All-fractal on the same held-out set).

    Returns (auc_a, auc_b, z_statistic, p_value). p_value is two-sided.
    """
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)
    if len(y_true) != len(y_score_a) or len(y_true) != len(y_score_b):
        raise ValueError("delong_test requires y_true, y_score_a, y_score_b "
                          "to be the same length (paired, same held-out sample)")

    order = np.argsort(-y_true)  # positives (1) first
    y_true_sorted = y_true[order]
    m = int(y_true_sorted.sum())
    if m == 0 or m == len(y_true_sorted):
        raise ValueError("delong_test requires both classes present in y_true")

    preds = np.vstack([y_score_a[order], y_score_b[order]])
    aucs, delongcov = _fast_delong(preds, m)

    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var_diff <= 0:
        # Degenerate (e.g. identical scores) -- no measurable difference.
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, float(z), float(p)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals -- the sixth item on the D3 freeze list.
#
# Added 2026-08-23. The other five (AUROC, macro-F1, TPR@1%FPR, TPR@5%FPR, ECE,
# DeLong) were written 2026-08-22; this one was specified in the freeze list and
# simply missing, so X-04 was 5/6 complete until now.
#
# STRATIFIED BY DEFAULT, and that is not cosmetic here. This corpus is 2.23%
# human after X-01 (5,000 vs 218,834). An unstratified resample of ~44,000 test
# rows will occasionally draw a fold with a badly distorted positive count, and
# for TPR@1%FPR -- which depends on a quantile deep in the negative tail -- that
# produces intervals driven by resampling artifacts rather than by model
# uncertainty. Stratifying preserves each class's count in every fold.
# ---------------------------------------------------------------------------
def bootstrap_ci(metric_fn, y_true, y_score, n_boot=2000, alpha=0.05,
                 stratified=True, seed=20260823, **kwargs):
    """Percentile bootstrap CI for any metric with signature (y_true, y_score).

    Returns (point_estimate, lo, hi, n_valid_folds).

    Folds where the metric is undefined (e.g. a resample containing one class)
    are SKIPPED, not silently replaced with 0 or NaN -- and the count of
    surviving folds is returned so a caller can see when an interval rests on
    fewer folds than it asked for. A CI computed from 200 usable folds out of
    2000 is a different object from one computed from 2000, and the caller
    should be able to tell.

    Example:
        auc, lo, hi, k = bootstrap_ci(auroc, y_true, y_score)
        f1, lo, hi, k = bootstrap_ci(macro_f1, y_true, y_score, threshold=0.4)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be the same length")
    rng = np.random.default_rng(seed)

    if stratified:
        pos = np.flatnonzero(y_true == 1)
        neg = np.flatnonzero(y_true != 1)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("bootstrap_ci requires both classes present")

    vals = []
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                                  rng.choice(neg, size=len(neg), replace=True)])
        else:
            idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            v = metric_fn(y_true[idx], y_score[idx], **kwargs)
        except Exception:
            continue
        if v is not None and v == v:  # skip NaN
            vals.append(float(v))

    if not vals:
        raise ValueError("no bootstrap fold produced a usable metric value")
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(metric_fn(y_true, y_score, **kwargs))
    return point, float(lo), float(hi), int(len(vals))


def bootstrap_ci_paired(metric_fn, y_true, y_score_a, y_score_b, n_boot=2000,
                        alpha=0.05, stratified=True, seed=20260823, **kwargs):
    """Percentile CI for the DIFFERENCE metric(a) - metric(b) on the same rows.

    Paired: both models are evaluated on the SAME resampled rows each fold, so
    the interval reflects the difference between models rather than the sum of
    two independent sampling errors. Use alongside delong_test() for AUROC --
    DeLong is the analytic paired test, this is its empirical counterpart, and
    they should agree. Where they disagree, prefer DeLong for AUROC and this for
    metrics DeLong does not cover (macro-F1, TPR@FPR, ECE).

    Returns (observed_diff, lo, hi, n_valid_folds).
    """
    y_true = np.asarray(y_true)
    a, b = np.asarray(y_score_a), np.asarray(y_score_b)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("y_true, y_score_a, y_score_b must be the same length")
    rng = np.random.default_rng(seed)

    if stratified:
        pos = np.flatnonzero(y_true == 1)
        neg = np.flatnonzero(y_true != 1)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("bootstrap_ci_paired requires both classes present")

    diffs = []
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                                  rng.choice(neg, size=len(neg), replace=True)])
        else:
            idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            d = metric_fn(y_true[idx], a[idx], **kwargs) - \
                metric_fn(y_true[idx], b[idx], **kwargs)
        except Exception:
            continue
        if d == d:
            diffs.append(float(d))

    if not diffs:
        raise ValueError("no bootstrap fold produced a usable difference")
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    obs = float(metric_fn(y_true, a, **kwargs) - metric_fn(y_true, b, **kwargs))
    return obs, float(lo), float(hi), int(len(diffs))
