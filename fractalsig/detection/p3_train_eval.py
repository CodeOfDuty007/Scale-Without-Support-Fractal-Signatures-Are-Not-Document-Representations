#!/usr/bin/env python3
"""
P3 Task 5 -- train + evaluate the T1 binary detector.

Implements the P3 training design notes, which was pre-registered BEFORE any
fit (D-13, anti-selective-reporting). Read that file first; this module does not
restate its reasoning, only its decisions.

D-11 sequence, and it is load-bearing:
    class weighting  ->  calibrate on validation at TRUE prevalence  ->  report

POSITIVE CLASS = machine-generated (LLM). FPR is the rate at which HUMAN text is
flagged as AI, over 5,000 human rows.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd


from fractalsig.detection.metrics import (auroc, macro_f1, tpr_at_1pct_fpr, tpr_at_5pct_fpr,
                          expected_calibration_error, delong_test,
                          bootstrap_ci, bootstrap_ci_paired)
from fractalsig.detection.splits import group_key  # the frozen grouping rule -- never reimplemented here
from fractalsig.detection.balancing import balanced_sample_weight  # shared, not restated -- see fractalsig/detection/balancing.py

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, *p)

SEED = 20260824
BOOT_SEED = 20260823          # the frozen harness default; not re-chosen here
N_BOOT = 2000
HUMAN = "Human"

# --------------------------------------------------------------------------
# Feature groups -- fixed by the design doc, not discovered from the data.
# --------------------------------------------------------------------------
STAT = ["Gemma_LogPPL_Mean", "Gemma_LogPPL_Var", "Gemma_LogPPL_Skew",
        "Gemma_LogPPL_Kurtosis", "Gemma_Entropy_Mean", "Gemma_GLTR_top10_frac",
        "Gemma_GLTR_top100_frac", "Gemma_GLTR_top1000_frac",
        "Mistral_LogPPL_Mean", "Mistral_LogPPL_Var", "Mistral_LogPPL_Skew",
        "Mistral_LogPPL_Kurtosis", "Mistral_Entropy_Mean",
        "Mistral_GLTR_top10_frac", "Mistral_GLTR_top100_frac",
        "Mistral_GLTR_top1000_frac", "Cross_PPL_Ratio"]

STYLE = ["Sent_Len_Mean", "Sent_Len_Std", "Para_Len_Mean", "Para_Len_Std",
         "Sent_Transition_Entropy", "Discourse_Marker_Contrastive_Per100W",
         "Discourse_Marker_Additive_Per100W", "Discourse_Marker_Causal_Per100W",
         "Discourse_Marker_Exemplifying_Per100W", "TTR",
         "Bigram_Repetition_Rate", "Embedding_Dispersion", "N_Sentences",
         "N_Words_Used"]

# P0-P5 audit (2026-08-28), step 6: the log-PPL-only subset of STAT, for the
# 9-step brief's Step 8 ablation ("fractal + log-PPL" specifically, narrower
# than STAT's 17 features which also bundle entropy and GLTR rank-buckets --
# neither of those is log-PPL). `Cross_PPL_Ratio` IS included: it is a ratio
# of the two scorers' own log-PPL values (Gemma vs Mistral), derived from
# log-PPL and nothing else, not an entropy/rank-based feature -- a
# deliberate inclusion decision, stated here rather than left implicit.
# Derived from STAT (2026-08-29, was a hand-typed re-listing until code
# review flagged the drift risk: a future rename/removal of a *_LogPPL_*
# column in STAT would have left this list silently pointing at a
# nonexistent name). The predicate encodes the actual documented intent
# above (log-PPL and nothing else, Cross_PPL_Ratio included deliberately)
# rather than an exclude-list, so it doesn't silently absorb any future
# non-LogPPL column STAT might grow.
LOGPPL = [c for c in STAT if "LogPPL" in c or c == "Cross_PPL_Ratio"]
assert LOGPPL == ["Gemma_LogPPL_Mean", "Gemma_LogPPL_Var", "Gemma_LogPPL_Skew",
                  "Gemma_LogPPL_Kurtosis", "Mistral_LogPPL_Mean", "Mistral_LogPPL_Var",
                  "Mistral_LogPPL_Skew", "Mistral_LogPPL_Kurtosis", "Cross_PPL_Ratio"], \
    "LOGPPL derivation drifted from the frozen 9-column set"

# EXCLUDED as inputs. The 11 the P1 exit gate reclassified as metadata, plus the
# six window-length provenance columns found 2026-08-24 (design doc S2.1(b)):
# Human is pinned to a single value in all six, so they are one-way class
# indicators of the D-14/F-5 shape. Kept here by name so `provenance_only` can
# quantify what the exclusion cost.
PROVENANCE_LEAK = ["Gemma_Feature_Window_N_Scored", "Mistral_Feature_Window_N_Scored",
                   "Gemma_S_window_n_windows", "Gemma_H_window_n_windows",
                   "Mistral_S_window_n_windows", "Mistral_H_window_n_windows"]

EXIT_GATE_FLAGS = ["Holder_PerDoc_Unreliable", "Mistral_available", "Gemma_available",
                   "Family_C_Full_Text_Fallback", "Embedding_Dispersion_Deferred_To_GPU",
                   "Gemma_SelfScored", "Mistral_SelfScored",
                   "Gemma_Feature_Window_Available", "Mistral_Feature_Window_Available",
                   "Gemma_Used_Short_Window", "Mistral_Used_Short_Window",
                   "ZWSP_Present_In_Source_Text"]

SHORTWIN_FLAGS = ["Gemma_Used_Short_Window_RawLen", "Mistral_Used_Short_Window_RawLen"]

LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                  min_child_samples=50, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED,
                  n_jobs=-1, verbose=-1, class_weight="balanced")
# No early stopping. The validation split is reserved EXCLUSIVELY for
# calibration (D-11 step 2); using it for round selection as well would fit the
# calibrator on data the model had already been tuned against. n_estimators is
# fixed in advance rather than chosen after seeing a curve.


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def load_matrix():
    """p3_features.jsonl (Family A, F-4 vintage) + parquet (Families B, C).

    Family A is taken from the P3 file, NOT the parquet: the parquet carries the
    P1-vintage S/H and has no log-log residual columns at all. Mixing them would
    put two vintages of S in one matrix.
    """
    man = json.load(open(D("feature_store", "p3_feature_manifest.json")))
    fractal = list(man["features"])

    rows = []
    with open(D("data", "p3_features.jsonl")) as fh:
        for line in fh:
            rows.append(json.loads(line))
    p3 = pd.DataFrame(rows)

    # The retired GAGLE `split` label (X-01/D-07). Dropped, never read.
    p3 = p3.drop(columns=[c for c in ("split",) if c in p3.columns])

    pq = pd.read_parquet(D("feature_store", "v1_gagle.parquet"),
                         columns=["id", "domain", "Label"] + STAT + STYLE
                                 + PROVENANCE_LEAK)
    df = pq.merge(p3[["id"] + [c for c in fractal if c in p3.columns]],
                  on="id", how="inner", validate="1:1")

    flags = pd.read_json(D("data", "p3_short_window_flags.jsonl"), lines=True)
    flags = flags[["id"] + SHORTWIN_FLAGS]
    df = df.merge(flags, on="id", how="inner", validate="1:1")

    missing_declared = [c for c in fractal if c not in df.columns]
    if missing_declared:
        raise SystemExit(f"[FATAL] declared features absent from the store: "
                         f"{missing_declared}. D-13 requires every declared "
                         f"feature to be reported; a silent drop is the failure "
                         f"mode the manifest exists to prevent.")

    assert len(df) == 223834, f"expected 223,834 rows, got {len(df):,}"
    assert df.id.is_unique
    assert "split" not in df.columns, "the retired GAGLE split label leaked in"

    df["y"] = (df["Label"] != HUMAN).astype(int)      # positive = LLM
    df["group"] = [group_key({"id": i, "domain": d, "Label": l})
                   for i, d, l in zip(df.id, df.domain, df.Label)]
    if df["group"].isna().any():
        raise SystemExit("[FATAL] some ids yielded no group key")
    df["group"] = df["group"].map(lambda g: f"{g[0]}|{g[1]}")

    groups = {"fractal": fractal, "stat": STAT, "style": STYLE}
    return df, groups


def arm_features(arm, groups):
    f, s, y = groups["fractal"], groups["stat"], groups["style"]
    LENGTH = ["N_Words_Used", "N_Sentences"]
    return {"fractal_only": f,
            "all_minus_length": [c for c in f + s + y if c not in LENGTH],
            "stat_only": s,
            "style_only": y,
            "fractal+stat": f + s,
            "fractal+style": f + y,
            "all": f + s + y,
            "all_minus_fractal": s + y,
            # P0-P5 audit (2026-08-28), step 6: LOGPPL columns are already
            # part of STAT (loaded into df regardless of arm), so no new
            # groups-dict entry or load_matrix change is needed -- just a
            # narrower feature-name subset of what's already there.
            "fractal+logppl": f + LOGPPL}[arm]


# --------------------------------------------------------------------------
# Split schemes
# --------------------------------------------------------------------------
def build_folds(df, splits):
    """Yield (scheme, fold_name, train_mask, val_mask, test_mask)."""
    assign = splits["random"]
    a = df["group"].map(assign)
    if a.isna().any():
        raise SystemExit("[FATAL] rows map to groups absent from the frozen manifest")

    yield ("random", "random",
           (a == "train").values, (a == "val").values, (a == "test").values)

    for d in splits["domains"]:
        val_groups = set(splits["lodo"]["lodo_val"][d])
        in_d = (df["domain"] == d).values
        is_val = df["group"].isin(val_groups).values
        # TRAIN = other domains, minus their val-assigned groups (design doc S6)
        train = (~in_d) & (a != "val").values
        yield ("lodo", d, train, is_val, in_d)

    for gen in splits["generators"]:
        is_gen = (df["Label"] == gen).values
        is_hum = (df["Label"] == HUMAN).values
        other = (~is_gen) & (~is_hum)
        tr_g = (a == "train").values
        va_g = (a == "val").values
        te_g = (a == "test").values
        # Group partition FIRST, generator holdout second (manifest _leak_warning)
        yield ("logo", gen,
               tr_g & (other | is_hum),
               va_g & (other | is_hum),
               te_g & (is_gen | is_hum))


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------
def config_mask(df, cfg, feats_all):
    """Row mask for a configuration. Returns (mask, note)."""
    n = len(df)
    if cfg == "full":
        return np.ones(n, bool), "all rows"

    if cfg == "no_short_window":
        # NONE POLICY, STATED: exclude if either flag is True OR either is None.
        # None = "no Gemma array at all", so the branch is unknowable and the row
        # cannot be certified clean. The flag file cannot express a policy; this
        # is it.
        g, m = df[SHORTWIN_FLAGS[0]], df[SHORTWIN_FLAGS[1]]
        drop = (g.fillna(True).astype(bool)) | (m.fillna(True).astype(bool))
        keep = ~drop.values
        assert keep.sum() == 202477, (
            f"config 2 kept {keep.sum():,}; the stated None policy predicts "
            f"202,477. A silent no-op join would show 223,834.")
        return keep, f"excluded {int((~keep).sum()):,} (either flag True or None)"

    if cfg == "no_nan_signal":
        return np.ones(n, bool), "all rows; NaN replaced by draws from the train-fold distribution"

    if cfg == "clean_both":
        # BOTH confound routes removed at once: short-window rows excluded AND
        # no missing values anywhere. Provenance columns are already excluded
        # from the input space in every config, so this is the configuration
        # with the fewest un-ablated routes -- and by the pre-registered rule in
        # the design doc it carries the paper's claim even if it is the weakest.
        g, m = df[SHORTWIN_FLAGS[0]], df[SHORTWIN_FLAGS[1]]
        drop = (g.fillna(True).astype(bool)) | (m.fillna(True).astype(bool))
        keep = (~drop.values) & df[feats_all].notna().all(axis=1).values
        assert keep.sum() == 70471, (
            f"clean_both kept {keep.sum():,}, expected 70,471. This constant is "
            f"complete-case over the CURRENT 85-column feature set intersected "
            f"with the D-14 flags. If the feature GROUP DEFINITIONS changed, the "
            f"complete-case count legitimately changes and this constant must be "
            f"recomputed DELIBERATELY and the design doc updated -- do not just "
            f"edit the number to make the assertion pass.")
        return keep, "short-window excluded AND zero missing values"

    if cfg == "complete_case":
        keep = df[feats_all].notna().all(axis=1).values
        return keep, f"rows with zero missing values across all {len(feats_all)} features"

    raise ValueError(cfg)


def impute_draw(Xtr, Xother, rng):
    """Replace NaN with a seeded random draw from the column's TRAIN-fold
    observed values. Destroys the point mass a median leaves behind, which is
    what makes this an ablation rather than a relocation (design doc S3)."""
    out = [Xtr.copy()] + [x.copy() for x in Xother]
    for j in range(Xtr.shape[1]):
        obs = Xtr[:, j][~np.isnan(Xtr[:, j])]
        if obs.size == 0:
            obs = np.array([0.0])
        for arr in out:
            bad = np.isnan(arr[:, j])
            if bad.any():
                arr[bad, j] = rng.choice(obs, size=int(bad.sum()), replace=True)
    return out[0], out[1:]


def impute_median(Xtr, Xother):
    """The positive control. Expected to FAIL the leak check."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    out = []
    for arr in [Xtr] + list(Xother):
        a = arr.copy()
        idx = np.where(np.isnan(a))
        a[idx] = np.take(med, idx[1])
        out.append(a)
    return out[0], out[1:]


# --------------------------------------------------------------------------
# Fit + evaluate
# --------------------------------------------------------------------------
def ap_human(y_true, y_score):
    """Average precision for detecting the HUMAN class (no-skill floor 0.0223).

    With positive = LLM at 97.77% prevalence, AP has a 0.9777 floor and reads
    ~0.99 for every arm including the pure-confound diagnostics -- it would
    discriminate nothing. Flipped here, and labelled AP_human everywhere.
    """
    y_true = np.asarray(y_true)
    return float(average_precision_score(1 - y_true, 1.0 - np.asarray(y_score)))


def evaluate(y, p, with_ci):
    out = {"n": int(len(y)), "n_pos_llm": int(y.sum()),
           "n_neg_human": int((y == 0).sum()),
           "prevalence_human": float((y == 0).mean())}
    point = {"tpr_at_1pct_fpr": tpr_at_1pct_fpr(y, p),
             "tpr_at_5pct_fpr": tpr_at_5pct_fpr(y, p),
             "auroc": auroc(y, p),
             "ap_human": ap_human(y, p),
             "macro_f1_at_0.5_calibrated": macro_f1(y, p, threshold=0.5),
             "ece_calibrated": expected_calibration_error(y, p)}
    out["point"] = point
    if with_ci:
        ci = {}
        for name, fn in [("tpr_at_1pct_fpr", tpr_at_1pct_fpr),
                         ("tpr_at_5pct_fpr", tpr_at_5pct_fpr),
                         ("auroc", auroc), ("ap_human", ap_human),
                         ("macro_f1_at_0.5_calibrated", macro_f1)]:
            try:
                est, lo, hi, k = bootstrap_ci(fn, y, p, n_boot=N_BOOT,
                                              stratified=True, seed=BOOT_SEED)
                ci[name] = {"point": est, "lo": lo, "hi": hi, "n_valid_folds": k}
            except Exception as e:
                ci[name] = {"error": str(e)}
        out["ci95"] = ci
    return out


def _imputed_scaled_pipeline(final_step):
    """SimpleImputer(median) + StandardScaler() + final_step, as a FRESH
    Pipeline per call (never a module-level shared instance -- each fit()
    call must own its own fitted state). Extracted 2026-08-29 -- the
    "logreg" and "mlp" branches of fit_predict built this identical
    preamble independently."""
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     final_step])


def fit_predict(learner, Xtr, ytr, Xva, yva, Xte, feat_names):
    """D-11: weight -> calibrate on val at true prevalence -> return probs.

    The calibrator receives NO sample_weight and the validation rows are neither
    resampled nor subset. That -- not prevalence equality -- is what D-11
    constrains; see the design doc S4 correction.
    """
    if learner == "lgbm":
        base = lgb.LGBMClassifier(**LGB_PARAMS)
        base.fit(Xtr, ytr)   # names zipped back by column order, not passed in
    elif learner == "logreg":
        base = _imputed_scaled_pipeline(
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)))
        base.fit(Xtr, ytr)
    elif learner == "mlp":
        # D-05: "shallow MLP" was deferred from P3, promised for P5, never
        # landed there either -- added here in the P0-P5 audit (2026-08-28).
        # One hidden layer (shallow, per the spec's own wording), imputed +
        # scaled the same way as logreg since MLPs (unlike LightGBM) can't
        # handle NaN or unscaled inputs natively. sklearn's MLPClassifier has
        # no native class_weight -- balanced via sample_weight on .fit()
        # instead, since Pipeline.fit() only forwards **fit_params by the
        # final step's name (mlp__sample_weight), not a constructor arg.
        # De-duplicated 2026-08-29 (code review): this formula was
        # independently hand-rolled here and in 3 RoBERTa training scripts --
        # see fractalsig/detection/balancing.py for the shared version and why it's kept as
        # hand-rolled math rather than sklearn.utils.class_weight.
        sample_weight = balanced_sample_weight(ytr)
        base = _imputed_scaled_pipeline(
            ("mlp", MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-4,
                                   max_iter=300, early_stopping=False, random_state=SEED)))
        base.fit(Xtr, ytr, mlp__sample_weight=sample_weight)
    else:
        raise ValueError(learner)

    n_val_before = len(Xva)
    cal = CalibratedClassifierCV(estimator=base, cv="prefit", method="sigmoid")
    cal.fit(Xva, yva)                      # no sample_weight, no resampling
    assert len(Xva) == n_val_before, "validation rows changed during calibration"

    pos = int(np.where(cal.classes_ == 1)[0][0])   # never a hardcoded [:, 1]
    return cal.predict_proba(Xte)[:, pos], base


def assert_disjoint(df, tr, va, te, tag):
    gtr, gva, gte = (set(df.loc[m, "group"]) for m in (tr, va, te))
    bad = (gtr & gte) | (gtr & gva) | (gva & gte)
    if bad:
        raise SystemExit(f"[FATAL] {tag}: {len(bad)} groups span >1 partition, "
                         f"e.g. {list(bad)[:3]}. Group-disjointness is what "
                         f"prevents an article's human original sitting in train "
                         f"while its own continuation sits in test.")


# --------------------------------------------------------------------------
# Diagnostic arms -- each exists so a confound is measured, not argued.
# Each has a stated failure condition; see the design doc S8.
# --------------------------------------------------------------------------
def diagnostic_matrix(df, which, feats_all):
    if which == "provenance_only":
        return df[PROVENANCE_LEAK].to_numpy(float), list(PROVENANCE_LEAK)
    if which == "missingness_only":
        names = [f"isna__{c}" for c in feats_all]
        return df[feats_all].isna().to_numpy(float), names
    if which == "length_only":
        # Human documents are far longer than generated ones (mean 691 words vs
        # ~318): GAGLE's LLM continuations stop at a generation budget. That is a
        # property of the corpus, not of our preprocessing, so the columns are
        # KEPT as inputs -- but a referee will ask how much of the detector is
        # just a length ruler, and this arm answers it with a number.
        return df[["N_Words_Used", "N_Sentences"]].to_numpy(float), ["N_Words_Used", "N_Sentences"]
    if which == "shortwindow_only":
        cols = []
        for c in SHORTWIN_FLAGS:
            v = df[c]
            cols.append(v.map({True: 1.0, False: 0.0}).fillna(-1.0).to_numpy(float))
        return np.column_stack(cols), list(SHORTWIN_FLAGS)
    raise ValueError(which)


def _fit_auroc(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                           random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, int(np.where(m.classes_ == 1)[0][0])]
    return float(auroc(yte, p))


# The 20 columns that are never NaN anywhere in the corpus. They carry no
# imputation artifact by construction, so what they predict about
# `was_missing_any` is the GENUINE mechanism (missingness is caused by the data:
# p_eps(tau) reaches zero at large tau, so the scale drops out of the fit), not
# an artifact of how we imputed.
ALWAYS_PRESENT = ["Mistral_H", "Mistral_S_R2", "Mistral_H_R2",
                  "Mistral_H_CI_halfwidth", "Mistral_h_res_8", "Mistral_h_res_16",
                  "Mistral_h_res_32", "Mistral_h_res_48", "Mistral_h_res_64",
                  "Mistral_H_window_mean", "Mistral_H_window_var",
                  "Mistral_H_window_min", "Mistral_H_window_max",
                  "Mistral_H_window_drift",
                  "Discourse_Marker_Contrastive_Per100W",
                  "Discourse_Marker_Additive_Per100W",
                  "Discourse_Marker_Causal_Per100W",
                  "Discourse_Marker_Exemplifying_Per100W",
                  "N_Sentences", "N_Words_Used"]


def leak_check(df, tr, te, names, was_missing):
    """Did config 3's imputation ABLATE the missingness signal, or merely
    RELOCATE it?

    Reading `AUROC(was_missing) ~ 0.5 => success` is WRONG here and the first
    draft of this check said exactly that. Missingness is mechanistically caused
    by the data, so `was_missing_any` is partly predictable from perfectly
    observed columns no matter what we impute. The check therefore needs a floor,
    and it has three arms:

      floor   -- predict was_missing from the 20 NEVER-missing columns only.
                 No imputed value is involved, so this is the mechanism alone.
      draw    -- predict it from the randomly-imputed matrix (config 3).
      median  -- POSITIVE CONTROL, expected to fail: median imputation leaves a
                 point mass one split recovers.

    PASS  = draw is at or near floor  -> the imputation added nothing recoverable
    FAIL  = draw is well above floor  -> it relocated the signal, and config 3's
            headline number must not be read as a missingness ablation.
    The median arm's job is to prove the check CAN fail.
    """
    wm_tr, wm_te = was_missing[tr], was_missing[te]
    if wm_tr.sum() in (0, len(wm_tr)) or wm_te.sum() in (0, len(wm_te)):
        return {"skipped": "no variation in was_missing on this fold"}

    Xall = df[names].to_numpy(float)
    Xfl = df[ALWAYS_PRESENT].to_numpy(float)
    Xd_tr, (Xd_te,) = impute_draw(Xall[tr], [Xall[te]], np.random.default_rng(SEED))
    Xm_tr, (Xm_te,) = impute_median(Xall[tr], [Xall[te]])

    floor = _fit_auroc(Xfl[tr], wm_tr, Xfl[te], wm_te)
    draw = _fit_auroc(Xd_tr, wm_tr, Xd_te, wm_te)
    med = _fit_auroc(Xm_tr, wm_tr, Xm_te, wm_te)
    return {"auroc_floor_never_missing_cols": floor,
            "auroc_random_draw_imputed": draw,
            "auroc_median_imputed_positive_control": med,
            "excess_over_floor": (None if (draw is None or floor is None)
                                  else round(draw - floor, 4)),
            "n_test": int(len(wm_te)),
            "frac_missing_test": float(wm_te.mean())}


def imputation_detectability(df, tr, te, names, rng_seed=SEED):
    """The TIGHT version of the leak check -- no mechanism confound at all.

    leak_check()'s floor is a LOWER bound on the mechanism (it uses only 20
    columns, so it under-reports how much a full matrix legitimately knows about
    missingness), which makes its `excess_over_floor` an UPPER bound on the
    artifact rather than a measurement of it. This check removes the confound by
    construction:

      Take only COMPLETE rows -- no NaN anywhere. Split them in half.
      Half B gets a real missingness pattern (sampled from actual incomplete
      rows) applied and then imputed; half A is left alone. Train a model to
      tell A from B.

    A and B are drawn from the same population and differ ONLY by
    mask-then-impute, so there is no mechanism to confound with. AUROC ~ 0.5
    means an imputed value is indistinguishable from a real one; AUROC well
    above 0.5 means the imputation is detectable and config 3 removed less than
    it appears to. It CAN fail -- median imputation is run beside it as the
    positive control.
    """
    X = df[names].to_numpy(float)
    nan_mask = np.isnan(X)
    complete = ~nan_mask.any(axis=1)
    donors = np.flatnonzero(nan_mask.any(axis=1))
    rng = np.random.default_rng(rng_seed)

    def build(idx_rows):
        idx = np.flatnonzero(idx_rows & complete)
        if len(idx) < 400 or len(donors) == 0:
            return None
        rng.shuffle(idx)
        half = len(idx) // 2
        A, B = idx[:half], idx[half:]
        Xa, Xb = X[A].copy(), X[B].copy()
        pat = nan_mask[rng.choice(donors, size=len(B), replace=True)]
        Xb[pat] = np.nan
        lab = np.concatenate([np.zeros(len(A), int), np.ones(len(B), int)])
        return np.vstack([Xa, Xb]), lab

    btr, bte = build(tr), build(te)
    if btr is None or bte is None:
        return {"skipped": "too few complete rows in this fold"}
    Xtr, ytr = btr
    Xte, yte = bte
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte))}
    for tag, fn in (("random_draw", impute_draw), ("median_positive_control", impute_median)):
        if fn is impute_draw:
            a, (b,) = impute_draw(Xtr, [Xte], np.random.default_rng(rng_seed))
        else:
            a, (b,) = impute_median(Xtr, [Xte])
        # NOTE the train-fold distribution the imputer draws from is Xtr itself,
        # which here is half-real/half-masked. That is the conservative choice:
        # it gives the imputer the best possible marginal to imitate, so a high
        # AUROC cannot be blamed on a bad reference distribution.
        out[f"auroc_{tag}"] = _fit_auroc(a, ytr, b, yte)
    return out


N_SHUFFLES = 10
LEAK_FRACTIONS = (0.0, 0.02, 0.05, 0.20)


def label_shuffle_control(df, groups, tr, te, yall, cfg, scheme, fold, log):
    """HARNESS control -- tests the evaluation path, not any confound.

    Ten INDEPENDENT permutations of the training labels, evaluated against the
    TRUE test labels on the BASE model's scores.

    Three design points, each of which a first draft got wrong:

    1. ONLY the training labels are permuted, and NO calibrator is fitted.
       Permuting the validation labels too fits the Platt sigmoid on noise --
       that measures calibration-on-noise, a different question. But fitting the
       calibrator on TRUE validation labels is worse: a monotone sigmoid with a
       fitted sign would ORIENT the null model's random correlation to be
       positive every time, so every shuffle would land above 0.5 by
       construction and the control could not fail. Rank metrics are unaffected
       by monotone calibration, so the base scores are the honest object here.

    2. The criterion is ONE-SIDED. Measured on this corpus, a label-shuffled
       model lands systematically BELOW chance (10/10 seeds, mean ~0.357), not
       at 0.5: human rows sit in a sparse, extreme region of feature space
       (23.08% of them have N_Words_Used outside the entire LLM range), and a
       class-weighted booster fitted on random labels treats that region by its
       global prior rather than its true composition. That is a property of the
       model and the corpus, NOT leakage. **Leakage inflates performance toward
       1.0; it cannot make a model worse than chance.** So the failure condition
       is "materially ABOVE 0.5", and a below-chance result is recorded and
       explained rather than treated as a failure.

    3. The one-sided criterion is itself verified, not asserted, by INJECTING
       leakage: a fraction of the test rows is added to training WITH TRUE
       LABELS. If the shuffled-label AUROC does not rise above 0.5 as that
       fraction grows, this control cannot detect the thing it exists to detect
       and no number from the run may be published.
    """
    names = arm_features("all", groups)
    X = df[names].to_numpy(float)
    tri, tei = np.flatnonzero(tr), np.flatnonzero(te)
    yte = yall[tei]

    aucs = []
    for seed in range(N_SHUFFLES):
        ysh = np.random.default_rng(20260824 + seed).permutation(yall[tri])
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
        m.fit(X[tri], ysh)
        p = m.predict_proba(X[tei])[:, int(np.where(m.classes_ == 1)[0][0])]
        aucs.append(float(auroc(yte, p)))

    a = np.asarray(aucs)
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    n_above = int((a > 0.5).sum())
    # FAIL only in the leakage direction.
    fails = bool((a.mean() - 1.96 * se) > 0.5 or n_above >= N_SHUFFLES - 1)

    out = {"config": cfg, "scheme": scheme, "fold": fold,
           "control": "label_shuffle", "n_shuffles": N_SHUFFLES,
           "aucs": aucs, "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
           "sem": se, "n_above_0.5": n_above,
           "criterion": "one-sided: FAIL iff mean-1.96*SEM > 0.5 or >= 9/10 above 0.5",
           "verdict": "FAIL (harness leaks)" if fails else "PASS"}

    # leak injection -- only worth its cost once, on the in-distribution split
    if scheme == "random" and cfg == "full":
        inj = []
        rng = np.random.default_rng(99)
        for frac in LEAK_FRACTIONS:
            k = int(len(tei) * frac)
            leaked = rng.choice(tei, size=k, replace=False) if k else np.array([], int)
            rows = np.concatenate([tri, leaked])
            ysh = np.random.default_rng(20260824).permutation(yall[tri])
            ylab = np.concatenate([ysh, yall[leaked]])
            m = lgb.LGBMClassifier(**LGB_PARAMS)
            m.fit(X[rows], ylab)
            p = m.predict_proba(X[tei])[:, int(np.where(m.classes_ == 1)[0][0])]
            inj.append({"leak_fraction": frac, "n_leaked_rows": int(k),
                        "auroc": float(auroc(yte, p))})
        out["leak_injection"] = inj
        rises = all(inj[i]["auroc"] <= inj[i + 1]["auroc"] + 1e-9
                    for i in range(len(inj) - 1)) and inj[-1]["auroc"] > 0.5
        out["leak_injection_detects_leakage"] = bool(rises)
        out["control_is_falsifiable"] = bool(rises)
        if not rises:
            out["verdict"] = "BROKEN CONTROL (leak injection did not register)"

    log(f"      [label_shuffle] {N_SHUFFLES} shuffles, base scores: "
        f"mean AUROC={a.mean():.4f} sd={a.std(ddof=1):.4f} "
        f"range [{a.min():.4f}, {a.max():.4f}]  above 0.5: {n_above}/{N_SHUFFLES}"
        f"  -> {out['verdict']}")
    if "leak_injection" in out:
        log("      [label_shuffle] leak injection (positive control for the control): "
            + "  ".join(f"{d['leak_fraction']:.0%}->{d['auroc']:.4f}"
                        for d in out["leak_injection"])
            + f"   detects leakage: {out['leak_injection_detects_leakage']}")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
GRID_FULL = ["fractal_only", "stat_only", "style_only", "fractal+stat",
             "fractal+style", "all", "all_minus_fractal", "all_minus_length"]
GRID_CONFOUND = ["all", "all_minus_fractal", "fractal_only", "stat_only"]
DIAGNOSTICS = ["provenance_only", "missingness_only", "shortwindow_only",
               "length_only"]

# NEGATIVE CONTROL on the harness itself, not on any confound. Train the `all`
# arm on PERMUTED training labels and evaluate against the TRUE test labels. A
# model that learned nothing must land at chance. Anything meaningfully above
# 0.5 means the evaluation path leaks -- test rows reachable from training, a
# metric computed against the wrong vector, an index misalignment between
# predictions and labels. None of the four confound configs can detect that
# class of failure, because all four would be wrong together.
#
# WHAT IT CANNOT DETECT (stated, so it is not over-trusted): it does not test
# group-disjointness (a duplicated article with a shuffled label teaches
# nothing), and it does not test whether a feature is a legitimate signal or an
# artifact -- a perfect one-way provenance indicator passes this control
# trivially, because the control is about the harness, not the features.
CONTROLS = ["label_shuffle"]
# (CI_ARMS retired 2026-08-24: every cell now gets a CI -- see the with_ci
#  comment below for the measured cost that made the restriction unnecessary.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=D("data", "p3_t1_results.json"))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: random split only, 2 arms, n_boot=50")
    ap.add_argument("--configs", default="full,no_short_window,no_nan_signal,complete_case,clean_both")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm subset; empty = the design's full grid")
    args = ap.parse_args()

    global N_BOOT
    t0 = time.time()
    log("=" * 78)
    log("P3 TASK 5 -- T1 binary detection.  positive class = LLM (machine).")
    log("D-11 order: class weighting -> calibrate on val at true prevalence -> report")
    log("=" * 78)

    df, groups = load_matrix()
    feats_all = groups["fractal"] + groups["stat"] + groups["style"]
    log(f"[load] {len(df):,} rows x {len(feats_all)} features "
        f"({len(groups['fractal'])} fractal / {len(groups['stat'])} stat / "
        f"{len(groups['style'])} style)")
    log(f"[load] composition {df.Label.value_counts().to_dict()}  "
        f"human prevalence {100*(df.y==0).mean():.3f}%")
    log(f"[guard] retired GAGLE 'split' column absent from the matrix: "
        f"{'split' not in df.columns}")
    log(f"[guard] excluded as inputs: {len(EXIT_GATE_FLAGS)} exit-gate flags + "
        f"{len(PROVENANCE_LEAK)} window-length provenance columns")

    splits = json.load(open(D("data", "splits_v1.json")))
    folds = list(build_folds(df, splits))
    log(f"[splits] {len(folds)} folds: " +
        ", ".join(f"{s}/{f}" for s, f, *_ in folds))

    configs = args.configs.split(",")
    if args.quick:
        # --quick is a smoke test: 1 fold, 50 bootstrap folds. It no longer
        # overrides an EXPLICIT --configs/--arms, so a specific path can be
        # smoke-tested rather than only the default pair.
        N_BOOT = 50
        folds = folds[:1]
        if "--configs" not in " ".join(sys.argv):
            configs = ["full", "no_nan_signal"]

    Xall = df[feats_all].to_numpy(float)
    was_missing = np.isnan(Xall).any(axis=1)
    yall = df["y"].to_numpy()

    results = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "design_doc": "the P3 training design notes",
               "positive_class": "LLM (machine-generated); FPR is over human rows",
               "n_boot": N_BOOT, "seed": SEED,
               "corpus": {"n": int(len(df)),
                          "composition": {k: int(v) for k, v in
                                          df.Label.value_counts().items()},
                          "prevalence_human": float((yall == 0).mean())},
               "excluded_inputs": {"exit_gate_flags": EXIT_GATE_FLAGS,
                                   "window_length_provenance": PROVENANCE_LEAK},
               "feature_groups": {k: v for k, v in groups.items()},
               "cells": [], "importance": {}, "leak_checks": [],
               "paired": [], "controls": []}

    for cfg in configs:
        cmask, cnote = config_mask(df, cfg, feats_all)
        log("")
        log("#" * 78)
        log(f"# CONFIG {cfg}: {int(cmask.sum()):,} rows ({100*cmask.mean():.2f}%) -- {cnote}")
        sub = df.loc[cmask]
        log(f"#   composition {sub.Label.value_counts().to_dict()}  "
            f"human {100*(sub.y==0).mean():.3f}%")
        log("#" * 78)

        arms = GRID_FULL + DIAGNOSTICS + CONTROLS if cfg == "full" else GRID_CONFOUND
        if args.arms:
            arms = [a for a in args.arms.split(",")]
        if args.quick and not args.arms:
            arms = ["all", "all_minus_fractal"]

        for scheme, fold, tr, va, te in folds:
            tr, va, te = tr & cmask, va & cmask, te & cmask
            if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
                log(f"  [skip] {cfg}/{scheme}/{fold}: an empty partition")
                continue
            assert_disjoint(df, tr, va, te, f"{cfg}/{scheme}/{fold}")
            ytr, yva, yte = yall[tr], yall[va], yall[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2 or len(np.unique(yte)) < 2:
                log(f"  [skip] {cfg}/{scheme}/{fold}: a partition holds one class")
                continue

            log(f"\n  --- {cfg} / {scheme} / {fold}: "
                f"train {tr.sum():,} val {va.sum():,} test {te.sum():,} | "
                f"human prevalence  train {100*(ytr==0).mean():.3f}%  "
                f"val {100*(yva==0).mean():.3f}%  test {100*(yte==0).mean():.3f}%")
            log(f"      (D-11: val prevalence is REPORTED, not asserted equal to "
                f"corpus prevalence -- splits.py assigns by hash and calls "
                f"per-domain balance an outcome, not a constraint)")

            te_idx = np.flatnonzero(te)
            for arm in arms:
                for learner in (["lgbm", "logreg"]
                                if arm in ("all", "all_minus_fractal")
                                else ["lgbm"]):
                    if arm in DIAGNOSTICS:
                        if learner != "lgbm":
                            continue
                        Xa, names = diagnostic_matrix(df, arm, feats_all)
                    elif arm == "label_shuffle":
                        if learner != "lgbm":
                            continue
                        results["controls"].append(
                            label_shuffle_control(df, groups, tr, te, yall,
                                                  cfg, scheme, fold, log))
                        continue
                    else:
                        names = arm_features(arm, groups)
                        Xa = df[names].to_numpy(float)

                    Xtr, Xva, Xte = Xa[tr], Xa[va], Xa[te]
                    imp_note = "native NaN"
                    if cfg == "no_nan_signal" and arm not in DIAGNOSTICS:
                        Xtr, (Xva, Xte) = impute_draw(Xtr, [Xva, Xte],
                                                      np.random.default_rng(SEED))
                        imp_note = "random draw from train-fold distribution"

                    t1 = time.time()
                    p, base = fit_predict(learner, Xtr, ytr, Xva, yva, Xte, names)
                    # Measured 2026-08-24: a 2000-fold stratified bootstrap
                    # over a 43k-row test fold costs ~7.9 s (AUROC) / 6.3 s
                    # (TPR@1%) / 6.5 s (AP) / 2.5 s (macro-F1) -- ~30 s for all
                    # five. The design doc's S7.2 restriction to four arms was
                    # budgeted against an unmeasured guess; every cell gets a CI
                    # instead, so no table carries a blank that reads as an
                    # omission. Struck and replaced, not silently widened.
                    with_ci = True
                    cell = {"config": cfg, "scheme": scheme, "fold": fold,
                            "arm": arm, "learner": learner,
                            "n_features": Xa.shape[1],
                            "imputation": imp_note,
                            "n_train": int(tr.sum()), "n_val": int(va.sum()),
                            "prevalence_human_train": float((ytr == 0).mean()),
                            "prevalence_human_val": float((yva == 0).mean()),
                            "fit_seconds": round(time.time() - t1, 1)}
                    cell.update(evaluate(yte, p, with_ci))
                    results["cells"].append(cell)
                    pt = cell["point"]
                    log(f"      {arm:<20s} {learner:<7s} n_feat={Xa.shape[1]:<3d} "
                        f"AUROC={pt['auroc']:.4f}  TPR@1%={pt['tpr_at_1pct_fpr']:.4f}  "
                        f"TPR@5%={pt['tpr_at_5pct_fpr']:.4f}  "
                        f"AP_h={pt['ap_human']:.4f}  F1m={pt['macro_f1_at_0.5_calibrated']:.4f}  "
                        f"ECE={pt['ece_calibrated']:.4f}  "
                        f"[{cell['fit_seconds']}s{' +CI' if with_ci else ''}]")

                    # store predictions + the test row indices for BOTH the
                    # within-config (all vs all-fractal) and the cross-config
                    # (config c vs config 1) paired comparisons. A cross-config
                    # comparison is only paired on the rows both configs kept.
                    results.setdefault("_pred", {})[
                        f"{cfg}|{scheme}|{fold}|{arm}|{learner}"] = p
                    results.setdefault("_idx", {})[f"{cfg}|{scheme}|{fold}"] = te_idx
                    results.setdefault("_auc", {})[
                        f"{cfg}|{scheme}|{fold}|{arm}|{learner}"] = cell["point"]["auroc"]

                    # D-13: every declared feature's importance, failures included
                    if (learner == "lgbm" and arm == "all"
                            and cfg == "full" and scheme == "random"):
                        results["importance"]["lgbm_gain"] = dict(
                            zip(names, [float(x) for x in
                                        base.booster_.feature_importance("gain")]))
                        results["importance"]["lgbm_split"] = dict(
                            zip(names, [int(x) for x in
                                        base.booster_.feature_importance("split")]))
                        # Implementation-independent cross-check on the same
                        # test fold. Gain is LightGBM's own accounting; if the
                        # two orderings disagree badly, the gain ranking is an
                        # artifact of the splitter and the report says so.
                        t2 = time.time()
                        pi = permutation_importance(
                            base, Xte, yte, scoring="roc_auc", n_repeats=5,
                            random_state=SEED, n_jobs=1)
                        results["importance"]["permutation_auroc_drop"] = dict(
                            zip(names, [float(x) for x in pi.importances_mean]))
                        results["importance"]["permutation_auroc_drop_std"] = dict(
                            zip(names, [float(x) for x in pi.importances_std]))
                        log(f"      [importance] gain + split counts for all "
                            f"{len(names)} declared inputs (D-13: failures "
                            f"included); permutation cross-check "
                            f"{time.time()-t2:.0f}s")

            # --- config 3: did the imputation ablate, or merely relocate? ---
            if cfg == "no_nan_signal" and not args.arms:
                r = leak_check(df, tr, te, arm_features("all", groups), was_missing)
                r.update({"scheme": scheme, "fold": fold})
                r["imputation_detectability"] = imputation_detectability(
                    df, tr, te, arm_features("all", groups))
                results["leak_checks"].append(r)
                if "skipped" in r:
                    log(f"      [leak check] skipped: {r['skipped']}")
                else:
                    verdict = ("PASS" if r["excess_over_floor"] is not None
                               and r["excess_over_floor"] <= 0.02 else "FAIL")
                    log(f"      [leak check] was_missing AUROC -- "
                        f"floor(never-missing cols only)={r['auroc_floor_never_missing_cols']:.4f}  "
                        f"random-draw={r['auroc_random_draw_imputed']:.4f}  "
                        f"median-control={r['auroc_median_imputed_positive_control']:.4f}  "
                        f"excess over floor={r['excess_over_floor']:+.4f}  -> {verdict}")
                    idt = r.get("imputation_detectability", {})
                    if "auroc_random_draw" in idt:
                        log(f"      [imputation detectability -- no mechanism confound] "
                            f"real vs mask-then-impute on COMPLETE rows only: "
                            f"random-draw AUROC={idt['auroc_random_draw']:.4f}  "
                            f"median-control AUROC={idt['auroc_median_positive_control']:.4f}  "
                            f"(0.5 = an imputed value is indistinguishable from a real one)")
                    log(f"                   (the median arm is the POSITIVE CONTROL: "
                        f"it is expected to sit far above the floor, which is what "
                        f"proves this check can fail)")

    # --- paired comparisons ---------------------------------------------
    # Every difference is reported with an interval. "Config 2 ~ config 1" is
    # only allowed to mean THE PAIRED CI INCLUDES ZERO -- never "the point
    # estimates looked close".
    preds = results.pop("_pred", {})
    idxs = results.pop("_idx", {})
    aucs = results.pop("_auc", {})

    # ALIGNMENT CHECK, and it can fail. Recompute each cell's AUROC from the
    # STORED prediction vector and the STORED test-row indices. If either got
    # misaligned with what evaluate() scored, these disagree. label_shuffle
    # cannot see this class of bug -- a model trained on noise emits
    # near-constant scores, so a misalignment would be invisible there.
    for k, pv in preds.items():
        cfg_, sch_, fold_ = k.split("|")[:3]
        ti = idxs[f"{cfg_}|{sch_}|{fold_}"]
        if len(ti) != len(pv):
            raise SystemExit(f"[FATAL] {k}: stored predictions ({len(pv):,}) and "
                             f"stored test indices ({len(ti):,}) differ in length")
        re_auc = auroc(yall[ti], pv)
        if abs(re_auc - aucs[k]) > 1e-9:
            raise SystemExit(
                f"[FATAL] {k}: AUROC recomputed from the stored prediction/index "
                f"pair is {re_auc:.10f} but evaluate() reported {aucs[k]:.10f}. "
                f"The stored vectors are misaligned with what was scored; every "
                f"paired comparison built on them would be wrong.")
    log(f"[align] {len(preds)} stored prediction vectors reproduce their own "
        f"reported AUROC exactly (max |delta| < 1e-9)")

    # (a) within-config: All vs All-fractal, the P3 exit-gate comparison
    for key in list(preds):
        if not key.endswith("|all|lgbm"):
            continue
        base_key = key.replace("|all|lgbm", "|all_minus_fractal|lgbm")
        if base_key not in preds:
            continue
        cfg, scheme, fold = key.split("|")[:3]
        te_idx = idxs[f"{cfg}|{scheme}|{fold}"]
        yte = yall[te_idx]
        a, b = preds[key], preds[base_key]
        auc_a, auc_b, z, pv = delong_test(yte, a, b)
        obs, lo, hi, k = bootstrap_ci_paired(tpr_at_1pct_fpr, yte, a, b,
                                             n_boot=N_BOOT, seed=BOOT_SEED)
        oa, la, ha, ka = bootstrap_ci_paired(auroc, yte, a, b,
                                             n_boot=N_BOOT, seed=BOOT_SEED)
        agree = (pv < 0.05) == (la > 0 or ha < 0)
        results["paired"].append(
            {"kind": "within_config__all_vs_all_minus_fractal",
             "config": cfg, "scheme": scheme, "fold": fold,
             "auroc_all": auc_a, "auroc_all_minus_fractal": auc_b,
             "delong_z": z, "delong_p": pv,
             "d_auroc": oa, "d_auroc_lo": la, "d_auroc_hi": ha,
             "d_auroc_excludes_zero": bool(la > 0 or ha < 0),
             "tpr1_diff": obs, "tpr1_diff_lo": lo, "tpr1_diff_hi": hi,
             "tpr1_diff_excludes_zero": bool(lo > 0 or hi < 0),
             "tpr1_n_valid_folds": k,
             "delong_and_paired_bootstrap_agree_on_significance": bool(agree)})
        log(f"  [paired within] {cfg}/{scheme}/{fold}  AUROC all={auc_a:.4f} "
            f"all-fractal={auc_b:.4f}  DeLong z={z:+.3f} p={pv:.3g}  "
            f"dAUROC={oa:+.4f} [{la:+.4f}, {ha:+.4f}]  "
            f"dTPR@1%={obs:+.4f} [{lo:+.4f}, {hi:+.4f}]  agree={agree}")

    # (b) cross-config: config c vs config 1 (`full`), same arm, same fold.
    # Configs 2 and 4 drop rows, so the comparison is paired on the INTERSECTION
    # of the two test sets -- comparing a metric computed on different row sets
    # is not a paired test and the difference would absorb the population change.
    for key in sorted(preds):
        cfg, scheme, fold, arm, learner = key.split("|")
        if cfg == "full" or learner != "lgbm":
            continue
        ref = f"full|{scheme}|{fold}|{arm}|lgbm"
        if ref not in preds:
            continue
        i_c = idxs[f"{cfg}|{scheme}|{fold}"]
        i_f = idxs[f"full|{scheme}|{fold}"]
        common, pc, pf = np.intersect1d(i_c, i_f, return_indices=True)
        if len(common) < 500:
            continue
        yc = yall[common]
        if len(np.unique(yc)) < 2:
            continue
        a = preds[key][pc]          # config c, on shared rows
        b = preds[ref][pf]          # config 1, on the same shared rows
        rec = {"kind": "cross_config__vs_full", "config": cfg, "scheme": scheme,
               "fold": fold, "arm": arm, "n_shared_test_rows": int(len(common)),
               "frac_of_full_test": float(len(common) / len(i_f))}
        for mname, mfn in (("auroc", auroc), ("tpr_at_1pct_fpr", tpr_at_1pct_fpr),
                           ("ap_human", ap_human)):
            o, lo, hi, k = bootstrap_ci_paired(mfn, yc, a, b, n_boot=N_BOOT,
                                               seed=BOOT_SEED)
            rec[f"d_{mname}"] = o
            rec[f"d_{mname}_lo"] = lo
            rec[f"d_{mname}_hi"] = hi
            rec[f"d_{mname}_excludes_zero"] = bool(lo > 0 or hi < 0)
            rec[f"d_{mname}_n_valid_folds"] = k
        results["paired"].append(rec)
        log(f"  [paired cross] {cfg} vs full / {scheme}/{fold}/{arm}  "
            f"shared n={len(common):,} ({100*rec['frac_of_full_test']:.1f}% of full test)  "
            f"dAUROC={rec['d_auroc']:+.5f} [{rec['d_auroc_lo']:+.5f}, {rec['d_auroc_hi']:+.5f}]"
            f"{'  EXCLUDES 0' if rec['d_auroc_excludes_zero'] else '  includes 0'}  "
            f"dTPR@1%={rec['d_tpr_at_1pct_fpr']:+.4f} "
            f"[{rec['d_tpr_at_1pct_fpr_lo']:+.4f}, {rec['d_tpr_at_1pct_fpr_hi']:+.4f}]"
            f"{'  EXCLUDES 0' if rec['d_tpr_at_1pct_fpr_excludes_zero'] else '  includes 0'}")

    # --- Part B: prevalence disclosure ------------------------------------
    results["prevalence_disclosure"] = {
        "corpus_human_prevalence": float((yall == 0).mean()),
        "note": (
            "Calibrated probabilities are calibrated TO THE VALIDATION SPLIT'S "
            "OWN PREVALENCE, not to corpus prevalence and not to any deployment "
            "prevalence. The three differ: splits are group-disjoint and assigned "
            "by a hash of the (domain, dataset_id) key, so per-split class "
            "balance is an OUTCOME of the hash, not a constraint (splits.py "
            "docstring). NO prior-shift correction was applied. Reason: the "
            "calibrator is refit per (config, scheme) on that scheme's own "
            "validation split and every reported metric is computed on the test "
            "split of the SAME scheme, so calibration and evaluation sit at "
            "essentially the same prevalence -- correcting to corpus prevalence "
            "would move the probabilities away from the population they are "
            "evaluated on. Deployment prevalence is unknown and is not any of "
            "these three; a deployer must recalibrate. Per-fold train/val/test "
            "prevalences are listed below and in every results row."),
        "per_fold": [{"config": c["config"], "scheme": c["scheme"],
                      "fold": c["fold"],
                      "train": c["prevalence_human_train"],
                      "val": c["prevalence_human_val"],
                      "test": c["prevalence_human"]}
                     for c in results["cells"]
                     if c["arm"] == "all" and c["learner"] == "lgbm"]}

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    log(f"\n[done] {len(results['cells'])} cells -> {args.out}  "
        f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
