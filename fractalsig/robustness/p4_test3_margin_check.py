#!/usr/bin/env python3
"""Test 3 re-check (internal review, 2026-08-27): the calibrated probability
comparison saturated in [0.9995, 1.0000] and had ~0.0004 of dynamic range --
not enough to detect a bias of any real size. This re-runs the survivors-vs-
dropped comparison on the UNCALIBRATED LightGBM raw margin (log-odds,
unbounded) from the same persisted full/random/random/all model instead of
the calibrated p3_refit_predictions.jsonl probability.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix  # noqa: E402

D = lambda *p: os.path.join(HERE, *p)

ALL_TYPES = [
    "paraphrase_light", "paraphrase_heavy", "human_revision",
    "style_transfer", "summarize_then_expand", "sentence_reorder", "typo_noise",
]


def main():
    bundle = joblib.load(D("feature_store", "p3_refit_models", "full_random_random_all_lgbm.joblib"))
    base_model = bundle["base_model"]
    feat_names = bundle["feature_names"]
    print(f"[margin] loaded base_model, {len(feat_names)} features")

    df, groups = load_matrix()
    df = df.set_index("id")

    sample = [json.loads(l) for l in open(D("data", "p4_sampled_docs.jsonl"))]
    sample_ids = [r["id"] for r in sample]
    present = [i for i in sample_ids if i in df.index]
    print(f"[margin] {len(present)}/{len(sample_ids)} P4 sample ids found in feature matrix")

    X = df.loc[present, feat_names].to_numpy(float)
    margin = base_model.predict(X, raw_score=True)
    margin_by_id = dict(zip(present, margin))
    print(f"[margin] raw-margin range: min={margin.min():.3f} max={margin.max():.3f} "
          f"p1={np.percentile(margin,1):.3f} p50={np.percentile(margin,50):.3f} p99={np.percentile(margin,99):.3f}")

    edits1 = {json.loads(l)["id"]: json.loads(l) for l in open(D("data", "p4_edits_batch1.jsonl"))}
    edits2 = {json.loads(l)["id"]: json.loads(l) for l in open(D("data", "p4_edits_batch2.jsonl"))}
    edits = {**edits1, **edits2}

    THRESH = 0.5  # pre-registered here: >0.5 median margin diff (logit units,
    # roughly >12pp probability swing at the operating point) OR non-overlapping
    # IQR constitutes material selection bias on this axis.
    print(f"[margin] pre-registered threshold: median margin diff > {THRESH} logit units, "
          f"or non-overlapping IQR")

    for etype in ALL_TYPES:
        surv_m, drop_m = [], []
        for doc_id, rec in edits.items():
            if doc_id not in margin_by_id:
                continue
            variants = rec.get("variants", {})
            v = variants.get(etype)
            below_min = v.get("below_min_tokens", False) if v else True
            is_empty = v.get("is_empty", False) if v else True
            dropped = (v is None) or below_min or is_empty
            (drop_m if dropped else surv_m).append(margin_by_id[doc_id])
        surv_m = np.array(surv_m)
        drop_m = np.array(drop_m)
        if len(drop_m) == 0:
            print(f"  {etype:22} n_surv={len(surv_m):5} n_drop=0 -- no dropped docs, skip")
            continue
        med_s, med_d = np.median(surv_m), np.median(drop_m)
        iqr_s = (np.percentile(surv_m, 25), np.percentile(surv_m, 75))
        iqr_d = (np.percentile(drop_m, 25), np.percentile(drop_m, 75))
        overlap = not (iqr_s[1] < iqr_d[0] or iqr_d[1] < iqr_s[0])
        diff = med_s - med_d
        verdict = "FAIL (bias)" if (abs(diff) > THRESH or not overlap) else "pass"
        print(f"  {etype:22} n_surv={len(surv_m):5} n_drop={len(drop_m):5}  "
              f"med_surv={med_s:+.3f} med_drop={med_d:+.3f} diff={diff:+.3f}  "
              f"iqr_surv=({iqr_s[0]:+.2f},{iqr_s[1]:+.2f}) iqr_drop=({iqr_d[0]:+.2f},{iqr_d[1]:+.2f}) "
              f"overlap={overlap}  {verdict}")


if __name__ == "__main__":
    main()
