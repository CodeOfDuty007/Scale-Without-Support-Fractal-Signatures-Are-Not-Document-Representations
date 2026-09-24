#!/usr/bin/env python3
"""P4 task 5, tabular LightGBM detector -- CPU-only feature build on the
edited-variant text, reusing P1/P3's own feature functions rather than
reimplementing them (design decision, 2026-08-27): `features_for` from
`build_features.py` (fractal group: S/H point estimates, R2, CI, 5-scale
residuals, window summaries, per scorer -- D-13's exact declared set),
`family_b_feature_summary` from `fractalsig/pipeline/build_p1_feature_store_gpu.py` (stat
group: LogPPL/GLTR/entropy, per scorer -- the rescore arrays already store
log-probs/ranks/entropies, job206, no new GPU forward pass), `family_c_style`
(style group, text-only) and `embedding_dispersion` (needs the MiniLM
embedder only -- `load_models(scorers=())` loads just that, no Gemma/Mistral
resident) from the same two files. Cross_PPL_Ratio and Scorer_Divergence_S/H
match `fractalsig/pipeline/build_p1_join_feature_store.py`'s exact formulas.

Total-token reconstruction for word_budget/embedding_dispersion: the rescore
files store `window=[start,end]` from `compute_align_window`, not the raw
total. Reconstructed: window[0]==0 -> total=window[1] (exact, doc was under
CLIP_TO); window[0]==WARMUP_DROP -> total>=CLIP_TO, approximated as CLIP_TO
(documented approximation -- P1's own word_budget only branches on the
>=CLIP_TO boolean plus a ratio, insensitive to the exact overflow amount).

Scores the population through the persisted full/random/random/all LightGBM
model (`feature_store/p3_refit_models/full_random_random_all_lgbm.joblib`):
both the calibrated probability (`model`) for recall-at-threshold reporting,
matching every other detector in this battery, and the raw margin (`base_model`,
raw_score=True) for the record.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys

import joblib
import numpy as np

HERE = _paths.WORKSPACE_STR

from fractalsig.detection.build_features import features_for  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import family_c_style, compute_word_budget  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import (family_b_feature_summary, embedding_dispersion,  # noqa: E402
                                          load_models, WARMUP_DROP, CLIP_TO)

DATA = lambda *p: os.path.join(HERE, "data", *p)


def reconstruct_total(window):
    if window is None:
        return None
    start, end = window
    if start == 0:
        return end
    return CLIP_TO  # start==WARMUP_DROP branch -- documented approximation, see module docstring


def build_feature_row(text, gemma_rec, mistral_rec, embedder, device):
    feats = {}
    gemma_scores = np.array(gemma_rec["scores"])
    mistral_scores = np.array(mistral_rec["scores"])

    feats.update(features_for(gemma_scores, "Gemma"))
    feats.update(features_for(mistral_scores, "Mistral"))
    for e in ("S", "H"):
        g, m = feats.get(f"Gemma_{e}"), feats.get(f"Mistral_{e}")
        feats[f"Scorer_Divergence_{e}"] = float(abs(g - m)) if g == g and m == m else float("nan")

    b_gemma = family_b_feature_summary({"log_probs": -gemma_scores, "ranks": gemma_rec["ranks"],
                                          "entropies": gemma_rec["entropies"]})
    b_mistral = family_b_feature_summary({"log_probs": -mistral_scores, "ranks": mistral_rec["ranks"],
                                            "entropies": mistral_rec["entropies"]})
    for k, v in b_gemma.items():
        if k != "Feature_Window_N_Scored":
            feats[f"Gemma_{k}"] = v
    for k, v in b_mistral.items():
        if k != "Feature_Window_N_Scored":
            feats[f"Mistral_{k}"] = v

    have_both = feats.get("Gemma_LogPPL_Mean") == feats.get("Gemma_LogPPL_Mean") and \
                feats.get("Mistral_LogPPL_Mean") == feats.get("Mistral_LogPPL_Mean")
    feats["Cross_PPL_Ratio"] = float(np.exp(feats["Gemma_LogPPL_Mean"] - feats["Mistral_LogPPL_Mean"])) \
        if have_both else float("nan")

    gemma_total = reconstruct_total(gemma_rec.get("window"))
    mistral_total = reconstruct_total(mistral_rec.get("window"))
    ref_total = gemma_total or mistral_total
    word_budget = compute_word_budget(text, ref_total)
    feats.update(family_c_style(text, word_budget))
    feats.pop("Family_C_Full_Text_Fallback", None)  # not one of the 85 model features
    # ZWSP_Present_In_Source_Text (added 2026-08-29) is DELIBERATELY KEPT,
    # unlike the line above -- RAID row metadata, not a model feature, but
    # far more informative here than on GAGLE (RAID has a real
    # zero_width_space attack; GAGLE never does). Downstream consumers that
    # auto-detect feature columns from this row's keys must exclude it by
    # name or by boolean dtype -- see p5_task6_generator_family_id.py and
    # p5_exit_gate_raid.py for the two places this was already handled.

    feats["Embedding_Dispersion"] = embedding_dispersion(embedder, text, gemma_total, mistral_total)
    return feats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args()

    bundle = joblib.load(os.path.join(HERE, "feature_store", "p3_refit_models", "full_random_random_all_lgbm.joblib"))
    feature_names = bundle["feature_names"]
    cal_model = bundle["model"]
    base_model = bundle["base_model"]

    _, device, _, _, _, embedder = load_models(scorers=())
    print(f"[tabular] embedder loaded on device={device}", flush=True)

    rescore = {}
    for scorer in ("gemma", "mistral"):
        rescore[scorer] = {}
        for batch in (1, 2):
            for l in open(DATA(f"p4_rescore_batch{batch}_{scorer}.jsonl")):
                r = json.loads(l)
                rescore[scorer][(r["id"], r["variant"])] = r

    pop = [json.loads(l) for l in open(DATA("p4_task5_population.jsonl"))]
    if args.limit:
        pop = pop[:args.limit]
    out_path = DATA("p4_task5_tabular_scores.jsonl")
    if args.num_shards:
        pop = pop[args.shard_index::args.num_shards]
        out_path = DATA(f"p4_task5_tabular_scores.shard{args.shard_index}.jsonl")
    print(f"[tabular] {len(pop)} population rows to score -> {out_path}", flush=True)

    out = open(out_path, "w")
    n_ok, n_err = 0, 0
    for i, row in enumerate(pop):
        gemma_rec = rescore["gemma"].get((row["id"], row["variant"]))
        mistral_rec = rescore["mistral"].get((row["id"], row["variant"]))
        if gemma_rec is None or mistral_rec is None:
            n_err += 1
            continue
        try:
            feats = build_feature_row(row["text"], gemma_rec, mistral_rec, embedder, device)
            X = np.array([[feats.get(n, np.nan) for n in feature_names]], dtype=float)
            pos = int(np.where(cal_model.classes_ == 1)[0][0])
            p_cal = float(cal_model.predict_proba(X)[0, pos])
            margin = float(base_model.predict(X, raw_score=True)[0])
            out.write(json.dumps({
                "id": row["id"], "variant": row["variant"], "domain": row["domain"],
                "generator": row["generator"], "separate_bucket": row["separate_bucket"],
                "calibrated_prob": p_cal, "raw_margin": margin,
            }) + "\n")
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"[tabular] ERROR row {row['id']}/{row['variant']}: {e}", flush=True)
        if (i + 1) % 2000 == 0:
            out.flush()
            print(f"[tabular]   {i+1}/{len(pop)}  ok={n_ok} err={n_err}", flush=True)

    out.close()
    print(f"[tabular] DONE  ok={n_ok} err={n_err}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
