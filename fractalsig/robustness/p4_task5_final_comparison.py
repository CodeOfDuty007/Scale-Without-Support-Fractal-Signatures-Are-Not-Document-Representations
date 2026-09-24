#!/usr/bin/env python3
"""P4 task 5 close-out: cross-detector comparison table restricted to the
39,773-row intersection all 5 detectors share (Binoculars is the limiting
set -- 37 rows dropped, Falcon-tokenizer-specific, 0.09% of population).
Perplexity-threshold and tabular are dual-scorer (gemma/mistral); this table
reports gemma as the representative column, matching the primary-scorer
convention used elsewhere in P4, with mistral kept in the underlying JSON.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
from collections import defaultdict

import numpy as np

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

TYPES = ["paraphrase_light", "paraphrase_heavy", "style_transfer",
         "sentence_reorder", "typo_noise", "summarize_then_expand"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def main():
    bino_rows = load_jsonl(D("p4_task5_binoculars_scores.jsonl"))
    intersection = {(r["id"], r["variant"]) for r in bino_rows}
    print(f"[final] intersection n={len(intersection)}")

    bino_thr = json.load(open(D("p4_task5_binoculars_summary.json")))["thresholds"]
    bino_by_key = {(r["id"], r["variant"]): r for r in bino_rows}

    tab_rows = load_jsonl(D("p4_task5_tabular_scores.jsonl"))
    tab_by_key = {(r["id"], r["variant"]): r for r in tab_rows}
    tab_thr = json.load(open(D("p4_task5_tabular_thresholds.json")))

    fdg_rows = load_jsonl(D("p4_task5_fastdetectgpt_scores.jsonl"))
    fdg_by_key = {(r["id"], r["variant"]): r for r in fdg_rows}
    fdg_thr = json.load(open(D("p4_task5_fastdetectgpt_summary.json")))["thresholds"]

    rb_rows = load_jsonl(D("p4_task5_roberta_scores.jsonl"))
    rb_by_key = {(r["id"], r["variant"]): r for r in rb_rows}
    rb_thr = json.load(open(D("p4_task5_roberta_summary.json")))["thresholds_raw"]

    pop = load_jsonl(D("p4_task5_population.jsonl"))
    rescore = {}
    for scorer in ("gemma", "mistral"):
        rescore[scorer] = {}
        for batch in (1, 2):
            for l in open(D(f"p4_rescore_batch{batch}_{scorer}.jsonl")):
                r = json.loads(l)
                rescore[scorer][(r["id"], r["variant"])] = r
    perp_thr = json.load(open(D("p4_task5_perplexity_threshold.json")))["thresholds"]["gemma"]

    detectors = {}

    # perplexity (gemma)
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for row in pop:
        key = (row["id"], row["variant"])
        if key not in intersection:
            continue
        rec = rescore["gemma"].get(key)
        if rec is None or not rec["scores"]:
            continue
        s = perp_thr["sign"] * float(np.mean(rec["scores"]))
        d = by_type[row["variant"]]
        d["n"] += 1
        if s >= perp_thr["t1_fpr1pct"]:
            d["r1"] += 1
        if s >= perp_thr["t5_fpr5pct"]:
            d["r5"] += 1
    detectors["perplexity_gemma"] = {t: (d["r1"] / d["n"], d["r5"] / d["n"], d["n"])
                                       for t, d in by_type.items()}

    # tabular
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for key in intersection:
        r = tab_by_key.get(key)
        if r is None:
            continue
        d = by_type[key[1]]
        d["n"] += 1
        if r["calibrated_prob"] >= tab_thr["t1_fpr1pct"]:
            d["r1"] += 1
        if r["calibrated_prob"] >= tab_thr["t5_fpr5pct"]:
            d["r5"] += 1
    detectors["tabular"] = {t: (d["r1"] / d["n"], d["r5"] / d["n"], d["n"])
                              for t, d in by_type.items()}

    # fast-detectgpt
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for key in intersection:
        r = fdg_by_key.get(key)
        if r is None:
            continue
        d = by_type[key[1]]
        d["n"] += 1
        if r["d"] >= fdg_thr["t1"]:
            d["r1"] += 1
        if r["d"] >= fdg_thr["t5"]:
            d["r5"] += 1
    detectors["fastdetectgpt"] = {t: (d["r1"] / d["n"], d["r5"] / d["n"], d["n"])
                                    for t, d in by_type.items()}

    # roberta
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for key in intersection:
        r = rb_by_key.get(key)
        if r is None:
            continue
        d = by_type[key[1]]
        d["n"] += 1
        if r["prob_raw"] >= rb_thr["t1"]:
            d["r1"] += 1
        if r["prob_raw"] >= rb_thr["t5"]:
            d["r5"] += 1
    detectors["roberta"] = {t: (d["r1"] / d["n"], d["r5"] / d["n"], d["n"])
                              for t, d in by_type.items()}

    # binoculars (score_signed already has sign applied at scoring time)
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for key in intersection:
        r = bino_by_key.get(key)
        if r is None:
            continue
        s = r["score_signed"]
        d = by_type[key[1]]
        d["n"] += 1
        if s >= bino_thr["t1"]:
            d["r1"] += 1
        if s >= bino_thr["t5"]:
            d["r5"] += 1
    detectors["binoculars"] = {t: (d["r1"] / d["n"], d["r5"] / d["n"], d["n"])
                                 for t, d in by_type.items()}

    pre_attack = {
        "perplexity_gemma": json.load(open(D("p4_task5_perplexity_threshold.json")))["pre_attack"]["gemma"],
        "tabular": {"recall_at_1pct_fpr": tab_thr["pre_attack_recall_1pct"],
                    "recall_at_5pct_fpr": tab_thr["pre_attack_recall_5pct"]},
        "fastdetectgpt": json.load(open(D("fastdetectgpt_eval.json")))["random/random"],
        "roberta": json.load(open(D("roberta_summaries", "random_random.json"))),
        "binoculars": json.load(open(D("p4_task5_binoculars_summary.json")))["pre_attack"],
    }
    pre_r1 = {
        "perplexity_gemma": pre_attack["perplexity_gemma"]["recall_at_1pct_fpr"],
        "tabular": pre_attack["tabular"]["recall_at_1pct_fpr"],
        "fastdetectgpt": pre_attack["fastdetectgpt"]["tpr_at_1pct_fpr"],
        "roberta": pre_attack["roberta"]["tpr_at_1pct_fpr"],
        "binoculars": pre_attack["binoculars"]["recall_at_1pct_fpr"],
    }
    pre_r5 = {
        "perplexity_gemma": pre_attack["perplexity_gemma"]["recall_at_5pct_fpr"],
        "tabular": pre_attack["tabular"]["recall_at_5pct_fpr"],
        "fastdetectgpt": pre_attack["fastdetectgpt"]["tpr_at_5pct_fpr"],
        "roberta": pre_attack["roberta"]["tpr_at_5pct_fpr"],
        "binoculars": pre_attack["binoculars"]["recall_at_5pct_fpr"],
    }
    FLOOR = 0.15  # pre-attack recall below this = detector never worked on this fold; drops carry no robustness signal

    print(f"\n{'detector':16}{'pre@1%':>9}{'pre@5%':>9}  {'type':22}{'n':>6}{'recall@1%':>12}{'recall@5%':>12}")
    out = {}
    for det, by_type in detectors.items():
        out[det] = {"pre_attack_recall_1pct": pre_r1[det], "pre_attack_recall_5pct": pre_r5[det], "by_type": {}}
        floor_flag = " *FLOOR*" if pre_r1[det] < FLOOR else ""
        for t in TYPES:
            if t not in by_type:
                continue
            r1, r5, n = by_type[t]
            print(f"{det:16}{pre_r1[det]:9.4f}{pre_r5[det]:9.4f}  {t:22}{n:6}{r1:12.4f}{r5:12.4f}{floor_flag}")
            out[det]["by_type"][t] = {"n": n, "recall_at_1pct_fpr": r1, "recall_at_5pct_fpr": r5}

    caption = (
        "Rows marked *FLOOR* (pre-attack recall@1%FPR < 0.15) had near-floor detection "
        "before any attack was applied -- their post-attack numbers cannot support a "
        "robustness claim in either direction (no signal existed for the attack to destroy "
        "or for the attack to have spared). This applies to perplexity_gemma (pre-attack "
        "9.35%@1%FPR) here, matching the same floor-effect caveat already applied to the "
        "fractal_only arm in the sentence_reorder decomposition. fastdetectgpt's paraphrase-"
        "type rows (pre-attack 48.2%@1%FPR, well above floor) show recall INCREASING under "
        "attack (63-86% vs 48% baseline) -- a real, unresolved anomaly, not a floor artifact; "
        "see ledger for the open attacker/scorer same-family question."
    )
    print(f"\n{caption}")

    with open(D("p4_task5_final_comparison.json"), "w") as f:
        json.dump({"intersection_n": len(intersection), "floor_threshold_recall_1pct": FLOOR,
                    "caption": caption, "detectors": out}, f, indent=2)
    print("\nwritten to data/p4_task5_final_comparison.json")


if __name__ == "__main__":
    main()
