#!/usr/bin/env python3
"""B2 stage 3: score the ladder generations (6 attackers x ~1000 docs,
paraphrase_heavy) with all 5 P4 task5 detectors, reusing each detector's
existing FIXED thresholds (derived once, on the original corpus's val fold,
per this project's established robustness-eval convention -- never
re-derived per attacker, that would answer a different question). One
process, detectors run sequentially, each frees its GPU memory before the
next loads (Gemma for FDG -> RoBERTa retrain -> Falcon pair for Binoculars)
-- same VRAM-management discipline as every other multi-model pass this
project has run on the single 48GB MIG slice.

RoBERTa is the one exception to "reuse existing thresholds": its checkpoint
was never persisted (save_strategy="no", confirmed by p4_task5_roberta.py's
own inspection), so it retrains fresh here too and derives its OWN
thresholds from its own retrain's val fold -- reusing an old retrain's
threshold would compare apples (this retrain's raw probability space) to
oranges (a different retrain's), a real methods error, not a shortcut.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

from fractalsig.detection.balancing import balanced_class_weights, make_weighted_trainer  # noqa: E402


def load_rescore(scorer):
    out = {}
    for l in open(D(f"p4_b2_ladder_rescore_{scorer}.jsonl")):
        r = json.loads(l)
        out[(r["id"], r["attacker"], r["variant"])] = r
    return out


def main():
    generations = [json.loads(l) for l in open(D("p4_b2_ladder_generations.jsonl"))]
    gemma_rescore = load_rescore("gemma")
    mistral_rescore = load_rescore("mistral")
    print(f"[b2-score] {len(generations)} generations, "
          f"{len(gemma_rescore)} gemma rescores, {len(mistral_rescore)} mistral rescores", flush=True)

    attackers = sorted(set(g["attacker"] for g in generations))
    print(f"[b2-score] attackers: {attackers}", flush=True)

    all_results = {}  # detector -> attacker -> {n, r1, r5}

    # ============================================================
    # 1. PERPLEXITY-THRESHOLD (CPU, gemma scorer, reuse existing thresholds)
    # ============================================================
    print("\n[b2-score] === perplexity-threshold ===", flush=True)
    perp_thr = json.load(open(D("p4_task5_perplexity_threshold.json")))["thresholds"]["gemma"]
    by_attacker = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for g in generations:
        rec = gemma_rescore.get((g["id"], g["attacker"], "paraphrase_heavy"))
        if rec is None or not rec["scores"]:
            continue
        s = perp_thr["sign"] * float(np.mean(rec["scores"]))
        d = by_attacker[g["attacker"]]
        d["n"] += 1
        if s >= perp_thr["t1_fpr1pct"]:
            d["r1"] += 1
        if s >= perp_thr["t5_fpr5pct"]:
            d["r5"] += 1
    all_results["perplexity_gemma"] = {a: {"n": d["n"], "recall_1pct": d["r1"] / d["n"] if d["n"] else 0,
                                             "recall_5pct": d["r5"] / d["n"] if d["n"] else 0}
                                        for a, d in by_attacker.items()}
    for a in attackers:
        r = all_results["perplexity_gemma"].get(a, {"n": 0, "recall_1pct": 0, "recall_5pct": 0})
        print(f"[b2-score]   {a:24} n={r['n']:5} recall@1%={r['recall_1pct']:.4f} recall@5%={r['recall_5pct']:.4f}", flush=True)

    # ============================================================
    # 2. TABULAR LightGBM (CPU, reuse build_feature_row + persisted model)
    # ============================================================
    print("\n[b2-score] === tabular LightGBM ===", flush=True)
    import joblib
    from fractalsig.robustness.p4_task5_tabular_features import build_feature_row
    from fractalsig.pipeline.build_p1_feature_store_gpu import load_models as load_scoring_models
    tab_thr = json.load(open(D("p4_task5_tabular_thresholds.json")))
    bundle = joblib.load(os.path.join(HERE, "feature_store", "p3_refit_models", "full_random_random_all_lgbm.joblib"))
    cal_model, feature_names = bundle["model"], bundle["feature_names"]

    torch_mod, device, dtype, tokenizers, models, embedder = load_scoring_models(scorers=())
    by_attacker = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    n_missing_rescore = 0
    for i, g in enumerate(generations):
        grec = gemma_rescore.get((g["id"], g["attacker"], "paraphrase_heavy"))
        mrec = mistral_rescore.get((g["id"], g["attacker"], "paraphrase_heavy"))
        if grec is None or mrec is None or grec.get("skipped_below_min_tokens") or mrec.get("skipped_below_min_tokens"):
            n_missing_rescore += 1
            continue
        feats = build_feature_row(g["text"], grec, mrec, embedder, device)
        x = np.array([[feats.get(n, np.nan) for n in feature_names]], dtype=float)
        if np.isnan(x).any():
            n_missing_rescore += 1
            continue
        p = cal_model.predict_proba(x)[0, list(cal_model.classes_).index(1)]
        d = by_attacker[g["attacker"]]
        d["n"] += 1
        if p >= tab_thr["t1_fpr1pct"]:
            d["r1"] += 1
        if p >= tab_thr["t5_fpr5pct"]:
            d["r5"] += 1
        if (i + 1) % 1000 == 0:
            print(f"[b2-score]   tabular {i+1}/{len(generations)}", flush=True)
    print(f"[b2-score]   {n_missing_rescore} rows excluded (missing/below-min rescore, complete-case per F-5(a))", flush=True)
    all_results["tabular"] = {a: {"n": d["n"], "recall_1pct": d["r1"] / d["n"] if d["n"] else 0,
                                    "recall_5pct": d["r5"] / d["n"] if d["n"] else 0}
                               for a, d in by_attacker.items()}
    for a in attackers:
        r = all_results["tabular"].get(a, {"n": 0, "recall_1pct": 0, "recall_5pct": 0})
        print(f"[b2-score]   {a:24} n={r['n']:5} recall@1%={r['recall_1pct']:.4f} recall@5%={r['recall_5pct']:.4f}", flush=True)

    # ============================================================
    # 3. FAST-DETECTGPT (GPU, gemma-2b curvature)
    # ============================================================
    print("\n[b2-score] === Fast-DetectGPT ===", flush=True)
    from fractalsig.baselines.fast_detectgpt_score import score_documents
    fdg_thr = json.load(open(D("p4_task5_fastdetectgpt_summary.json")))["thresholds"]
    torch_mod2, device2, dtype2, tokenizers2, models2, _ = load_scoring_models(scorers=("gemma-2b",))
    gemma_tok, gemma_model = tokenizers2["gemma-2b"], models2["gemma-2b"]

    by_attacker = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for attacker in attackers:
        rows = [{"id": g["id"], "text": g["text"]} for g in generations if g["attacker"] == attacker]
        scores = score_documents(torch_mod2, gemma_model, gemma_tok, device2, rows)
        d = by_attacker[attacker]
        for idx, sc in scores.items():
            d["n"] += 1
            if sc["d"] >= fdg_thr["t1"]:
                d["r1"] += 1
            if sc["d"] >= fdg_thr["t5"]:
                d["r5"] += 1
        print(f"[b2-score]   {attacker:24} n={d['n']:5} recall@1%={d['r1']/d['n'] if d['n'] else 0:.4f} "
              f"recall@5%={d['r5']/d['n'] if d['n'] else 0:.4f}", flush=True)
    all_results["fastdetectgpt"] = {a: {"n": d["n"], "recall_1pct": d["r1"] / d["n"] if d["n"] else 0,
                                          "recall_5pct": d["r5"] / d["n"] if d["n"] else 0}
                                     for a, d in by_attacker.items()}
    del gemma_model, gemma_tok, torch_mod2
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # 4. ROBERTA (GPU, retrain fresh, threshold from its own val fold)
    # ============================================================
    print("\n[b2-score] === RoBERTa (fresh retrain) ===", flush=True)
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    from sklearn.linear_model import LogisticRegression
    from fractalsig.detection.p3_train_eval import load_matrix, build_folds
    from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr

    SEED = 20260827
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    class TextDS(Dataset):
        def __init__(self, encodings, labels):
            self.encodings, self.labels = encodings, labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    df, groups = load_matrix()
    splits = json.load(open(D("splits_v1.json")))
    fold_lookup = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = fold_lookup["random/random"]
    texts_by_id = {}
    for l in open(D("all_doc_texts.jsonl")):
        r = json.loads(l)
        texts_by_id[r["id"]] = r["text"]
    ids, y = df["id"].to_numpy(), df["y"].to_numpy()

    def build_split(mask):
        idx = np.flatnonzero(mask)
        return ids[idx], y[idx], [texts_by_id[i] for i in ids[idx]]

    tr_ids, tr_y, tr_texts = build_split(tr)
    va_ids, va_y, va_texts = build_split(va)
    te_ids, te_y, te_texts = build_split(te)

    MODEL_ID, MAX_LEN = "roberta-base", 512
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    def encode(texts):
        return tok(texts, truncation=True, max_length=MAX_LEN, padding=False)

    t0 = time.time()
    tr_enc, va_enc, te_enc = encode(tr_texts), encode(va_texts), encode(te_texts)
    print(f"[b2-score]   tokenized in {time.time()-t0:.0f}s", flush=True)
    tr_ds, va_ds, te_ds = TextDS(tr_enc, tr_y.tolist()), TextDS(va_enc, va_y.tolist()), TextDS(te_enc, te_y.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    # De-duplicated 2026-08-29 (code review) into fractalsig/detection/balancing.py.
    w_neg, w_pos = balanced_class_weights(tr_y)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)

    WeightedTrainer = make_weighted_trainer(class_weights)

    targs = TrainingArguments(
        output_dir=D("..", "roberta_b2_tmp_ckpt"), num_train_epochs=2.0,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, warmup_ratio=0.06, weight_decay=0.01, logging_steps=200,
        save_strategy="no", eval_strategy="no", report_to=[], seed=SEED,
        fp16=torch.cuda.is_available(), dataloader_num_workers=2, max_grad_norm=1.0)
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=tr_ds,
                              data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    trainer.train()
    print(f"[b2-score]   training done in {time.time()-t0:.0f}s", flush=True)

    def predict_logits(ds):
        return trainer.predict(ds).predictions

    va_logits, te_logits = predict_logits(va_ds), predict_logits(te_ds)
    va_p1_raw = torch.softmax(torch.tensor(va_logits), dim=-1)[:, 1].numpy()
    te_p1_raw = torch.softmax(torch.tensor(te_logits), dim=-1)[:, 1].numpy()
    a_auroc = auroc(te_y, te_p1_raw)
    print(f"[b2-score]   PRE-ATTACK (this retrain): AUROC={a_auroc:.4f} "
          f"TPR@1%={tpr_at_1pct_fpr(te_y, te_p1_raw):.4f} TPR@5%={tpr_at_5pct_fpr(te_y, te_p1_raw):.4f}", flush=True)

    fpr, tpr_, thresh = roc_curve(va_y, va_p1_raw)
    idx1, idx5 = np.where(fpr <= 0.01)[0], np.where(fpr <= 0.05)[0]
    rb_t1 = thresh[idx1[-1]] if len(idx1) else None
    rb_t5 = thresh[idx5[-1]] if len(idx5) else None

    by_attacker = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    for attacker in attackers:
        rows = [g for g in generations if g["attacker"] == attacker]
        enc = encode([g["text"] for g in rows])
        ds = TextDS(enc, [1] * len(rows))
        logits = predict_logits(ds)
        p1 = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        d = by_attacker[attacker]
        d["n"] = len(rows)
        d["r1"] = int((p1 >= rb_t1).sum())
        d["r5"] = int((p1 >= rb_t5).sum())
        print(f"[b2-score]   {attacker:24} n={d['n']:5} recall@1%={d['r1']/d['n']:.4f} recall@5%={d['r5']/d['n']:.4f}", flush=True)
    all_results["roberta"] = {a: {"n": d["n"], "recall_1pct": d["r1"] / d["n"] if d["n"] else 0,
                                    "recall_5pct": d["r5"] / d["n"] if d["n"] else 0}
                               for a, d in by_attacker.items()}
    all_results["roberta"]["_pre_attack_this_retrain"] = {
        "auroc": float(a_auroc), "recall_1pct": float(tpr_at_1pct_fpr(te_y, te_p1_raw)),
        "recall_5pct": float(tpr_at_5pct_fpr(te_y, te_p1_raw))}
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # 5. BINOCULARS (GPU, Falcon pair, ~28GB, reuse existing sign/thresholds)
    # ============================================================
    print("\n[b2-score] === Binoculars ===", flush=True)
    from fractalsig.baselines.binoculars_score import load_falcon_pair, score_batch_binoculars, compute_window
    bino_summary = json.load(open(D("p4_task5_binoculars_summary.json")))
    sign, bino_thr = bino_summary["sign"], bino_summary["thresholds"]
    device3 = "cuda" if torch.cuda.is_available() else "cpu"
    bino_tok, observer, performer = load_falcon_pair(device3)

    usable = []
    for g in generations:
        ids_ = bino_tok(g["text"])["input_ids"]
        window = compute_window(len(ids_))
        if window is None:
            continue
        usable.append((g, window))
    usable.sort(key=lambda x: x[1][1])

    by_attacker = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    pos, bs = 0, 4
    while pos < len(usable):
        batch = usable[pos:pos + bs]
        texts = [b[0]["text"] for b in batch]
        windows = [b[1] for b in batch]
        try:
            results = score_batch_binoculars(bino_tok, observer, performer, device3, texts, windows)
        except RuntimeError as e:
            if "memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                continue
            raise
        for (g, window), r in zip(batch, results):
            if r is None:
                continue
            sc = sign * r["score"]
            d = by_attacker[g["attacker"]]
            d["n"] += 1
            if sc >= bino_thr["t1"]:
                d["r1"] += 1
            if sc >= bino_thr["t5"]:
                d["r5"] += 1
        pos += len(batch)
        if pos % 400 < bs:
            print(f"[b2-score]   binoculars {pos}/{len(usable)}", flush=True)
    all_results["binoculars"] = {a: {"n": d["n"], "recall_1pct": d["r1"] / d["n"] if d["n"] else 0,
                                       "recall_5pct": d["r5"] / d["n"] if d["n"] else 0}
                                  for a, d in by_attacker.items()}
    for a in attackers:
        r = all_results["binoculars"].get(a, {"n": 0, "recall_1pct": 0, "recall_5pct": 0})
        print(f"[b2-score]   {a:24} n={r['n']:5} recall@1%={r['recall_1pct']:.4f} recall@5%={r['recall_5pct']:.4f}", flush=True)

    with open(D("p4_b2_ladder_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[b2-score] written to data/p4_b2_ladder_results.json")


if __name__ == "__main__":
    main()
