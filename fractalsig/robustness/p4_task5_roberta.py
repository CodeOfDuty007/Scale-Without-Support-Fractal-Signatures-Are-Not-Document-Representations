#!/usr/bin/env python3
"""P4 task 5, RoBERTa. `roberta_finetune.py`'s checkpoint was never persisted
(`save_strategy="no"`, deliberate disk-space call across 8 folds x many
seeds) -- confirmed by inspection before writing this, not assumed. So this
retrains on the random/random fold (same D-11 train -> calibrate-on-val
sequence, matching roberta_finetune.py's method exactly) and scores the P4
attacked population in the SAME process, right after training, so the
checkpoint never needs to touch disk. Mandatory pre-attack baseline: the
retrained model's own test-fold recall, cross-checked against the already-
persisted data/roberta_summaries/random_random.json (AUROC=0.9998,
TPR@1%=0.9989, TPR@5%=0.99986) before trusting it.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.linear_model import LogisticRegression

HERE = _paths.WORKSPACE_STR
from fractalsig.detection.p3_train_eval import load_matrix, build_folds  # noqa: E402
from fractalsig.detection.metrics import auroc, tpr_at_1pct_fpr, tpr_at_5pct_fpr  # noqa: E402
from fractalsig.detection.balancing import balanced_class_weights, make_weighted_trainer  # noqa: E402

D = lambda *p: os.path.join(HERE, "data", *p)
MODEL_ID = "roberta-base"
MAX_LEN = 512
SEED = 20260827


class TextDS(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_texts():
    texts = {}
    with open(D("all_doc_texts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            texts[r["id"]] = r["text"]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit-p4", type=int, default=None)
    ap.add_argument("--subsample-train", type=int, default=None)
    args = ap.parse_args()

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    df, groups = load_matrix()
    splits = json.load(open(D("splits_v1.json")))
    fold_lookup = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    tr, va, te = fold_lookup["random/random"]
    print(f"[roberta-p4] fold random/random: train={tr.sum():,} val={va.sum():,} test={te.sum():,}", flush=True)

    texts_by_id = load_texts()
    ids = df["id"].to_numpy()
    y = df["y"].to_numpy()

    def build_split(mask, cap=None):
        idx = np.flatnonzero(mask)
        if cap is not None and len(idx) > cap:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(idx, size=cap, replace=False)
        split_ids = ids[idx]
        split_y = y[idx]
        split_texts = [texts_by_id[i] for i in split_ids]
        return split_ids, split_y, split_texts

    tr_ids, tr_y, tr_texts = build_split(tr, cap=args.subsample_train)
    va_ids, va_y, va_texts = build_split(va)
    te_ids, te_y, te_texts = build_split(te)
    print(f"[roberta-p4] train={len(tr_ids):,} val={len(va_ids):,} test={len(te_ids):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def encode(texts):
        return tok(texts, truncation=True, max_length=MAX_LEN, padding=False)

    t0 = time.time()
    tr_enc = encode(tr_texts)
    va_enc = encode(va_texts)
    te_enc = encode(te_texts)
    print(f"[roberta-p4] tokenized in {time.time()-t0:.0f}s", flush=True)

    tr_ds = TextDS(tr_enc, tr_y.tolist())
    va_ds = TextDS(va_enc, va_y.tolist())
    te_ds = TextDS(te_enc, te_y.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    # De-duplicated 2026-08-29 (code review) into fractalsig/detection/balancing.py.
    w_neg, w_pos = balanced_class_weights(tr_y)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    print(f"[roberta-p4] class weights: human(0)={w_neg:.4f} llm(1)={w_pos:.4f}", flush=True)

    WeightedTrainer = make_weighted_trainer(class_weights)

    ckpt_dir = D("..", "roberta_p4_tmp_ckpt")  # never written to (save_strategy=no)
    targs = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        max_grad_norm=1.0,
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=tr_ds,
                              data_collator=DataCollatorWithPadding(tok))

    t0 = time.time()
    trainer.train()
    print(f"[roberta-p4] training done in {time.time()-t0:.0f}s", flush=True)

    def predict_logits(ds):
        return trainer.predict(ds).predictions

    va_logits = predict_logits(va_ds)
    te_logits = predict_logits(te_ds)
    va_p1_raw = torch.softmax(torch.tensor(va_logits), dim=-1)[:, 1].numpy()
    te_p1_raw = torch.softmax(torch.tensor(te_logits), dim=-1)[:, 1].numpy()

    calibrator = LogisticRegression()
    calibrator.fit(va_p1_raw.reshape(-1, 1), va_y)
    te_p1_cal = calibrator.predict_proba(te_p1_raw.reshape(-1, 1))[:, 1]

    a = auroc(te_y, te_p1_cal)
    t1_recall = tpr_at_1pct_fpr(te_y, te_p1_cal)
    t5_recall = tpr_at_5pct_fpr(te_y, te_p1_cal)
    print(f"[roberta-p4] PRE-ATTACK (retrained): n_test={len(te_y):,} AUROC={a:.4f} "
          f"TPR@1%={t1_recall:.4f} TPR@5%={t5_recall:.4f}", flush=True)
    print(f"[roberta-p4] cross-check vs data/roberta_summaries/random_random.json: "
          f"AUROC=0.9998 TPR@1%=0.9989 TPR@5%=0.99986", flush=True)

    # fixed thresholds on the RAW val probability (pre-calibration input space),
    # same method as every other detector in this battery
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(va_y, va_p1_raw)
    idx1 = np.where(fpr <= 0.01)[0]
    idx5 = np.where(fpr <= 0.05)[0]
    t1 = thresh[idx1[-1]] if len(idx1) else None
    t5 = thresh[idx5[-1]] if len(idx5) else None

    # --- score P4 population with the SAME in-memory model ---
    pop = [json.loads(l) for l in open(D("p4_task5_population.jsonl"))]
    if args.limit_p4:
        pop = pop[:args.limit_p4]
    print(f"[roberta-p4] scoring {len(pop)} P4 population rows", flush=True)
    p4_texts = [r["text"] for r in pop]
    p4_enc = encode(p4_texts)
    p4_ds = TextDS(p4_enc, [1] * len(pop))  # labels unused for prediction
    p4_logits = predict_logits(p4_ds)
    p4_p1_raw = torch.softmax(torch.tensor(p4_logits), dim=-1)[:, 1].numpy()

    from collections import defaultdict
    by_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
    out = open(D("p4_task5_roberta_scores.jsonl"), "w")
    for row, p in zip(pop, p4_p1_raw):
        out.write(json.dumps({"id": row["id"], "variant": row["variant"], "prob_raw": float(p)}) + "\n")
        d = by_type[row["variant"]]
        d["n"] += 1
        if p >= t1:
            d["r1"] += 1
        if p >= t5:
            d["r5"] += 1
    out.close()

    print(f"\n{'type':22} {'n':>6} {'recall@1%FPR':>13} {'recall@5%FPR':>13}")
    results = {"pre_attack": {"recall_1pct": float(t1_recall), "recall_5pct": float(t5_recall),
                                "n": len(te_y), "auroc": float(a)}}
    for t, d in sorted(by_type.items()):
        r1 = d["r1"] / d["n"] if d["n"] else 0
        r5 = d["r5"] / d["n"] if d["n"] else 0
        print(f"{t:22} {d['n']:6} {r1:13.4f} {r5:13.4f}")
        results[t] = {"n": d["n"], "recall_1pct": r1, "recall_5pct": r5}

    with open(D("p4_task5_roberta_summary.json"), "w") as f:
        json.dump({"thresholds_raw": {"t1": float(t1), "t5": float(t5)}, "results": results}, f, indent=2)
    print("\nwritten to data/p4_task5_roberta_summary.json", flush=True)


if __name__ == "__main__":
    main()
