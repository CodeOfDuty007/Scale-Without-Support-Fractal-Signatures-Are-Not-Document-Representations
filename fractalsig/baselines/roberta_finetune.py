#!/usr/bin/env python3
"""RoBERTa baseline: fine-tune roberta-base on ONE fold, evaluate on that
fold's test split, persist per-row {id, y, score} in the same format as
p3_refit_predictions.jsonl and perplexity_threshold_baseline.json (score-once
pattern; joins by id, not index).

D-11 sequence, matched to the LightGBM harness for a fair comparison: class
weighting during training -> calibrate on val at true prevalence (sigmoid,
fit on the val split's raw logits) -> report calibrated probabilities.

One fold per invocation (--scheme --fold) so the 8 folds can be submitted as
independent SLURM jobs and run in parallel rather than serialized.
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

D = lambda *p: os.path.join(HERE, *p)
MODEL_ID = "roberta-base"
MAX_LEN = 512
SEED = 20260825
OUT_PRED = D("data", "roberta_predictions.jsonl")
OUT_SUMMARY_DIR = D("data", "roberta_summaries")
CKPT_ROOT = D("data", "roberta_checkpoints")


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
    with open(D("data", "all_doc_texts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            texts[r["id"]] = r["text"]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--subsample-train", type=int, default=None,
                    help="cap on training rows, for a fast smoke test")
    ap.add_argument("--no-fp16", action="store_true",
                    help="disable fp16 (2026-08-25: big_patent/wikipedia LODO folds showed "
                         "near-constant score saturation under fp16 with high class weight; "
                         "testing whether fp32 resolves it)")
    ap.add_argument("--cap-weight", type=float, default=None,
                    help="cap w_neg at this value (2026-08-25: capping was tested on "
                         "big_patent and REFUTED as the fix -- it made the score distribution "
                         "MORE degenerate, not less. Kept as a flag for reference, not "
                         "recommended; likely the relationship is inverted -- weight was "
                         "acting against majority-class collapse, not causing it.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="explicit gradient-clipping threshold, passed through to "
                         "TrainingArguments. NOTE: this equals the HF default and was "
                         "already active in every prior run (transformers 5.14.1's "
                         "TrainingArguments.max_grad_norm defaults to 1.0, unset in earlier "
                         "invocations). Exposed here for explicit documentation only -- the "
                         "logged 480.9 grad-norm spike is torch.nn.utils.clip_grad_norm_'s "
                         "PRE-clip total norm (its documented return value), not evidence "
                         "clipping was off.")
    ap.add_argument("--diag-suffix", type=str, default=None,
                    help="write predictions/summary to SEPARATE diagnostic-only files "
                         "(data/roberta_predictions_<suffix>.jsonl, "
                         "roberta_summaries/<scheme>_<fold>_<suffix>.json) instead of the "
                         "shared canonical paths. Use for any exploratory/diagnostic run so "
                         "it can never silently overwrite an already-landed, citable result.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="explicit seed, logged in the summary. 2026-08-25: job 161 and the "
                         "original big_patent run used the identical NOMINAL seed and weight "
                         "and still produced different final models (std 0.000005 vs "
                         "0.0000036-but-healthy-margin) -- CUDA/cuDNN non-determinism in fp16 "
                         "training means fixing the seed does not guarantee a reproducible "
                         "run. Multi-seed reruns use this flag to get DISTINCT, logged seeds "
                         "per run rather than relying on nominal-seed reproducibility.")
    ap.add_argument("--fp32-inference", action="store_true",
                    help="run trainer.predict() in fp32 even if training used fp16. "
                         "2026-08-25: job 161 found the big_patent FPR thresholds sitting on "
                         "fp16 quantization artifacts in logit-margin space (an exact tie at "
                         "the 1%-FPR rank, a gap at 5%-FPR matching one fp16 ULP at that "
                         "magnitude) -- training precision was already ruled out as the cause "
                         "of the underlying instability, but inference-side fp16 may still be "
                         "adding a separate, nearly-free-to-fix precision floor onto the "
                         "extreme-quantile TPR@FPR metrics specifically.")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(OUT_SUMMARY_DIR, exist_ok=True)
    tag = f"{args.scheme}_{args.fold}"
    if args.diag_suffix:
        tag = f"{tag}_{args.diag_suffix}"
    ckpt_dir = os.path.join(CKPT_ROOT, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    df, groups = load_matrix()
    splits = json.load(open(D("data", "splits_v1.json")))
    fold_lookup = {f"{s}/{f}": (tr, va, te) for s, f, tr, va, te in build_folds(df, splits)}
    key = f"{args.scheme}/{args.fold}"
    if key not in fold_lookup:
        print(f"[FATAL] unknown fold {key}. Available: {list(fold_lookup)}")
        sys.exit(1)
    tr, va, te = fold_lookup[key]

    print(f"[roberta] fold {key}: train={tr.sum():,} val={va.sum():,} test={te.sum():,}", flush=True)

    texts_by_id = load_texts()
    ids = df["id"].to_numpy()
    y = df["y"].to_numpy()

    def build_split(mask, cap=None):
        idx = np.flatnonzero(mask)
        if cap is not None and len(idx) > cap:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(idx, size=cap, replace=False)
        split_ids = ids[idx]
        split_y = y[idx]
        split_texts = [texts_by_id[i] for i in split_ids]
        return split_ids, split_y, split_texts

    tr_ids, tr_y, tr_texts = build_split(tr, cap=args.subsample_train)
    va_ids, va_y, va_texts = build_split(va)
    te_ids, te_y, te_texts = build_split(te)
    print(f"[roberta] using train={len(tr_ids):,} val={len(va_ids):,} test={len(te_ids):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def encode(texts):
        return tok(texts, truncation=True, max_length=MAX_LEN, padding=False)

    t0 = time.time()
    # Keep encodings as plain Python int lists (the tokenizer's native
    # output), NOT individually-wrapped torch.Tensor objects -- wrapping
    # each of up to 223,834 documents' input_ids/attention_mask as its own
    # tensor object (as an earlier version of this script did) multiplies
    # per-example Python/tensor overhead by ~3x and was enough to OOM-kill
    # 6 of 8 fold jobs on host memory before training ever started (see
    # 2026-08-25 P3 error analysis session). DataCollatorWithPadding
    # converts to padded tensors per-batch at collate time, which is the
    # only point tensors are actually needed.
    tr_enc = encode(tr_texts)
    va_enc = encode(va_texts)
    te_enc = encode(te_texts)
    print(f"[roberta] tokenized in {time.time()-t0:.0f}s", flush=True)

    tr_ds = TextDS(tr_enc, tr_y.tolist())
    va_ds = TextDS(va_enc, va_y.tolist())
    te_ds = TextDS(te_enc, te_y.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    # class weighting (D-11 step 1) -- inverse-frequency, matching LGB_PARAMS'
    # class_weight="balanced" in spirit. De-duplicated 2026-08-29 (code
    # review) into fractalsig/detection/balancing.py -- this file's own cap_weight clamp,
    # unique to this script, is applied AFTER getting the base weights,
    # before the tensor is built (must stay in that order or the clamp
    # silently drops).
    w_neg, w_pos = balanced_class_weights(tr_y)
    if args.cap_weight is not None and w_neg > args.cap_weight:
        print(f"[roberta] capping w_neg {w_neg:.4f} -> {args.cap_weight:.4f}", flush=True)
        w_neg = args.cap_weight
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    print(f"[roberta] class weights: human(0)={w_neg:.4f} llm(1)={w_pos:.4f}", flush=True)

    WeightedTrainer = make_weighted_trainer(class_weights)

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
        seed=args.seed,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        dataloader_num_workers=2,
        max_grad_norm=args.max_grad_norm,
    )

    trainer = WeightedTrainer(model=model, args=targs, train_dataset=tr_ds,
                              data_collator=DataCollatorWithPadding(tok))

    t0 = time.time()
    trainer.train()
    print(f"[roberta] training done in {time.time()-t0:.0f}s", flush=True)

    if args.fp32_inference and targs.fp16:
        print("[roberta] switching to fp32 for inference (training stays fp16)", flush=True)
        trainer.args.fp16 = False
        trainer.model = trainer.model.float()

    def predict_logits(ds):
        preds = trainer.predict(ds)
        return preds.predictions  # (n, 2)

    va_logits = predict_logits(va_ds)
    te_logits = predict_logits(te_ds)
    va_p1_raw = torch.softmax(torch.tensor(va_logits), dim=-1)[:, 1].numpy()
    te_p1_raw = torch.softmax(torch.tensor(te_logits), dim=-1)[:, 1].numpy()

    # D-11 step 2: calibrate on val at TRUE prevalence -- sigmoid (Platt) on
    # the raw positive-class probability, matching fit_predict's cv="prefit"
    # sigmoid calibration in spirit (1-D input here, not the full estimator).
    calibrator = LogisticRegression()
    calibrator.fit(va_p1_raw.reshape(-1, 1), va_y)
    te_p1_cal = calibrator.predict_proba(te_p1_raw.reshape(-1, 1))[:, 1]

    a = auroc(te_y, te_p1_cal)
    t1 = tpr_at_1pct_fpr(te_y, te_p1_cal)
    t5 = tpr_at_5pct_fpr(te_y, te_p1_cal)
    print(f"[roberta] {key}: n_test={len(te_y):,} AUROC={a:.4f} TPR@1%={t1:.4f} TPR@5%={t5:.4f}", flush=True)

    # Discriminating check (review-identified, 2026-08-25): for LODO, the val
    # split used to fit the 1-D calibrator is OTHER-domain rows while test is
    # the held-out domain -- a calibrator fit on a mismatched distribution
    # can compress scores into a range where the 1%-FPR cutoff lands badly
    # WITHOUT the underlying ranking being as bad as the calibrated TPR@1%
    # implies. AUROC is threshold/monotone-invariant so it can't move between
    # raw and calibrated; TPR@1%FPR can. Report both explicitly.
    a_raw = auroc(te_y, te_p1_raw)
    t1_raw = tpr_at_1pct_fpr(te_y, te_p1_raw)
    t5_raw = tpr_at_5pct_fpr(te_y, te_p1_raw)
    print(f"[roberta] {key} RAW (uncalibrated): AUROC={a_raw:.4f} TPR@1%={t1_raw:.4f} TPR@5%={t5_raw:.4f}", flush=True)
    print(f"[roberta] {key} calibration delta: AUROC {a_raw:.4f}->{a:.4f} (should be ~equal) "
          f"TPR@1% {t1_raw:.4f}->{t1:.4f}  TPR@5% {t5_raw:.4f}->{t5:.4f}", flush=True)

    # Scale-vs-representation diagnostic (2026-08-25, after reframing):
    # softmax(logit_llm - logit_human) can saturate to a near-constant
    # probability even when the underlying logit MARGIN still has real,
    # well-separated variance -- softmax compresses large margins into a
    # tiny float range near 0 or 1. AUROC is rank-invariant so it can't tell
    # these apart; the raw logit margin can. If margin std here is
    # comparable to a healthy fold, the collapse is display-scale only
    # (softmax saturation) and the ranking is real. If margin std is ALSO
    # near zero, the representation itself never separated the classes on
    # this fold -- a genuine learning failure, not a scale artifact.
    te_margin = te_logits[:, 1] - te_logits[:, 0]
    margin_std = float(te_margin.std())
    margin_auroc = auroc(te_y, te_margin)
    margin_t1 = tpr_at_1pct_fpr(te_y, te_margin)
    margin_t5 = tpr_at_5pct_fpr(te_y, te_margin)
    print(f"[roberta] {key} logit-margin diagnostic: std={margin_std:.6f} "
          f"AUROC(margin)={margin_auroc:.4f} (compare to AUROC(prob)={a:.4f} -- "
          f"should match exactly, softmax is monotonic in the margin) "
          f"TPR@1%(margin)={margin_t1:.4f} TPR@5%(margin)={margin_t5:.4f} "
          f"(compare to TPR@1%(prob)={t1:.4f} TPR@5%(prob)={t5:.4f} -- these CAN "
          f"differ even though AUROC can't, because softmax loses resolution "
          f"exactly at the extreme-quantile tail where these are measured)", flush=True)
    margin_diag_path = D("data", f"roberta_logit_margin_diagnostic{'_' + args.diag_suffix if args.diag_suffix else ''}.jsonl")
    with open(margin_diag_path, "a") as f:
        for doc_id, y_, m_ in zip(te_ids, te_y, te_margin):
            f.write(json.dumps({"id": str(doc_id), "y": int(y_), "logit_margin": float(m_),
                                "scheme": args.scheme, "fold": args.fold}) + "\n")

    out_pred_path = D("data", f"roberta_predictions{'_' + args.diag_suffix if args.diag_suffix else ''}.jsonl")
    with open(out_pred_path, "a") as f:
        for doc_id, y_, p_ in zip(te_ids, te_y, te_p1_cal):
            f.write(json.dumps({"id": str(doc_id), "y": int(y_), "score": float(p_),
                                "config": "full", "scheme": args.scheme, "fold": args.fold,
                                "arm": "roberta_base_text", "learner": "roberta"}) + "\n")

    summary = {"scheme": args.scheme, "fold": args.fold, "n_train": len(tr_ids),
              "n_val": len(va_ids), "n_test": len(te_ids),
              "auroc": a, "tpr_at_1pct_fpr": t1, "tpr_at_5pct_fpr": t5,
              "auroc_raw_uncalibrated": a_raw, "tpr_at_1pct_fpr_raw_uncalibrated": t1_raw,
              "tpr_at_5pct_fpr_raw_uncalibrated": t5_raw,
              "class_weights": {"human": w_neg, "llm": w_pos},
              "max_grad_norm": args.max_grad_norm, "learning_rate": 2e-5,
              "seed": args.seed, "fp32_inference": args.fp32_inference,
              "score_std_calibrated": float(te_p1_cal.std()),
              "logit_margin_std": margin_std, "logit_margin_auroc": margin_auroc,
              "logit_margin_tpr_at_1pct_fpr": margin_t1, "logit_margin_tpr_at_5pct_fpr": margin_t5,
              "diag_suffix": args.diag_suffix}
    with open(os.path.join(OUT_SUMMARY_DIR, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[roberta] summary written to {OUT_SUMMARY_DIR}/{tag}.json", flush=True)


if __name__ == "__main__":
    main()
