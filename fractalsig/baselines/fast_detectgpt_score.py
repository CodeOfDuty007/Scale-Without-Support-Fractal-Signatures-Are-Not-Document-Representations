#!/usr/bin/env python3
"""Fast-DetectGPT (Bao et al. 2023), analytic/sampling-free variant, self-scored.

Scoring model = sampling model = gemma-2b (the "same model" white-box setup;
the paper's own strongest configuration). Self-scored, not GAGLE-array-based,
for the same reason the rest of Family B is: no candidate sweep, no alignment
gate, 100% coverage modulo MIN_TOKENS.

THE METHOD, adapted to what this pipeline already computes
  Per position t, the model's predictive distribution p(.|x_<t) gives:
    observed  log p(x_t | x_<t)          -- score_batch already returns this
    mu_t    = E_{y~p}[log p(y|x_<t)]     = -entropy_t  (entropy is already
                                            computed by score_batch/family_b's
                                            entropy feature -- this is exact,
                                            not an approximation: entropy is
                                            DEFINED as -sum_v p(v) log p(v))
    sigma_t^2 = Var_{y~p}[log p(y|x_<t)] = E[(log p)^2] - mu_t^2
                                          = sum_v p(v) (log p(v))^2 - entropy_t^2
                                            -- NOT currently computed anywhere
                                            in this pipeline; the one new
                                            quantity this script adds.
  Document score: d(x) = (sum_t log p(x_t) - sum_t mu_t) / sqrt(sum_t sigma_t^2)
  Higher d(x) => more machine-like (observed log-prob sits above the local
  mean of what the model itself would sample), per the paper's convention.

CORRECTNESS TRAP THIS SCRIPT EXISTS TO AVOID (flagged before writing this):
  mu_t and sigma_t^2 MUST be computed over exactly the same window
  score_batch scores (post-WARMUP_DROP, clipped by compute_align_window,
  target range clipped to max(start,1) per F-3's root cause -- see
  build_p1_feature_store_gpu.score_batch's own docstring). This script does
  NOT call score_batch and then separately recompute moments over a
  different range; score_batch_curvature (below) is a single forward pass
  that extracts log-probs and BOTH moments from the identical positions in
  one vectorized pass, so there is no way for the two to desync.

score_batch itself is intentionally NOT modified (D-03/D-10: don't touch the
shared, checkpointed production path). score_batch_curvature below is a
deliberate, targeted duplication of its forward-pass block (padding,
logits_to_keep sizing, target-position extraction) with one added vectorized
line for the second moment -- not a from-scratch reimplementation.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq


from fractalsig.pipeline.build_p0_local import CLIP_TO  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import (  # noqa: E402
    MANIFEST_DIR, load_models, is_memory_error, empty_device_cache,
    compute_align_window,
)

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "fastdetectgpt_scores.jsonl")
SHARD_CKPT_PATH = os.path.join(DATA_DIR, "fastdetectgpt_shard_checkpoint.txt")

BATCH_SIZE_INITIAL = 48  # measured ~96GiB allocation (2026-08-25 concurrency probe) vs
                         # the old 23.62GiB budget BATCH_SIZE_INITIAL=16 was set for --
                         # the memory-error halving path still guards the fallback case
MIN_VRAM_GIB_FLOOR = 20  # E-18's own stated floor for a model this size; NOT §2.3's 40,
                         # which would abort every job on the pre-authorized 23.62GiB
                         # fallback slice


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_line_set(path):
    s = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)
    return s


def append_line(path, value):
    with open(path, "a") as f:
        f.write(str(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_vram_floor(torch_mod, device):
    if not torch_mod.cuda.is_available():
        print("[abort] no CUDA device visible", flush=True)
        sys.exit(1)
    total_gib = torch_mod.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[gate] GPU total memory: {total_gib:.2f} GiB", flush=True)
    if total_gib < MIN_VRAM_GIB_FLOOR:
        print(f"[FAIL] {total_gib:.2f} GiB < {MIN_VRAM_GIB_FLOOR} GiB floor. Aborting "
              f"rather than OOMing mid-run.", flush=True)
        sys.exit(1)


def score_batch_curvature(torch_mod, model, tokenizer, device, token_id_lists, windows):
    """Targeted duplicate of build_p1_feature_store_gpu.score_batch's forward
    pass (padding / logits_to_keep sizing / target-position extraction
    identical, verified against that function line-for-line while writing
    this), with ONE addition: the vectorized second moment
    sum_v p(v)(log p(v))^2 at each extracted position, computed in the same
    pass as entropy so mu_t/sigma_t^2 can never desync from what was
    actually scored.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    fed_lists = [ids[:end] for ids, (start, end) in zip(token_id_lists, windows)]
    max_len = max(len(t) for t in fed_lists)
    input_ids = torch_mod.full((len(fed_lists), max_len), pad_id, dtype=torch_mod.long)
    attn_mask = torch_mod.zeros((len(fed_lists), max_len), dtype=torch_mod.long)
    for i, ids in enumerate(fed_lists):
        input_ids[i, max_len - len(ids):] = torch_mod.tensor(ids, dtype=torch_mod.long)
        attn_mask[i, max_len - len(ids):] = 1
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    min_logit_pos = None
    for ids, (start, end) in zip(fed_lists, windows):
        t_start = max(start, 1)
        if t_start >= end:
            continue
        pad_offset = max_len - len(ids)
        pos = pad_offset + t_start - 1
        if min_logit_pos is None or pos < min_logit_pos:
            min_logit_pos = pos
    keep = max_len if min_logit_pos is None else max(1, min(max_len, max_len - min_logit_pos + 2))

    with torch_mod.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, logits_to_keep=keep)
        logits = out.logits
        log_probs = torch_mod.log_softmax(logits.float(), dim=-1)
    kept_offset = max_len - keep

    probs = log_probs.exp()
    entropy_all = -(probs * log_probs).sum(dim=-1)              # E[-log p] = mu_t (negated)
    second_moment_all = (probs * log_probs.pow(2)).sum(dim=-1)  # E[(log p)^2]

    results = []
    for i, (ids, (start, end)) in enumerate(zip(fed_lists, windows)):
        pad_offset = max_len - len(ids)
        t_start = max(start, 1)
        if t_start >= end:
            results.append({"log_probs": [], "mu": [], "sigma2": []})
            continue
        target_positions = torch_mod.arange(t_start, end, device=input_ids.device)
        logit_positions = pad_offset + target_positions - 1 - kept_offset
        target_ids = input_ids[i, pad_offset + target_positions]
        lp_window = log_probs[i, logit_positions]
        target_logprobs = lp_window.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        ent = entropy_all[i, logit_positions]
        m2 = second_moment_all[i, logit_positions]
        mu = -ent                       # E[log p] = -entropy, exactly
        sigma2 = (m2 - mu.pow(2)).clamp(min=1e-12)  # numerical floor, variance can't be negative
        results.append({
            "log_probs": target_logprobs.tolist(),
            "mu": mu.tolist(),
            "sigma2": sigma2.tolist(),
        })
    return results


def score_documents(torch_mod, model, tok, device, rows):
    text_ids_by_idx = {i: tok(r["text"])["input_ids"] for i, r in enumerate(rows)}
    usable = []
    for idx, ids in text_ids_by_idx.items():
        window = compute_align_window(len(ids))
        if window is None:
            continue
        usable.append((idx, ids[:window[1]], window))

    scores = {}
    pos = 0
    batch_size = BATCH_SIZE_INITIAL
    while pos < len(usable):
        batch = usable[pos:pos + batch_size]
        batch_ids = [b[1] for b in batch]
        batch_windows = [b[2] for b in batch]
        try:
            scored_batch = score_batch_curvature(torch_mod, model, tok, device, batch_ids, batch_windows)
        except RuntimeError as e:
            if is_memory_error(e) and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                empty_device_cache(torch_mod, device)
                continue
            raise
        for (idx, ids, window), scored in zip(batch, scored_batch):
            lp, mu, s2 = scored["log_probs"], scored["mu"], scored["sigma2"]
            if len(lp) == 0:
                continue
            lp_sum = float(np.sum(lp))
            mu_sum = float(np.sum(mu))
            var_sum = float(np.sum(s2))
            d = (lp_sum - mu_sum) / (var_sum ** 0.5) if var_sum > 0 else float("nan")
            scores[idx] = {"n_scored": len(lp), "d": d}
        pos += len(batch)
    return scores


def main(limit_shards=None):
    _ensure_dirs()
    torch_mod, device, dtype, tokenizers, models, _ = load_models(scorers=("gemma-2b",))
    check_vram_floor(torch_mod, device)
    tok, model = tokenizers["gemma-2b"], models["gemma-2b"]

    done_shards = load_line_set(SHARD_CKPT_PATH)
    shard_files = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet")))
    if not shard_files:
        print(f"[abort] no shards found in {MANIFEST_DIR}", flush=True)
        sys.exit(1)
    remaining = [s for s in shard_files if s not in done_shards]
    if limit_shards:
        remaining = remaining[:limit_shards]
    print(f"[info] {len(shard_files)} total shards, {len(done_shards)} already done, "
          f"{len(remaining)} to process this run", flush=True)

    t0 = time.time()
    n_written = 0
    for si, shard in enumerate(remaining, 1):
        table = pq.read_table(shard, columns=["id", "text"])
        rows = table.to_pylist()
        scores = score_documents(torch_mod, model, tok, device, rows)
        with open(OUT_PATH, "a") as f:
            for idx, row in enumerate(rows):
                s = scores.get(idx)
                if s is None:
                    continue
                f.write(json.dumps({"id": row["id"], "score": s["d"],
                                    "n_scored": s["n_scored"]}) + "\n")
                n_written += 1
        append_line(SHARD_CKPT_PATH, shard)
        if si % 20 == 0 or si == len(remaining):
            elapsed = time.time() - t0
            rate = si / elapsed
            eta = (len(remaining) - si) / rate if rate > 0 else float("nan")
            print(f"[progress] {si}/{len(remaining)} shards  written={n_written}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"[done] {n_written} scores written to {OUT_PATH} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()
    main(limit_shards=args.limit_shards)
