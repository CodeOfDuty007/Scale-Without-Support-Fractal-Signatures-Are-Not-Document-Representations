#!/usr/bin/env python3
"""Canonical Binoculars (Hans et al. 2024), self-scored over this pipeline's
own window discipline. NOT the same as Cross_PPL_Ratio (Gemma/Mistral
log-PPL ratio, already one of the detector's own 54 features) -- that one is
reported separately, explicitly labelled as a cross-perplexity ABLATION of
an existing feature, never as Binoculars, per an explicit
correction on that point.

Observer = tiiuae/falcon-7b-instruct, Performer = tiiuae/falcon-7b (the
paper's own pairing). Both models loaded resident simultaneously in ONE
process -- Binoculars needs both models' full next-token distributions at
the SAME positions to compute the cross-perplexity term, so there is no
sequential-loading option the way there was for single-model scoring.
~14GB each in bf16, ~28GB total; this job must run WITHOUT other heavy GPU
jobs concurrent on this node (see 2026-08-25 finding: the MIG partition does
NOT give isolated physical memory -- confirmed by a real OOM cascade when 4
"concurrent" jobs shared one ~95GB pool).

Score:
  PPL(x)   = exp(mean NLL of x under the OBSERVER model)
  X-PPL(x) = exp(-1/T * sum_t sum_v P_performer(v|x_<t) * log P_observer(v|x_<t))
  binoculars_score(x) = PPL(x) / X-PPL(x)
Lower score => more likely machine-generated (paper's convention; published
threshold ~0.9015 on their own corpus -- NOT assumed to transfer here,
reported as a continuous score and evaluated via AUROC/TPR@FPR like every
other baseline, not via their fixed threshold).

Same window as everywhere else in this pipeline (self-scored, D-02):
post-WARMUP_DROP, clipped to CLIP_TO, MIN_TOKENS floor, t_start=max(start,1)
per F-3's root cause -- computed identically for both models over the SAME
token range so the two half-scores can never desync.
"""
from fractalsig import paths as _paths  # noqa: E402
import glob
import json
import os
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from fractalsig.pipeline.build_p0_local import CLIP_TO  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_gpu import MANIFEST_DIR, is_memory_error, empty_device_cache  # noqa: E402
from fractalsig.baselines.fast_detectgpt_score import check_vram_floor  # noqa: E402 -- reuse the same VRAM gate

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "binoculars_scores.jsonl")
SHARD_CKPT_PATH = os.path.join(DATA_DIR, "binoculars_shard_checkpoint.txt")

OBSERVER_ID = "tiiuae/falcon-7b-instruct"
PERFORMER_ID = "tiiuae/falcon-7b"
WARMUP_DROP = 64
MIN_TOKENS = 320
BATCH_SIZE_INITIAL = 8  # two 7B models resident -- deliberately conservative
                        # after today's memory lesson; the OOM-halving path
                        # still guards the rest


def compute_window(n_tokens):
    if not n_tokens:
        return None
    if n_tokens >= CLIP_TO:
        start, end = WARMUP_DROP, CLIP_TO
    else:
        start, end = 0, n_tokens
    if (end - start) < MIN_TOKENS:
        return None
    return start, end


def load_line_set(path):
    s = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)
    return s


def load_falcon_pair(device):
    tok = AutoTokenizer.from_pretrained(OBSERVER_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    observer = AutoModelForCausalLM.from_pretrained(OBSERVER_ID, dtype=torch.bfloat16).to(device).eval()
    performer = AutoModelForCausalLM.from_pretrained(PERFORMER_ID, dtype=torch.bfloat16).to(device).eval()
    return tok, observer, performer


def score_batch_binoculars(tok, observer, performer, device, texts, windows):
    ids_list = [tok(t)["input_ids"][:w[1]] for t, w in zip(texts, windows)]
    max_len = max(len(x) for x in ids_list)
    pad_id = tok.pad_token_id
    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attn[i, max_len - len(ids):] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    with torch.no_grad():
        obs_logits = observer(input_ids=input_ids, attention_mask=attn).logits.float()
        perf_logits = performer(input_ids=input_ids, attention_mask=attn).logits.float()
        obs_logp = torch.log_softmax(obs_logits, dim=-1)
        perf_p = torch.softmax(perf_logits, dim=-1)
        # cross term per position: -sum_v P_performer(v) * log P_observer(v)
        cross_ent_all = -(perf_p * obs_logp).sum(dim=-1)  # (batch, seq)

    results = []
    for i, (ids, (start, end)) in enumerate(zip(ids_list, windows)):
        pad_offset = max_len - len(ids)
        t_start = max(start, 1)
        if t_start >= end:
            results.append(None)
            continue
        target_positions = torch.arange(t_start, end, device=device)
        logit_positions = pad_offset + target_positions - 1
        target_ids = input_ids[i, pad_offset + target_positions]
        obs_logp_window = obs_logp[i, logit_positions]
        obs_nll = -obs_logp_window.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (n,)
        cross_ent_window = cross_ent_all[i, logit_positions]  # (n,)
        ppl = float(torch.exp(obs_nll.mean()))
        xppl = float(torch.exp(cross_ent_window.mean()))
        score = ppl / xppl if xppl > 0 else float("nan")
        results.append({"n_scored": len(target_positions), "ppl": ppl, "xppl": xppl, "score": score})
    return results


def main(limit_shards=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_vram_floor(torch, torch.device(device))
    tok, observer, performer = load_falcon_pair(device)
    print("[binoculars] both models loaded", flush=True)

    done_shards = load_line_set(SHARD_CKPT_PATH)
    shard_files = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.parquet")))
    remaining = [s for s in shard_files if s not in done_shards]
    if limit_shards:
        remaining = remaining[:limit_shards]
    print(f"[binoculars] {len(shard_files)} total, {len(done_shards)} done, {len(remaining)} to process", flush=True)

    t0 = time.time()
    n_written = 0
    for si, shard in enumerate(remaining, 1):
        table = pq.read_table(shard, columns=["id", "text"])
        rows = table.to_pylist()

        usable = []
        for idx, row in enumerate(rows):
            n_ids = len(tok(row["text"])["input_ids"])
            window = compute_window(n_ids)
            if window is not None:
                usable.append((idx, row["text"], window))

        pos = 0
        batch_size = BATCH_SIZE_INITIAL
        scored = {}
        while pos < len(usable):
            batch = usable[pos:pos + batch_size]
            batch_texts = [b[1] for b in batch]
            batch_windows = [b[2] for b in batch]
            try:
                results = score_batch_binoculars(tok, observer, performer, device, batch_texts, batch_windows)
            except RuntimeError as e:
                if is_memory_error(e) and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    empty_device_cache(torch, torch.device(device))
                    continue
                raise
            for (idx, _, _), r in zip(batch, results):
                if r is not None:
                    scored[idx] = r
            pos += len(batch)

        with open(OUT_PATH, "a") as f:
            for idx, r in scored.items():
                f.write(json.dumps({"id": rows[idx]["id"], "score": r["score"],
                                    "ppl": r["ppl"], "xppl": r["xppl"],
                                    "n_scored": r["n_scored"]}) + "\n")
                n_written += 1
        with open(SHARD_CKPT_PATH, "a") as f:
            f.write(shard + "\n")

        if si % 10 == 0 or si == len(remaining):
            elapsed = time.time() - t0
            rate = si / elapsed
            eta = (len(remaining) - si) / rate if rate > 0 else float("nan")
            print(f"[progress] {si}/{len(remaining)} shards  written={n_written}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"[done] {n_written} scores -> {OUT_PATH} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None)
    args = ap.parse_args()
    main(limit_shards=args.limit_shards)
