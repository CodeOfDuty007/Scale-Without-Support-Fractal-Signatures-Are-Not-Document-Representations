#!/usr/bin/env python3
"""P4 task 3 (2026-08-26/27): re-score every edited variant, PLUS the clean
snippet each was generated from, with Gemma-2B and Mistral-7B, producing
per-token surprisal arrays for Family D (Delta H, Delta S, Delta log-PPL).

Reuses the project's own scoring machinery rather than restating it:
`compute_align_window` (the [WARMUP_DROP:CLIP_TO] windowing rule) and
`score_batch` (the teacher-forced batched forward pass) from
`fractalsig/pipeline/build_p1_feature_store_gpu.py`, and `load_models` for a scorer-
restricted model load (one scorer resident at a time -- this project's own
established VRAM-management pattern for the 48GB MIG slice, see
`rescore_human_familyb.py`'s "two-pass, one scorer per PROCESS" precedent;
here done via `--scorer` restricting one process to one model).

Score convention matches GAGLE's own: score = -log_prob (surprisal), NOT
raw log_prob -- confirmed via build_p1_feature_store_gpu.py's own
`LogPPL_Mean = -mean(log_probs)` line.

CLEAN-SIDE BASELINE (added 2026-08-27, internal review): Family D is
edited-minus-clean. `data/p1_feature_store_gpu_selfscored.jsonl` (the
existing self-scored store) has only SUMMARIZED Family B features (mean/
var/skew/kurtosis of LogPPL etc.), not raw per-token arrays -- checked via
schema inspection before assuming otherwise. Differencing fresh edited-side
scores against GAGLE's own precomputed arrays would carry a checkpoint-
mismatch term (the entire alignment problem this repo's
ALIGNMENT_PEARSON_MIN=0.98 gate exists to guard against) that has nothing
to do with the attack. So the CLEAN snippet -- the exact
[WARMUP_DROP:CLIP_TO]-trimmed text `build_edits.py`'s `trim_to_clip()`
derived from each document's raw source, i.e. what was actually fed to
generation -- is scored here too, with the SAME code path and checkpoint as
every edited variant, closing that gap. Clean snippet text is re-derived
via Gemma's tokenizer (matching build_edits.py's original trim, regardless
of which scorer is currently doing the SCORING) -- loading a tokenizer
alone is cheap even during a Mistral run, no second model resident.

OOM guard (added 2026-08-27, internal review): score_batch's
`logits_to_keep` scales with the batch's longest sequence -- its own module
comment records a RuntimeError("Invalid buffer size: 10.80 GiB") from an
unsorted N=40 batch. Chunks are sorted by token length before scoring so
`keep` stays tight, and Mistral (32k vocab, larger hidden state) uses a
smaller batch size than Gemma by default.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys
import time

HERE = _paths.WORKSPACE_STR

from fractalsig.robustness.build_edits import trim_to_clip, CLIP_TO, SAMPLE_PATH  # noqa: E402

ALL_TYPES = [
    "paraphrase_light", "paraphrase_heavy", "human_revision",
    "style_transfer", "summarize_then_expand", "sentence_reorder", "typo_noise",
]
SCORER_KEYS = {"gemma": "gemma-2b", "mistral": "mistral-7b"}
DEFAULT_BATCH_SIZE = {"gemma": 16, "mistral": 4}


def out_path(batch, scorer):
    return os.path.join(HERE, "data", f"p4_rescore_batch{batch}_{scorer}.jsonl")


def load_edited_docs(batch):
    path = os.path.join(HERE, "data", f"p4_edits_batch{batch}.jsonl")
    return [json.loads(l) for l in open(path)]


def load_clean_snippets(gemma_tok):
    """id -> clean snippet text, re-derived exactly as build_edits.py did
    (Gemma tokenizer, trim to CLIP_TO), from the untouched raw source."""
    out = {}
    with open(SAMPLE_PATH) as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]], _ = trim_to_clip(gemma_tok, row["text"], CLIP_TO)
    return out


def load_done_keys(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], r["variant"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True, choices=[1, 2])
    ap.add_argument("--scorer", required=True, choices=["gemma", "mistral"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen_batch_size", type=int, default=None)
    args = ap.parse_args()
    gen_batch_size = args.gen_batch_size or DEFAULT_BATCH_SIZE[args.scorer]

    from fractalsig.pipeline.build_p1_feature_store_gpu import load_models, compute_align_window, score_batch

    scorer_key = SCORER_KEYS[args.scorer]
    docs = load_edited_docs(args.batch)
    if args.limit is not None:
        docs = docs[:args.limit]

    print(f"[rescore] loading {scorer_key}...", flush=True)
    t0 = time.time()
    torch_mod, device, dtype, tokenizers, models, _embedder = load_models(scorers=(scorer_key,))
    tok = tokenizers[scorer_key]
    model = models[scorer_key]
    print(f"[rescore] loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    # Clean-snippet text always re-derived via GEMMA's own tokenizer (matches
    # what generation actually saw), regardless of which scorer is active.
    from transformers import AutoTokenizer
    from fractalsig.robustness.build_edits import MODEL_ID as GEMMA_MODEL_ID
    gemma_tok_for_clip = tok if scorer_key == "gemma-2b" else AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
    clean_snippets = load_clean_snippets(gemma_tok_for_clip)

    items = []
    for d in docs:
        if d["id"] in clean_snippets:
            items.append({"id": d["id"], "domain": d["domain"], "generator": d["generator"],
                           "variant": "clean", "text": clean_snippets[d["id"]], "cost_class": "clean"})
        for vtype in ALL_TYPES:
            v = d["variants"][vtype]
            items.append({"id": d["id"], "domain": d["domain"], "generator": d["generator"],
                           "variant": vtype, "text": v["text"], "cost_class": v.get("cost_class")})

    dst_path = out_path(args.batch, args.scorer)
    done = load_done_keys(dst_path)
    todo = [it for it in items if (it["id"], it["variant"]) not in done]
    print(f"[rescore] batch {args.batch} scorer={args.scorer}: {len(items)} total "
          f"(doc x [clean+7 variants]), {len(done)} already done, {len(todo)} remaining, "
          f"gen_batch_size={gen_batch_size}", flush=True)
    if not todo:
        print("[rescore] nothing to do", flush=True)
        return

    # sort by token length so score_batch's shared `logits_to_keep` stays
    # tight within each chunk (internal review: an unsorted batch can blow
    # up to a multi-GiB logits tensor -- see module docstring)
    pre_tok = [(it, tok(it["text"], truncation=False)["input_ids"]) for it in todo]
    pre_tok.sort(key=lambda pair: len(pair[1]))

    out_f = open(dst_path, "a")
    run_start = time.time()
    n_scored = 0
    n_skipped_short = 0
    n_processed = 0

    for chunk_start in range(0, len(pre_tok), gen_batch_size):
        chunk = pre_tok[chunk_start:chunk_start + gen_batch_size]

        token_id_lists, windows, valid_pos = [], [], set()
        for pos, (it, ids) in enumerate(chunk):
            win = compute_align_window(len(ids))
            if win is None:
                n_skipped_short += 1
                continue
            token_id_lists.append(ids)
            windows.append(win)
            valid_pos.add(pos)

        results = score_batch(torch_mod, model, tok, device, token_id_lists, windows) if token_id_lists else []

        r_i = 0
        for pos, (it, ids) in enumerate(chunk):
            if pos not in valid_pos:
                out_f.write(json.dumps({
                    "id": it["id"], "domain": it["domain"], "generator": it["generator"],
                    "variant": it["variant"], "cost_class": it["cost_class"],
                    "scorer": scorer_key, "window": None, "n_scored": 0,
                    "scores": [], "ranks": [], "entropies": [], "skipped_below_min_tokens": True,
                }) + "\n")
                continue
            r = results[r_i]
            win = windows[r_i]
            r_i += 1
            surprisal = [-lp for lp in r["log_probs"]]  # GAGLE convention: score = -log_prob
            out_f.write(json.dumps({
                "id": it["id"], "domain": it["domain"], "generator": it["generator"],
                "variant": it["variant"], "cost_class": it["cost_class"],
                "scorer": scorer_key, "window": list(win),
                "n_scored": len(surprisal), "scores": surprisal,
                "ranks": r["ranks"], "entropies": r["entropies"],
            }) + "\n")
            n_scored += 1

        out_f.flush()
        os.fsync(out_f.fileno())
        n_processed += len(chunk)
        elapsed = time.time() - run_start
        rate = elapsed / n_processed
        eta_h = (rate * (len(pre_tok) - n_processed)) / 3600
        if n_processed % (gen_batch_size * 20) == 0 or n_processed >= len(pre_tok):
            print(f"[rescore] {n_processed}/{len(pre_tok)} | {rate:.3f}s/item | ETA {eta_h:.2f}h | "
                  f"scored={n_scored} skipped_short={n_skipped_short}", flush=True)

    out_f.close()
    print(f"[rescore] batch {args.batch} scorer={args.scorer} done. "
          f"scored={n_scored} skipped_short={n_skipped_short}/{len(pre_tok)}", flush=True)


if __name__ == "__main__":
    main()
