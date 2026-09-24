#!/usr/bin/env python3
"""P5 track A, RAID scoring stage. RAID has NO precomputed token scores
(unlike GAGLE, which ships gagle_*_sliced_scores) -- so unlike P4's rescore
passes, this is the ONLY scoring RAID ever gets, not a delta against a
precomputed baseline. Score every row in the stratified sample
(`p5_sample_raid.py`'s 120,005-row output) with Gemma-2B and Mistral-7B,
same code path as `p4_rescore_edits.py`/`p4_b2_ladder_rescore.py` (
`compute_align_window`, `score_batch`, `load_models` from
`fractalsig/pipeline/build_p1_feature_store_gpu.py`), same OOM-guarded length-sorted
chunking. Output feeds `p5_raid_build_features.py` directly -- schema
matches p4_b2_ladder_rescore.py's rescore records exactly (same downstream
consumer, `build_feature_row`).
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys
import time

HERE = _paths.WORKSPACE_STR

SCORER_KEYS = {"gemma": "gemma-2b", "mistral": "mistral-7b"}
DEFAULT_BATCH_SIZE = {"gemma": 16, "mistral": 4}
D = lambda *p: os.path.join(HERE, "data", *p)


def out_path(scorer):
    return D(f"p5_raid_rescore_{scorer}.jsonl")


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        for l in open(path):
            done.add(json.loads(l)["id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=["gemma", "mistral"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen_batch_size", type=int, default=None)
    args = ap.parse_args()
    gen_batch_size = args.gen_batch_size or DEFAULT_BATCH_SIZE[args.scorer]

    from fractalsig.pipeline.build_p1_feature_store_gpu import load_models, compute_align_window, score_batch

    scorer_key = SCORER_KEYS[args.scorer]
    rows = [json.loads(l) for l in open(D("p5_raid_sample.jsonl"))]
    if args.limit is not None:
        rows = rows[:args.limit]

    print(f"[p5-rescore] loading {scorer_key}...", flush=True)
    t0 = time.time()
    torch_mod, device, dtype, tokenizers, models, _embedder = load_models(scorers=(scorer_key,))
    tok = tokenizers[scorer_key]
    model = models[scorer_key]
    print(f"[p5-rescore] loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    dst_path = out_path(args.scorer)
    done = load_done_ids(dst_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[p5-rescore] scorer={args.scorer}: {len(rows)} total, {len(done)} already done, "
          f"{len(todo)} remaining, gen_batch_size={gen_batch_size}", flush=True)
    if not todo:
        print("[p5-rescore] nothing to do", flush=True)
        return

    pre_tok = [(r, tok(r["text"], truncation=False)["input_ids"]) for r in todo]
    pre_tok.sort(key=lambda pair: len(pair[1]))

    out_f = open(dst_path, "a")
    run_start = time.time()
    n_scored = 0
    n_skipped_short = 0

    for chunk_start in range(0, len(pre_tok), gen_batch_size):
        chunk = pre_tok[chunk_start:chunk_start + gen_batch_size]

        token_id_lists, windows, valid_pos = [], [], set()
        for pos, (r, ids) in enumerate(chunk):
            win = compute_align_window(len(ids))
            if win is None:
                n_skipped_short += 1
                continue
            token_id_lists.append(ids)
            windows.append(win)
            valid_pos.add(pos)

        results = score_batch(torch_mod, model, tok, device, token_id_lists, windows) if token_id_lists else []

        r_i = 0
        for pos, (r, ids) in enumerate(chunk):
            if pos not in valid_pos:
                out_f.write(json.dumps({
                    "id": r["id"], "domain": r["domain"], "Label": r["Label"], "attack": r["attack"],
                    "scorer": scorer_key, "window": None, "n_scored": 0,
                    "scores": [], "ranks": [], "entropies": [], "skipped_below_min_tokens": True,
                }) + "\n")
                continue
            res = results[r_i]
            win = windows[r_i]
            r_i += 1
            surprisal = [-lp for lp in res["log_probs"]]
            out_f.write(json.dumps({
                "id": r["id"], "domain": r["domain"], "Label": r["Label"], "attack": r["attack"],
                "scorer": scorer_key, "window": list(win),
                "n_scored": len(surprisal), "scores": surprisal,
                "ranks": res["ranks"], "entropies": res["entropies"],
            }) + "\n")
            n_scored += 1

        out_f.flush()
        os.fsync(out_f.fileno())
        if (chunk_start + gen_batch_size) % 4000 < gen_batch_size:
            elapsed = time.time() - run_start
            rate = (n_scored + n_skipped_short) / elapsed if elapsed > 0 else 0
            eta = (len(pre_tok) - chunk_start - len(chunk)) / rate if rate > 0 else float("nan")
            print(f"[p5-rescore]   {chunk_start + len(chunk)}/{len(pre_tok)} "
                  f"({elapsed:.0f}s, {n_scored} scored, {n_skipped_short} below-min, "
                  f"ETA {eta:.0f}s)", flush=True)

    out_f.close()
    print(f"[p5-rescore] DONE scorer={args.scorer}: {n_scored} scored, {n_skipped_short} skipped below-min "
          f"in {time.time()-run_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
