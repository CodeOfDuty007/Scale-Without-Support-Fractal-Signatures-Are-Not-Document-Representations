#!/usr/bin/env python3
"""B2 stage 2: re-score every ladder generation, PLUS the clean snippet it
was generated from, with Gemma-2B and Mistral-7B -- same code path,
windowing, and OOM guard as `p4_rescore_edits.py` (Family D's own
rescoring), reused directly rather than restated. Needed for the
perplexity-threshold and tabular detectors (Family A/B features), which
were built for the ORIGINAL P4 attack corpus and have no scores for these
NEW ladder generations.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import sys
import time

HERE = _paths.WORKSPACE_STR

from fractalsig.robustness.build_edits import trim_to_clip, CLIP_TO  # noqa: E402

SCORER_KEYS = {"gemma": "gemma-2b", "mistral": "mistral-7b"}
DEFAULT_BATCH_SIZE = {"gemma": 16, "mistral": 4}
D = lambda *p: os.path.join(HERE, "data", *p)


def out_path(scorer):
    return D(f"p4_b2_ladder_rescore_{scorer}.jsonl")


def load_generations():
    return [json.loads(l) for l in open(D("p4_b2_ladder_generations.jsonl"))]


def load_done_keys(path):
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done.add((r["id"], r["attacker"], r["variant"]))
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
    generations = load_generations()
    if args.limit is not None:
        generations = generations[:args.limit]

    print(f"[b2-rescore] loading {scorer_key}...", flush=True)
    t0 = time.time()
    torch_mod, device, dtype, tokenizers, models, _embedder = load_models(scorers=(scorer_key,))
    tok = tokenizers[scorer_key]
    model = models[scorer_key]
    print(f"[b2-rescore] loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    # clean-snippet text: re-derive per doc id via GEMMA's own tokenizer
    # (matches what generation actually saw), one entry per unique doc id
    # regardless of how many attackers used it (same doc pool for all 6).
    from transformers import AutoTokenizer
    from fractalsig.robustness.build_edits import MODEL_ID as GEMMA_MODEL_ID
    gemma_tok_for_clip = tok if scorer_key == "gemma-2b" else AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)

    clean_snippets = {}  # id -> snippet text
    for g in generations:
        if g["id"] not in clean_snippets:
            clean_snippets[g["id"]] = g["snippet"]

    items = []
    seen_clean = set()
    for g in generations:
        if g["id"] not in seen_clean:
            seen_clean.add(g["id"])
            items.append({"id": g["id"], "domain": g["domain"], "generator": g["generator"],
                           "attacker": "clean", "variant": "clean", "text": clean_snippets[g["id"]]})
        items.append({"id": g["id"], "domain": g["domain"], "generator": g["generator"],
                       "attacker": g["attacker"], "variant": "paraphrase_heavy", "text": g["text"]})

    dst_path = out_path(args.scorer)
    done = load_done_keys(dst_path)
    todo = [it for it in items if (it["id"], it["attacker"], it["variant"]) not in done]
    print(f"[b2-rescore] scorer={args.scorer}: {len(items)} total, {len(done)} already done, "
          f"{len(todo)} remaining, gen_batch_size={gen_batch_size}", flush=True)
    if not todo:
        print("[b2-rescore] nothing to do", flush=True)
        return

    pre_tok = [(it, tok(it["text"], truncation=False)["input_ids"]) for it in todo]
    pre_tok.sort(key=lambda pair: len(pair[1]))

    out_f = open(dst_path, "a")
    run_start = time.time()
    n_scored = 0
    n_skipped_short = 0

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
                    "attacker": it["attacker"], "variant": it["variant"],
                    "scorer": scorer_key, "window": None, "n_scored": 0,
                    "scores": [], "ranks": [], "entropies": [], "skipped_below_min_tokens": True,
                }) + "\n")
                continue
            r = results[r_i]
            win = windows[r_i]
            r_i += 1
            surprisal = [-lp for lp in r["log_probs"]]
            out_f.write(json.dumps({
                "id": it["id"], "domain": it["domain"], "generator": it["generator"],
                "attacker": it["attacker"], "variant": it["variant"],
                "scorer": scorer_key, "window": list(win),
                "n_scored": len(surprisal), "scores": surprisal,
                "ranks": r["ranks"], "entropies": r["entropies"],
            }) + "\n")
            n_scored += 1

        out_f.flush()
        os.fsync(out_f.fileno())
        if (chunk_start + gen_batch_size) % 500 < gen_batch_size:
            elapsed = time.time() - run_start
            print(f"[b2-rescore]   {chunk_start + len(chunk)}/{len(pre_tok)} "
                  f"({elapsed:.0f}s, {n_scored} scored, {n_skipped_short} below-min)", flush=True)

    out_f.close()
    print(f"[b2-rescore] DONE scorer={args.scorer}: {n_scored} scored, {n_skipped_short} skipped below-min "
          f"in {time.time()-run_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
