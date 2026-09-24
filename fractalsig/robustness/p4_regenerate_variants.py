#!/usr/bin/env python3
"""Targeted regeneration (2026-08-26): batches 1 and 2 were generated with a
decode config that produced severe excess repetition on 4 of 7 edit types
(quality-checked against a sentence_reorder baseline across all 5 domains,
2026-08-26 -- see ledger). Fix: no_repeat_ngram_size=3 for
paraphrase_light/paraphrase_heavy/style_transfer (ordinary greedy-decode
looping), and a raised summary budget (80->180 tokens) plus
no_repeat_ngram_size=3 for summarize_then_expand (a length-asymmetry defect,
not ordinary looping -- see build_edits.py's comment on SUMMARIZE_PROMPT).

Rather than re-run the full cascade (wasting the 3 already-healthy
generation types plus the 2 free programmatic types), this reads an
existing p4_edits_batchN.jsonl, regenerates ONLY the 4 affected variant
fields per document using the fixed PROMPTS/SUMMARIZE_PROMPT/EXPAND_PROMPT
from build_edits.py (imported, not restated), and rewrites the file.
human_revision, sentence_reorder, typo_noise pass through unchanged.

Checkpointed the same way as build_edits.py: reads its own output-in-
progress on restart and skips ids already regenerated.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from fractalsig.robustness.build_edits import (
    MODEL_ID, EMBED_MODEL_ID, MIN_TOKENS, CLIP_TO, PROMPTS, SUMMARIZE_PROMPT, EXPAND_PROMPT,
    trim_to_clip, batched_generate, embed_batch, cosine, GENERATION_TYPES, SAMPLE_PATH,
    strip_label_artifacts,
)

HERE = _paths.WORKSPACE_STR
REGEN_TYPES = ["paraphrase_light", "paraphrase_heavy", "style_transfer", "summarize_then_expand"]


def load_original_texts():
    """id -> raw source text, from the untouched sample -- NOT from any
    edited variant. p4_edits_batch*.jsonl never persisted the trimmed
    snippet itself (only its token COUNT), and reusing typo_noise's text
    (an early draft of this script's mistake) would feed char-level-noised
    text into the paraphrase/summarize prompts. trim_to_clip() is
    deterministic given the same source text and CLIP_TO, so re-deriving
    here reproduces the exact snippet build_edits.py originally used."""
    out = {}
    with open(SAMPLE_PATH) as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row["text"]
    return out


def in_path(batch):
    return os.path.join(HERE, "data", f"p4_edits_batch{batch}.jsonl")


def out_path(batch):
    return os.path.join(HERE, "data", f"p4_edits_batch{batch}_regen.jsonl")


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True, choices=[1, 2])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    args = ap.parse_args()

    src_path = in_path(args.batch)
    dst_path = out_path(args.batch)
    all_recs = [json.loads(l) for l in open(src_path)]
    orig_texts = load_original_texts()
    missing = [r["id"] for r in all_recs if r["id"] not in orig_texts]
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} record ids have no source text in "
                          f"{SAMPLE_PATH} (e.g. {missing[:3]}) -- cannot regenerate "
                          f"without the original snippet")
    done = load_done_ids(dst_path)
    todo = [r for r in all_recs if r["id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[regen] batch {args.batch}: {len(all_recs)} total, {len(done)} already regenerated, "
          f"{len(todo)} remaining", flush=True)
    if not todo:
        print("[regen] nothing to do", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    embed_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
    print("[regen] models loaded", flush=True)

    out_f = open(dst_path, "a")
    run_start = time.time()
    n_done = 0

    for chunk_start in range(0, len(todo), args.gen_batch_size):
        chunk = todo[chunk_start:chunk_start + args.gen_batch_size]

        # re-derive the EXACT original snippet build_edits.py used (see
        # load_original_texts() docstring for why this isn't reconstructed
        # from any edited variant)
        snippets = [trim_to_clip(tok, orig_texts[r["id"]], CLIP_TO)[0] for r in chunk]

        per_doc_new = [dict() for _ in chunk]
        for etype, (template, max_new, no_repeat) in PROMPTS.items():
            if etype not in REGEN_TYPES:
                continue
            prompts = [template.format(snippet=s) for s in snippets]
            t0 = time.time()
            results = batched_generate(model, tok, device, prompts, max_new,
                                        no_repeat_ngram_size=no_repeat)
            dt = time.time() - t0
            share = dt / len(chunk)
            for i, (text, n_new) in enumerate(results):
                per_doc_new[i][etype] = {"text": text, "gen_seconds": share, "gen_tokens": n_new}

        t0 = time.time()
        summarize_prompts = [SUMMARIZE_PROMPT.format(snippet=s) for s in snippets]
        summaries = batched_generate(model, tok, device, summarize_prompts, 180, no_repeat_ngram_size=3)
        expand_prompts = [EXPAND_PROMPT.format(summary=s) for s, _ in summaries]
        expansions = batched_generate(model, tok, device, expand_prompts, 460, no_repeat_ngram_size=3)
        dt = time.time() - t0
        share = dt / len(chunk)
        for i in range(len(chunk)):
            summ_text, n_sum = summaries[i]
            exp_text, n_exp = expansions[i]
            per_doc_new[i]["summarize_then_expand"] = {
                "text": exp_text, "gen_seconds": share, "gen_tokens": n_sum + n_exp,
                "intermediate_summary": summ_text,
            }

        orig_vecs = embed_batch(embed_model, embed_tok, device, snippets)
        for j, r in enumerate(chunk):
            for etype in REGEN_TYPES:
                stripped, had_artifact = strip_label_artifacts(per_doc_new[j][etype]["text"])
                per_doc_new[j][etype]["text"] = stripped
                per_doc_new[j][etype]["had_label_artifact"] = had_artifact
            edited_texts = [per_doc_new[j][t]["text"] for t in REGEN_TYPES]
            edited_vecs = embed_batch(embed_model, embed_tok, device, edited_texts)
            new_variants = dict(r["variants"])  # copy, overwrite only REGEN_TYPES
            for k, etype in enumerate(REGEN_TYPES):
                v = per_doc_new[j][etype]
                text = v["text"]
                sim = cosine(orig_vecs[j], edited_vecs[k])
                n_tokens = len(tok(text, truncation=False)["input_ids"])
                below_min = n_tokens < MIN_TOKENS
                is_empty = not text.strip()
                new_variants[etype] = {
                    "text": text, "gen_seconds": v["gen_seconds"], "gen_tokens": v.get("gen_tokens"),
                    "semantic_cosine": sim, "n_tokens": n_tokens, "below_min_tokens": below_min,
                    "is_empty": is_empty, "had_label_artifact": v.get("had_label_artifact", False),
                    "cost_class": "generation", "regenerated": True,
                }
                if "intermediate_summary" in v:
                    new_variants[etype]["intermediate_summary"] = v["intermediate_summary"]

            record = dict(r)
            record["variants"] = new_variants
            out_f.write(json.dumps(record) + "\n")

        out_f.flush()
        os.fsync(out_f.fileno())
        n_done += len(chunk)
        elapsed = time.time() - run_start
        rate = elapsed / n_done
        eta_h = (rate * (len(todo) - n_done)) / 3600
        print(f"[regen] {n_done}/{len(todo)} | {rate:.2f}s/doc | ETA {eta_h:.2f}h", flush=True)

    out_f.close()
    print(f"[regen] batch {args.batch} regeneration pass done, wrote {dst_path}", flush=True)


if __name__ == "__main__":
    main()
