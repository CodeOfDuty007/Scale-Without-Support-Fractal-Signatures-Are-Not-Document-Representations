#!/usr/bin/env python3
"""B2 stage 1 (2026-08-27): full attacker-capability ladder,
generation. Same experiment as B1, expanded to all 6 bench models (Qwen2.5
0.5B/1.5B/7B/14B-Instruct, Phi-3.5-mini-instruct, Falcon-7B-Instruct -- all
already on disk from the 24-doc bench probe) x paraphrase_heavy x ~1,000
docs each -- the SAME 1,000 docs across all 6 models, for a controlled
"detector robustness as a function of attacker capability" curve rather
than 6 independent samples. Converts B1's single diagnostic point into a
size sweep (Qwen family, controlled) plus two cross-family controls
(Phi-3.5-mini at 3.8B, Falcon-7B-Instruct at 7B).

Quality-check suite matches B1 exactly, PLUS a new CJK-contamination filter
(follow-up to B1's finding: Qwen2.5-7B-Instruct code-switched
into Chinese on 6.35% of outputs) -- flagged per-row, not silently dropped,
same treatment as the cosine floor and label-artifact rate throughout this
project.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import gc
import json
import os
import re
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

from fractalsig.robustness.build_edits import trim_to_clip, CLIP_TO  # noqa: E402

# Falcon-7B-Instruct DROPPED from the ladder 2026-08-28 (design decision, per
# AskUserQuestion): n=24 recheck (after a stop_strings=["\nUser",...] fix
# for an initial chat-turn-leakage defect) still showed 95.8% below-min-
# tokens and worse median cosine (0.707) than the n=8 smoke test -- a real,
# non-noisy finding. Root cause is genuine instruction-following failure on
# this specific task/prompt (hallucinates a fake "\nUser" turn almost
# immediately, or produces meta-commentary describing the rewrite instead
# of performing it, e.g. "The text has been rewritten using different
# sentence structures... 21 characters shorter than the original text.")
# -- not a decoding bug, extends the 24-doc bench probe's earlier flag of
# Falcon as the cosine outlier into a much stronger signal for THIS attack
# type specifically. Documented as a methods finding, same treatment as
# human_revision being dropped from the original P4 battery for the same
# class of reason (base-model/small-instruct-model instruction-following
# limits, not a fixable prompt bug).
LADDER = [
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B-Instruct"),
    ("Phi-3.5-mini-instruct", "microsoft/Phi-3.5-mini-instruct"),
]

INSTRUCTION = ("Thoroughly rewrite the following text using different sentence "
               "structures and vocabulary throughout, while preserving its exact "
               "meaning and approximate length. Output ONLY the rewritten passage, "
               "nothing else.\n\nOriginal passage:\n{snippet}")
MAX_NEW_TOKENS = 460
N_SAMPLE = 1000
SEED = 20260827

NUMBERED_LINE_RE = re.compile(r"^\s*\(?(\d{1,2})[\.\)]\s+", re.MULTILINE)
LABEL_ARTIFACT_RE = re.compile(
    r"\b(Paraphrase[d]?|Rewrite|Rewritten passage|Original passage)\s*:", re.IGNORECASE)
CJK_RE = re.compile(r"[一-鿿぀-ヿ]")


def extract_numbered_items(text):
    matches = list(NUMBERED_LINE_RE.finditer(text))
    if len(matches) < 3:
        return None
    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item_text = text[start:end].strip()
        if item_text:
            items.append(item_text)
    return items if len(items) >= 3 else None


def generate_for_model(name, model_id, docs, device, batch_size=8):
    print(f"\n[b2-gen] === generating with {name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()

    results = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        snippets = []
        for d in batch:
            snippet, ntok = trim_to_clip(tok, d["text"], CLIP_TO)
            snippets.append(snippet)
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION.format(snippet=s)}],
            tokenize=False, add_generation_prompt=True) for s in snippets]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  no_repeat_ngram_size=5, pad_token_id=tok.pad_token_id,
                                  stop_strings=["\nUser", "\nUser:", "User:"], tokenizer=tok)
        for j, d in enumerate(batch):
            new_tokens = out[j][enc["input_ids"].shape[1]:]
            gen_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
            results.append({"id": d["id"], "domain": d["domain"], "generator": d["generator"],
                             "attacker": name, "snippet": snippets[j], "text": gen_text})
        if (i + batch_size) % 200 == 0:
            print(f"[b2-gen]   {i + batch_size}/{len(docs)}", flush=True)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None,
                     help="subset of LADDER model names to run, default all 6")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[b2-gen] device={device}", flush=True)
    if device != "cuda":
        print("[b2-gen] WARNING: no GPU visible, this will be very slow", flush=True)

    sample = [json.loads(l) for l in open(D("p4_sampled_docs.jsonl"))]
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(sample), size=(args.limit or N_SAMPLE), replace=False)
    docs = [sample[i] for i in idx]
    print(f"[b2-gen] {len(docs)} docs sampled (seed={SEED}), SAME docs for every attacker model", flush=True)

    ladder = [m for m in LADDER if args.models is None or m[0] in args.models]

    out_path = D("p4_b2_ladder_generations.jsonl")
    done_attackers = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            done_attackers.add(json.loads(l)["attacker"])
    ladder = [(n, m) for n, m in ladder if n not in done_attackers]
    print(f"[b2-gen] {len(done_attackers)} attackers already done: {sorted(done_attackers)}", flush=True)
    print(f"[b2-gen] {len(ladder)} attackers to run: {[n for n, _ in ladder]}", flush=True)

    out_f = open(out_path, "a")
    all_new = []
    for name, model_id in ladder:
        results = generate_for_model(name, model_id, docs, device)
        for r in results:
            out_f.write(json.dumps(r) + "\n")
        out_f.flush()
        all_new.extend(results)
        print(f"[b2-gen] {name}: {len(results)} generations written", flush=True)
    out_f.close()

    if not all_new:
        print("[b2-gen] nothing new generated this run", flush=True)
        return

    # --- quality checks, including the new CJK filter ---
    print("\n[b2-gen] === quality checks (this run's new attackers only) ===", flush=True)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    from fractalsig.pipeline.build_p1_feature_store_gpu import load_models, compute_align_window
    torch_mod, _, _, tokenizers, models, _ = load_models(scorers=("gemma-2b",))
    gemma_tok = tokenizers["gemma-2b"]

    summary_path = D("p4_b2_ladder_quality_summary.json")
    by_attacker = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for name, _ in ladder:
        rows = [r for r in all_new if r["attacker"] == name]
        snippets = [r["snippet"] for r in rows]
        texts = [r["text"] for r in rows]

        orig_emb = embedder.encode(snippets, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        gen_emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        cos = (orig_emb * gen_emb).sum(axis=1) / (
            np.linalg.norm(orig_emb, axis=1) * np.linalg.norm(gen_emb, axis=1) + 1e-8)

        below_min = 0
        for r in rows:
            ids = gemma_tok(r["text"], truncation=False)["input_ids"]
            if compute_align_window(len(ids)) is None:
                below_min += 1

        label_artifact = sum(1 for t in texts if LABEL_ARTIFACT_RE.search(t))
        cjk_flagged = sum(1 for t in texts if CJK_RE.search(t))

        enum_flagged, enum_degenerate = 0, 0
        for t in texts:
            items = extract_numbered_items(t)
            if items is not None:
                enum_flagged += 1
                emb = embedder.encode(items, convert_to_numpy=True, show_progress_bar=False)
                norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
                sim = norm @ norm.T
                n = len(items)
                iu = np.triu_indices(n, k=1)
                if float(sim[iu].mean()) >= 0.80:
                    enum_degenerate += 1

        by_attacker[name] = {
            "n": len(rows),
            "median_cosine": float(np.median(cos)),
            "frac_near_identity_cos_gt_0.98": float(np.mean(cos > 0.98)),
            "frac_low_cosine_lt_0.5": float(np.mean(cos < 0.5)),
            "below_min_tokens_rate": below_min / len(rows),
            "label_artifact_rate": label_artifact / len(rows),
            "cjk_contamination_rate": cjk_flagged / len(rows),
            "enum_flagged": enum_flagged,
            "enum_degenerate": enum_degenerate,
            "enum_degenerate_rate": enum_degenerate / len(rows),
        }
        print(f"\n[b2-gen] {name}:")
        for k, v in by_attacker[name].items():
            print(f"[b2-gen]   {k}: {v}")

    with open(summary_path, "w") as f:
        json.dump(by_attacker, f, indent=2)
    print(f"\n[b2-gen] written to {summary_path}")


if __name__ == "__main__":
    main()
