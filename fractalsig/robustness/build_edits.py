#!/usr/bin/env python3
"""P4 attack cascade (2026-08-26): generates 7 edit variants per document
(2 paraphrase intensities, human-revision simulation, style transfer,
summarize-then-expand, sentence reorder, typo/noise) over the stratified
GAGLE sample `p4_sample_and_cache_docs.py` wrote, and appends `Family D`
inputs (the edited text + a semantic-preservation signal + a token-count
check) to a checkpointed output file.

Batch-1 design decisions (design review, 2026-08-26), all load-bearing:
- Every generation-based edit is pinned to Gemma-2B (the ONLY model this
  pilot measured -- 262.5x generation/scoring ratio, job 192). Mistral-7B
  is not used here; its rate is unmeasured and using it in a gated 8h run
  would make the gate untestable.
- Semantic-preservation guard runs embedding-cosine ONLY (`hf_cache` has no
  NLI-finetuned checkpoint, only plain-MLM roberta-base and MiniLM). The
  cosine score is RECORDED per variant, not used to silently drop rows --
  an unvalidated threshold destroying generated text during a multi-hour
  irreversible run is worse than deferring the cut to a review pass once
  the real score distribution is known.
- MIN_TOKENS=320 (build_p0_local.py's own constant) is checked on every
  edited variant and flagged, not silently dropped, for the same reason --
  and because the drop is expected to be attack-type-correlated (F-5(a)'s
  pattern), it needs to be visible per attack type, not just a final count.
- Checkpointed per CHUNK of `--gen_batch_size` documents (not per document,
  since generation is now batched across documents within an edit type --
  see BATCHING below): this file is read back on startup and already-
  completed ids are skipped, so a SLURM wall-clock hit (Binoculars job157's
  precedent) costs at most one unfinished chunk, not the whole batch.

BATCHING (added 2026-08-26 after smoke test jobs 194/195 measured a
corrected, validated per-doc cost of 12.24s/doc -> 17.00h projected for
5,000 docs, ~2x over the 8h runtime gate): generation is now batched ACROSS
DOCUMENTS within one edit type (all docs' `paraphrase_light` prompts in one
`generate()` call, etc.), not across edit types, since `max_new_tokens`
differs per type and padding every type to the slowest one's length would
waste the gain. Left-padding is required for decoder-only batched
generation (`tok.padding_side = "left"`) or the batch produces garbage.
Programmatic edits (sentence_reorder, typo_noise) stay per-document --
they're already near-zero cost, batching them buys nothing.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import json
import os
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

HERE = _paths.WORKSPACE_STR
SAMPLE_PATH = os.path.join(HERE, "data", "p4_sampled_docs.jsonl")
MODEL_ID = "google/gemma-2b"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TOKENS = 320  # fractalsig/pipeline/build_p0_local.py:87 -- do not re-derive
CLIP_TO = 464  # fractalsig/pipeline/build_p0_local.py:88 -- the window Family A/B features are computed over
# Snippet sizing corrected 2026-08-26 (internal review after smoke test job 194):
# the pilot's 200-word snippet was reused here uncritically and tokenizes to
# ~260 tokens BEFORE any edit -- already under MIN_TOKENS=320, so every
# variant failed the below_min_tokens check regardless of content. Original
# Family A/B features are computed over a [WARMUP_DROP:CLIP_TO]=[64:464]
# window; an edited variant needs to land in a comparable post-edit token
# range for Family D's delta (edited vs clean) to be a real content
# difference rather than a windowing artifact. Measured on this corpus
# (2026-08-26, 20 docs/domain, real Gemma tokenizer): per-domain tok/word
# ratio is 1.25-1.31, fairly consistent -- so an exact per-document
# token-based trim to CLIP_TO is used below rather than a flat word count.

GENERATION_TYPES = [
    "paraphrase_light", "paraphrase_heavy", "human_revision",
    "style_transfer", "summarize_then_expand",
]
PROGRAMMATIC_TYPES = ["sentence_reorder", "typo_noise"]
ALL_TYPES = GENERATION_TYPES + PROGRAMMATIC_TYPES

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def out_path(batch):
    return os.path.join(HERE, "data", f"p4_edits_batch{batch}.jsonl")


def load_sample(batch):
    docs = []
    with open(SAMPLE_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["batch"] == batch:
                docs.append(row)
    return docs


def load_done_ids(batch):
    path = out_path(batch)
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
                continue  # a partial last line from a killed job -- ignored, not fatal
    return done


# --------------------------------------------------------------------------
# Programmatic edits (near-zero cost, no GPU, stay per-document)
# --------------------------------------------------------------------------
def sentence_reorder(text, doc_id):
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < 2:
        return text  # nothing to reorder
    rng = random.Random(f"reorder_{doc_id}")
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def typo_noise(text, doc_id, rate=0.02):
    rng = random.Random(f"typo_{doc_id}")
    chars = list(text)
    i = 0
    out = []
    keyboard_neighbors = "qwertyuiopasdfghjklzxcvbnm"
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["swap_adjacent", "delete", "substitute"])
            if op == "swap_adjacent" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
            elif op == "delete":
                i += 1
                continue
            elif op == "substitute":
                out.append(rng.choice(keyboard_neighbors))
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------
# Generation-based edits (Gemma-2B only, per the batch-1 decision), batched
# across documents within one edit type.
# --------------------------------------------------------------------------
PROMPTS = {
    # (template, max_new_tokens, no_repeat_ngram_size)
    "paraphrase_light": (
        "Paraphrase the following text in your own words, preserving its "
        "meaning and approximate length:\n\n{snippet}\n\nParaphrase:", 420, 5),
    "paraphrase_heavy": (
        "Thoroughly rewrite the following text using different sentence "
        "structures and vocabulary throughout, while preserving its exact "
        "meaning and approximate length:\n\n{snippet}\n\nRewrite:", 460, 5),
    "human_revision": (
        "Below is a passage. Rewrite it exactly as a human copy-editor would "
        "after a light edit pass -- same meaning, same length, improved "
        "phrasing.\n\nOriginal passage:\n{snippet}\n\nCopy-edited passage:\n", 420, 0),
    "style_transfer": (
        "Rewrite the following text in a more casual, conversational tone, "
        "keeping the same approximate length:\n\n{snippet}\n\nCasual version:", 420, 5),
}
# summarize_then_expand fix (2026-08-26, internal review): the +53 to +68pp
# excess repetition here (all 5 domains) is NOT ordinary greedy-decode
# looping -- it's a length-asymmetry defect. An 80-token summary gave the
# expand stage ~6x less real content than its 460-token target, so the
# model filled the gap by looping; no_repeat_ngram_size would only hide
# the symptom while ~380 tokens of the output are still content-free.
# Fix: raise the summary budget to 180 tokens (still a real summary, far
# shorter than the 420-464 token original) so expand has enough source
# material, plus no_repeat_ngram_size as a second layer since 180->460 is
# still a real expansion factor.
SUMMARIZE_PROMPT = "Summarize the following text in 4-6 sentences, covering its main points:\n\n{snippet}\n\nSummary:"
EXPAND_PROMPT = ("Expand the following summary back into a full-length passage "
                  "with specific details, at least as long as the original "
                  "article:\n\n{summary}\n\nExpanded:")


import re as _re  # local alias, SENTENCE_SPLIT_RE above already owns `re`
LABEL_ARTIFACT_RE = _re.compile(
    r"\b(Paraphrase[d]?|Rewrite|Casual version|Copy-edited passage|Expanded|"
    r"Summary|Revised)\s*:", _re.IGNORECASE)


def strip_label_artifacts(text):
    """Base-model completions sometimes emit a fresh section label mid-
    generation and restart (the same class of failure as the 'Revised:'/
    patent-boilerplate collision that broke human_revision, just not always
    fatal enough to trip the repetition detector -- found 2026-08-26 by
    manually reading regenerated output, NOT by the automated checks, which
    is itself worth remembering: measure the actual text, not just proxies
    for it). Truncates at the first label occurrence -- text after that
    point is a restart, not part of the requested single edited output."""
    m = LABEL_ARTIFACT_RE.search(text)
    if m is None:
        return text, False
    return text[:m.start()].rstrip(), True


def trim_to_clip(tok, text, clip_to):
    """Trim TEXT to at most CLIP_TO tokens by real tokenization, not a word
    count -- an exact token-based cut, not an estimate (see CLIP_TO comment
    above for why an estimate was the original defect)."""
    ids = tok(text, truncation=False)["input_ids"]
    if len(ids) <= clip_to:
        return text, len(ids)
    trimmed_ids = ids[:clip_to]
    return tok.decode(trimmed_ids, skip_special_tokens=True), len(trimmed_ids)


def batched_generate(model, tok, device, prompts, max_new_tokens, no_repeat_ngram_size=0):
    """Left-padded batched generation. Returns a list of (text, n_new_tokens)
    aligned to `prompts`. Requires tok.padding_side == 'left' (set once at
    load time) -- right-padding a decoder-only model during generation
    produces garbage, since the model would attend past padding into the
    real prompt for later positions.

    no_repeat_ngram_size added 2026-08-26 after quality-checking all 4,999
    batch-1 records against a sentence_reorder baseline (same content, no
    generation -- the right control for "how much repetition is just this
    corpus's own formulaic phrasing, e.g. patent claims"): paraphrase_light/
    heavy and style_transfer showed a consistent +8 to +22pp EXCESS 5-gram
    repetition rate over that baseline in all 5 domains -- ordinary greedy-
    decode looping, not a length-asymmetry problem (see EXPAND_PROMPT's own
    fix below for that distinct failure mode)."""
    inputs = tok(prompts, return_tensors="pt", truncation=True, max_length=768,
                 padding=True).to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    if no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]  # same for every row: left-padded to one width
    results = []
    for row in out:
        gen_ids = row[input_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        n_new = int((gen_ids != tok.pad_token_id).sum())
        results.append((text, n_new))
    return results


def run_generation_edits_batched(model, tok, device, snippets):
    """snippets: list of per-document snippet strings (same order as the
    calling chunk). Returns list of per-document dicts, one per edit type,
    aligned to `snippets`."""
    n = len(snippets)
    per_doc = [dict() for _ in range(n)]

    for etype, (template, max_new, no_repeat) in PROMPTS.items():
        prompts = [template.format(snippet=s) for s in snippets]
        t0 = time.time()
        results = batched_generate(model, tok, device, prompts, max_new,
                                    no_repeat_ngram_size=no_repeat)
        dt = time.time() - t0
        per_call_share = dt / n  # approximation: batched wall-clock split evenly
        for i, (text, n_new) in enumerate(results):
            per_doc[i][etype] = {"text": text, "gen_seconds": per_call_share, "gen_tokens": n_new}

    # summarize-then-expand: two sequential batched stages
    t0 = time.time()
    summarize_prompts = [SUMMARIZE_PROMPT.format(snippet=s) for s in snippets]
    summaries = batched_generate(model, tok, device, summarize_prompts, 180,
                                  no_repeat_ngram_size=5)
    expand_prompts = [EXPAND_PROMPT.format(summary=summ_text) for summ_text, _ in summaries]
    expansions = batched_generate(model, tok, device, expand_prompts, 460,
                                   no_repeat_ngram_size=5)
    dt = time.time() - t0
    per_call_share = dt / n
    for i in range(n):
        summ_text, n_sum = summaries[i]
        exp_text, n_exp = expansions[i]
        per_doc[i]["summarize_then_expand"] = {
            "text": exp_text, "gen_seconds": per_call_share,
            "gen_tokens": n_sum + n_exp, "intermediate_summary": summ_text,
        }

    return per_doc


# --------------------------------------------------------------------------
# Semantic-preservation signal (cosine only, per the batch-1 decision),
# also batched.
# --------------------------------------------------------------------------
def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


def embed_batch(embed_model, embed_tok, device, texts):
    inputs = embed_tok(texts, return_tensors="pt", truncation=True, max_length=256,
                        padding=True).to(device)
    with torch.no_grad():
        out = embed_model(**inputs)
    vecs = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    return torch.nn.functional.normalize(vecs, dim=-1)


def cosine(a, b):
    return float((a * b).sum())


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True, choices=[1, 2])
    ap.add_argument("--limit", type=int, default=None,
                     help="process only the first N not-yet-done docs (smoke testing)")
    ap.add_argument("--gen_batch_size", type=int, default=8,
                     help="documents per batched generate() call within one edit type")
    args = ap.parse_args()

    docs = load_sample(args.batch)
    done = load_done_ids(args.batch)
    todo = [d for d in docs if d["id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[edits] batch {args.batch}: {len(docs)} total, {len(done)} already done, "
          f"{len(todo)} remaining, gen_batch_size={args.gen_batch_size}", flush=True)
    if not todo:
        print("[edits] nothing to do", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load0 = time.time()
    print(f"[edits] loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    print(f"[edits] loading {EMBED_MODEL_ID} for the semantic-preservation signal...", flush=True)
    embed_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
    print(f"[edits] models loaded in {time.time()-t_load0:.1f}s", flush=True)

    out_f = open(out_path(args.batch), "a")
    run_start = time.time()
    n_below_min_tokens = {t: 0 for t in ALL_TYPES}
    n_done_this_run = 0

    for chunk_start in range(0, len(todo), args.gen_batch_size):
        chunk = todo[chunk_start:chunk_start + args.gen_batch_size]
        t_chunk0 = time.time()

        snippets, snippet_tokens_list = [], []
        for doc in chunk:
            s, n_tok = trim_to_clip(tok, doc["text"], CLIP_TO)
            snippets.append(s)
            snippet_tokens_list.append(n_tok)

        gen_per_doc = run_generation_edits_batched(model, tok, device, snippets)

        prog_per_doc = []
        for doc, snippet in zip(chunk, snippets):
            prog_per_doc.append({
                "sentence_reorder": {"text": sentence_reorder(snippet, doc["id"]), "gen_seconds": 0.0},
                "typo_noise": {"text": typo_noise(snippet, doc["id"]), "gen_seconds": 0.0},
            })

        orig_vecs = embed_batch(embed_model, embed_tok, device, snippets)

        for j, doc in enumerate(chunk):
            variants = {**gen_per_doc[j], **prog_per_doc[j]}
            for etype in GENERATION_TYPES:  # label-artifact strip; programmatic types never emit labels
                stripped, had_artifact = strip_label_artifacts(variants[etype]["text"])
                variants[etype]["text"] = stripped
                variants[etype]["had_label_artifact"] = had_artifact
            edited_texts = [variants[etype]["text"] for etype in ALL_TYPES]
            edited_vecs = embed_batch(embed_model, embed_tok, device, edited_texts)

            record_variants = {}
            for k, etype in enumerate(ALL_TYPES):
                v = variants[etype]
                edited_text = v["text"]
                sim = cosine(orig_vecs[j], edited_vecs[k])
                n_tokens = len(tok(edited_text, truncation=False)["input_ids"])
                below_min = n_tokens < MIN_TOKENS
                is_empty = not edited_text.strip()
                if below_min:
                    n_below_min_tokens[etype] += 1
                # is_empty added 2026-08-26 (internal review): 73/4999 batch-1
                # human_revision outputs were empty strings (a distinct
                # failure mode from below_min_tokens -- MIN_TOKENS=320 still
                # correctly flags these too, but an explicit boolean means a
                # downstream Family D consumer can't miss it inside a
                # generic below_min_tokens filter).
                record_variants[etype] = {
                    "text": edited_text,
                    "gen_seconds": v["gen_seconds"],
                    "gen_tokens": v.get("gen_tokens"),
                    "semantic_cosine": sim,
                    "n_tokens": n_tokens,
                    "below_min_tokens": below_min,
                    "is_empty": is_empty,
                    "had_label_artifact": v.get("had_label_artifact", False),
                    "cost_class": "generation" if etype in GENERATION_TYPES else "programmatic",
                }
                if "intermediate_summary" in v:
                    record_variants[etype]["intermediate_summary"] = v["intermediate_summary"]

            record = {
                "id": doc["id"], "domain": doc["domain"], "generator": doc["generator"],
                "batch": doc["batch"], "snippet_tokens": snippet_tokens_list[j],
                "variants": record_variants,
            }
            out_f.write(json.dumps(record) + "\n")

        out_f.flush()
        os.fsync(out_f.fileno())

        n_done_this_run += len(chunk)
        elapsed = time.time() - run_start
        rate = elapsed / n_done_this_run
        remaining = len(todo) - n_done_this_run
        eta_hours = (rate * remaining) / 3600
        total_extrapolated_hours = (rate * len(docs)) / 3600
        chunk_seconds = time.time() - t_chunk0
        print(f"[edits] chunk of {len(chunk)} done in {chunk_seconds:.1f}s "
              f"({chunk_seconds/len(chunk):.2f}s/doc this chunk) | "
              f"{n_done_this_run}/{len(todo)} this run "
              f"({len(done)+n_done_this_run}/{len(docs)} total) | "
              f"avg {rate:.2f}s/doc | ETA {eta_hours:.2f}h | "
              f"extrapolated full-batch {total_extrapolated_hours:.2f}h "
              f"(gate threshold 8h)", flush=True)

    out_f.close()
    total_elapsed_h = (time.time() - run_start) / 3600
    print(f"\n=== BATCH {args.batch} RUN SUMMARY ===", flush=True)
    print(f"Processed {len(todo)} docs this run in {total_elapsed_h:.2f}h "
          f"(gen_batch_size={args.gen_batch_size})", flush=True)
    print(f"Below-MIN_TOKENS({MIN_TOKENS}) counts by attack type: {n_below_min_tokens}", flush=True)
    if args.batch == 1 and len(done) == 0:
        gate = "PASS" if total_elapsed_h < 8.0 else "FAIL"
        print(f"GATE: {gate} ({total_elapsed_h:.2f}h vs 8h threshold) -- "
              f"{'proceed to batch 2' if gate == 'PASS' else 'batch 2 deferred per the runtime gate'}",
              flush=True)


if __name__ == "__main__":
    main()
