#!/usr/bin/env python3
"""
Investigates why 4 of 8 GAGLE prompt_types (excerpt, short_keywords,
summary_keywords, cot) do not reconstruct under either of the two rules
already confirmed for the other 4 types + the human side:
  - human side / continue_it / continue_pt: prefix_ids + text_ids[1:],
    length-matched trim from the prefix end when off by a token or two.
  - keywords / summary: text_ids alone, capped at 2048 tokens.

Both candidates were tried on the 4 failing types (ad hoc,
not previously scripted) and neither reproduced gagle's stored gemma-2b
scores: prefix+text gave token-length deltas of 70-280 (too long), while
text-alone gave small deltas of 8-83 (closer, but still wrong) with r
between 0.03-0.9 depending on doc and alignment tried. Text-alone is the
closer candidate by an order of magnitude in every prior sample, so it is
the only candidate re-tested here with model forward passes; prefix+text
is excluded from the expensive per-doc scoring based on that prior
length evidence (recorded in the report, not re-derived here).

Two-phase design, split because measured throughput (fp32, MPS, gemma-2b)
is 37-90s per forward pass for a ~500 token doc -- a full N=200/type run
across 4 types (800 forward passes) is a 13-40 hour job, infeasible in an
interactive session:
  - Phase "cheap": tokenizer-only checks (length triangulation, excerpt
    substring test, cot marker detection, prefix structure diff against
    the 4 working types) at large N (default 500/type) -- seconds, no
    model load.
  - Phase "model": one forward pass per doc (text-alone, capped at 2048)
    at a much smaller N (default 60/type, override with --model-n),
    feeding an offset sweep + bootstrap CI per doc.

Outputs one JSON file per prompt_type under
$FRACTALSIG_WORKSPACE/alignment_probe/<prompt_type>.json.
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import json
import os
import random
import re
import zipfile

import numpy as np

ZIP_PATH = _paths.GAGLE_ZIP_STR
OUT_DIR = str(_paths.ALIGNMENT_PROBE_DIR)
CAP = 2048
MIN_TOKENS = 320  # from build_p0_local.py -- the P1 pipeline's own usability floor

FAILING_TYPES = ["excerpt", "short_keywords", "summary_keywords", "cot"]
WORKING_TYPES = ["continue_it", "continue_pt", "keywords", "summary"]
ALL_TYPES = FAILING_TYPES + WORKING_TYPES

COT_MARKERS = [
    r"\bstep\s*\d+\b", r"\bfirst[,:]", r"\bfinal answer\b", r"\bin conclusion\b",
    r"^\s*\d+\.\s", r"\blet'?s think\b", r"\btherefore\b",
]

PAT = re.compile(
    r"^(.+?)_(\d+\.\d+)_(.+?)_(continue_it|continue_pt|cot|excerpt|"
    r"keywords|short_keywords|summary_keywords|summary|continue)\.jsonl$"
)


def stratified_sample(target_per_type, types, seed=0):
    """One pass over the zip, streaming each matching member line-by-line
    (not slurped via readlines(), which competed with the model for memory
    at higher N in an earlier version of this script) and stopping each
    stratum as soon as its cap is hit. Trades within-file randomness (first
    k matching docs per stratum, not a shuffled sample of the file) for a
    bounded, low memory footprint -- noted as a sampling caveat in the
    report rather than silently accepted."""
    rng = random.Random(seed)
    n_domains, n_models, n_temps = 5, 2, 3
    per_stratum_cap = -(-target_per_type // (n_domains * n_models * n_temps)) + 2
    buckets = {pt: collections.defaultdict(list) for pt in types}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(m for m in zf.namelist() if m.endswith(".jsonl"))
        for member in members:
            fname = member.split("/")[-1]
            mm = PAT.match(fname)
            if not mm:
                continue
            model, temp, domain, pt = mm.groups()
            if pt not in types:
                continue
            stratum = (domain, model, temp)
            if len(buckets[pt][stratum]) >= per_stratum_cap:
                continue
            with zf.open(member) as fh:
                for line in fh:
                    if len(buckets[pt][stratum]) >= per_stratum_cap:
                        break
                    entry = json.loads(line)
                    scores = entry.get("scores", {}).get("gemma-2b", {})
                    if not scores.get("llm_text") or not entry.get("prefix"):
                        continue
                    buckets[pt][stratum].append({
                        "id": entry.get("_id"),
                        "domain": domain, "model": model, "temp": temp, "split": member.split("/")[1],
                        "prompt_type": pt,
                        "prefix": entry["prefix"],
                        "llm_text": entry["llm_text"],
                        "original_text": entry.get("original_text"),
                        "gagle_scores": scores["llm_text"],
                    })
    samples = {}
    for pt in types:
        flat = [d for stratum_docs in buckets[pt].values() for d in stratum_docs]
        rng.shuffle(flat)
        samples[pt] = flat[:target_per_type] if len(flat) > target_per_type else flat
    return samples


def cheap_checks(doc, tok):
    prefix_ids = tok(doc["prefix"])["input_ids"]
    text_ids = tok(doc["llm_text"])["input_ids"]
    gagle_len = len(doc["gagle_scores"])
    text_alone_scoreable = len(text_ids) - 1
    combined_scoreable = len(prefix_ids) + len(text_ids) - 1 - 1
    rec = {
        "id": doc["id"], "domain": doc["domain"], "model": doc["model"],
        "temp": doc["temp"], "split": doc["split"], "prompt_type": doc["prompt_type"],
        "prefix_len": len(prefix_ids), "text_len": len(text_ids),
        "gagle_len": gagle_len,
        "text_alone_delta": text_alone_scoreable - gagle_len,
        "combined_delta": combined_scoreable - gagle_len,
        "near_min_tokens": gagle_len < MIN_TOKENS + 50,
    }
    if doc["prompt_type"] == "excerpt" and doc.get("original_text"):
        rec["llm_text_is_substring_of_original"] = doc["llm_text"].strip() in doc["original_text"]
        rec["original_is_substring_of_llm_text"] = doc["original_text"].strip() in doc["llm_text"]
    if doc["prompt_type"] == "cot":
        text_lower = doc["llm_text"].lower()
        rec["cot_markers_found"] = [m for m in COT_MARKERS if re.search(m, text_lower, re.MULTILINE)]
    return rec, text_ids


def offset_sweep_for_doc(lps, gagle_scores, center_delta, half_width=100):
    """lps: full log-probs array from a single text-alone forward pass.
    Slides a window of length len(gagle_scores) across lps at offsets
    centered on the length-implied delta, +/- half_width (100, wide enough
    to cover the largest text_alone_delta observed in prior ad hoc testing,
    83 for cot), and returns the best-fitting offset/r/mad plus the full
    offset->r table."""
    from scipy.stats import pearsonr
    g = len(gagle_scores)
    table = {}
    lo = max(0, center_delta - half_width)
    hi = min(len(lps) - g, center_delta + half_width)
    if hi < lo:
        return None
    for off in range(lo, hi + 1):
        window = lps[off:off + g]
        if len(window) != g:
            continue
        r, _ = pearsonr(window, gagle_scores)
        mad = float(np.mean(np.abs(window - gagle_scores)))
        table[off] = {"r": float(r), "mad": mad}
    if not table:
        return None
    best_off = max(table, key=lambda k: table[k]["r"])
    return {"offset_table": table, "best_offset": best_off, "best_r": table[best_off]["r"], "best_mad": table[best_off]["mad"]}


def bootstrap_r_ci(window, gagle_scores, n_boot=200, seed=0):
    from scipy.stats import pearsonr
    rng = np.random.RandomState(seed)
    n = len(window)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        r, _ = pearsonr(window[idx], gagle_scores[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)
    if len(boots) == 0:
        return None
    return {"ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)), "n_boot": len(boots)}


def run_cheap_phase(target_per_type, seed):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-2b")

    print(f"[cheap] sampling ~{target_per_type}/type across {len(ALL_TYPES)} prompt_types (failing + working, for prefix diff)", flush=True)
    samples = stratified_sample(target_per_type, ALL_TYPES, seed=seed)
    for pt, docs in samples.items():
        strata = collections.Counter((d["domain"], d["model"], d["temp"]) for d in docs)
        print(f"[cheap] {pt}: n={len(docs)}, {len(strata)} distinct strata covered", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_records = {}
    for pt, docs in samples.items():
        records = [cheap_checks(d, tok)[0] for d in docs]
        all_records[pt] = records
        with open(os.path.join(OUT_DIR, f"cheap_{pt}.json"), "w") as f:
            json.dump(records, f)
        print(f"[cheap] wrote cheap_{pt}.json ({len(records)} records)", flush=True)
    return all_records


def run_model_phase(target_per_type, seed):
    import time
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained("google/gemma-2b")
    print(f"[model] sampling ~{target_per_type}/type for failing types only (model phase)", flush=True)
    samples = stratified_sample(target_per_type, FAILING_TYPES, seed=seed)
    for pt, docs in samples.items():
        print(f"[model] {pt}: n={len(docs)}", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", dtype=torch.float32).to(device).eval()
    print(f"[model] model loaded on device={device} dtype=float32", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for pt, docs in samples.items():
        out_path = os.path.join(OUT_DIR, f"model_{pt}.json")
        records = []
        for i, doc in enumerate(docs):
            t0 = time.time()
            rec, text_ids = cheap_checks(doc, tok)
            gagle_scores = np.array(doc["gagle_scores"])

            import gc
            ids_capped = text_ids[:CAP]
            ids_t = torch.tensor([ids_capped]).to(device)
            with torch.no_grad():
                logits = model(input_ids=ids_t).logits[0].float()
                n = ids_t.shape[1]
                target_ids = ids_t[0, 1:n]
                target_logits = logits[:n - 1, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
                logZ = torch.logsumexp(logits[:n - 1, :], dim=-1)
                gathered = target_logits - logZ
                lps = gathered.cpu().numpy().copy()
            del logits, gathered, target_logits, logZ, ids_t
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

            sweep = offset_sweep_for_doc(lps, gagle_scores, rec["text_alone_delta"])
            rec["offset_sweep_best_offset"] = sweep["best_offset"] if sweep else None
            rec["offset_sweep_best_r"] = sweep["best_r"] if sweep else None
            rec["offset_sweep_best_mad"] = sweep["best_mad"] if sweep else None
            # store a thinned table (every 5th offset) to keep the json small
            if sweep:
                rec["offset_table_thin"] = {k: v for k, v in sweep["offset_table"].items() if k % 5 == 0}
            if sweep is not None:
                best_off = sweep["best_offset"]
                window = lps[best_off:best_off + len(gagle_scores)]
                rec["bootstrap_ci"] = bootstrap_r_ci(window, gagle_scores)

            rec["elapsed_sec"] = round(time.time() - t0, 2)
            records.append(rec)
            print(f"[model][{pt}] {i+1}/{len(docs)} done in {rec['elapsed_sec']:.1f}s "
                  f"(text_len={rec['text_len']}, best_offset={rec['offset_sweep_best_offset']}, "
                  f"best_r={rec['offset_sweep_best_r']})", flush=True)

            # incremental save every doc, so a killed/timed-out run keeps partial results
            with open(out_path, "w") as f:
                json.dump(records, f)
        print(f"[model] wrote {out_path} ({len(records)} records)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cheap", "model", "both"], default="both")
    ap.add_argument("--cheap-n", type=int, default=500)
    ap.add_argument("--model-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.phase in ("cheap", "both"):
        run_cheap_phase(args.cheap_n, args.seed)
    if args.phase in ("model", "both"):
        run_model_phase(args.model_n, args.seed)


if __name__ == "__main__":
    main()
