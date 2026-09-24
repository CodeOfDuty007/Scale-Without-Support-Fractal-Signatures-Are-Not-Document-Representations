#!/usr/bin/env python3
"""
T2 Step 2 -- real-text bracket, one CPU pass over gagle.zip.

Manifest shards (gpu_manifest_shards/) only hold the 320-400-token SLICED
score arrays, so they can't support a length sweep. gagle.zip can: the
human side (`scores["gemma-2b"]["original_text"]`) is stored UNSLICED, up to
a hard 2047-token cap (fractalsig/pipeline/build_p0_local.py's own extraction rule,
confirmed in the internal verification notes). All
5,000 human articles in the corpus have this (confirmed in
the internal verification notes). Only gemma-2b is used here --
build_p0_local.py only ever extracts the human side for gemma-2b; a separate
Mistral human-side scoring pass exists (data/human_familyb_mistral_7b.jsonl)
but is not needed for this bracket since Gemma_S/Gemma_H are the paper's
primary quantities.

For each deduped human document (first occurrence wins, matching
build_p0_local.py's own precedent: sorted member order puts a
gemma_2b/continue_it member first for every article), and for each N in a
grid capped by what a ~2047-token document can actually support with >=2
disjoint blocks, cut DISJOINT (non-overlapping) N-token blocks and compute
Shat/Hhat per block via the project's own unmodified fractal_math functions
(module default scale list, matching production point-estimate calls).
Within-document variance of Shat/Hhat across blocks of the SAME document is NOT a
clean bound on Var(eta_N) in either direction (it is not an upper bound). It differs from
fBm's Var(eta_N) two ways that push opposite directions: (+) it includes real
within-document nonstationarity, which fBm's fixed-H generator has none of; (-) blocks
of ONE document share that document's own particular realization (its actual
token-score autocorrelation and trajectory), which SUPPRESSES between-block variance
relative to independent draws from a generator -- fBm's Var(eta_N) is measured across
independent realizations, this is measured within a single one. Empirically the
suppression wins: at matched N, real-text block variance runs BELOW the fBm floor for
both S and H (see t2_step2_full_doc_estimates.json for the
numbers and the calibration check -- full-document Hhat in this sample is centered at
0.629 [p10=0.578,p90=0.698], confirming the underlying process does sit in the fBm
floor's calibrated 0.60-0.72 band, so the gap is this shared-realization effect, not a
calibration mismatch). Treat this as a DISTINCT estimand from Var(eta_N), not a bound
on it in either direction.

Reports usable n (documents contributing >=2 blocks) at every N -- it falls
as N grows, and the plan says a reviewer will check.
"""
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile

import numpy as np

from fractalsig import paths as _paths  # noqa: E402
from fractalsig.runtime import default_workers  # noqa: E402
from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S  # noqa: E402

ZIP_PATH = _paths.GAGLE_ZIP_STR
OUT_DIR = _paths.STATS_DIR_STR
SCORER = "gemma-2b"
N_GRID = [100, 200, 400, 800, 1600]  # 1600 only usable for the longest human docs


KNOWN_TOTAL_HUMAN_DOCS = 5000  # confirmed in the internal verification notes


def extract_human_docs():
    """One pass over gagle.zip; dedup human docs by (domain, dataset_id),
    first occurrence wins (matches build_p0_local.py's own precedent:
    sorted namelist puts gemma_2b_*_continue_it first per (split,domain)
    group whenever one exists). Early-exits once all 5,000 known human
    articles are found (review-identified optimization: the corpus has a
    fixed, already-confirmed human-doc count, so scanning past that point
    can only re-find duplicates)."""
    docs = {}
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = sorted(n for n in zf.namelist() if n.startswith("gagle3/") and n.endswith(".jsonl"))
        print(f"[t2-step2] {len(members)} members to scan", flush=True)
        for mi, member in enumerate(members, 1):
            with zf.open(member) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    domain = entry.get("dataset_name")
                    dsid = entry.get("dataset_id")
                    key = (domain, dsid)
                    if key in docs:
                        continue
                    scores_block = entry.get("scores", {}).get(SCORER, {})
                    human_raw = scores_block.get("original_text")
                    if not human_raw or len(human_raw) < 100:
                        continue
                    docs[key] = human_raw
            if mi % 100 == 0:
                print(f"[t2-step2]   {mi}/{len(members)} members scanned, "
                      f"{len(docs):,} unique human docs so far, {time.time()-t0:.0f}s", flush=True)
            if len(docs) >= KNOWN_TOTAL_HUMAN_DOCS:
                print(f"[t2-step2] reached all {KNOWN_TOTAL_HUMAN_DOCS} known human docs at "
                      f"member {mi}/{len(members)}, {time.time()-t0:.0f}s -- stopping scan early", flush=True)
                break
    print(f"[t2-step2] done scanning: {len(docs):,} unique human docs, {time.time()-t0:.0f}s", flush=True)
    return docs


def _process_doc(args):
    key, raw = args
    arr = np.asarray(raw, dtype=np.float64)
    out = {}
    for n in N_GRID:
        n_blocks = len(arr) // n
        if n_blocks < 2:
            continue
        s_hats, h_hats = [], []
        for b in range(n_blocks):
            block = arr[b * n:(b + 1) * n]
            traj, incr = generate_trajectory(block)
            try:
                s_hat, _ = calc_S(traj)
                h_hat, _ = calc_H(incr)
            except Exception:
                continue
            if s_hat == s_hat:
                s_hats.append(s_hat)
            if h_hat == h_hat:
                h_hats.append(h_hat)
        if len(s_hats) >= 2:
            out[f"S_{n}"] = (float(np.var(s_hats)), int(len(s_hats)))
        if len(h_hats) >= 2:
            out[f"H_{n}"] = (float(np.var(h_hats)), int(len(h_hats)))
    return out


def main():
    t0 = time.time()
    docs = extract_human_docs()
    cache_path = os.path.join(OUT_DIR, "t2_step2_human_docs_cache_meta.json")
    json.dump({"n_unique_human_docs": len(docs), "lengths": sorted(len(v) for v in docs.values())[:10] +
               ["..."] + sorted(len(v) for v in docs.values())[-10:]}, open(cache_path, "w"), indent=2)

    items = list(docs.items())
    with mp.get_context("fork").Pool(processes=default_workers()) as pool:
        all_results = list(pool.imap_unordered(_process_doc, items, chunksize=20))
    print(f"[t2-step2] block-variance computation done, {time.time()-t0:.0f}s", flush=True)

    agg = {}
    for res in all_results:
        for k, (var_val, n_blocks) in res.items():
            agg.setdefault(k, []).append(var_val)

    out = {"N_grid": N_GRID, "n_human_docs_total": len(docs), "per_quantity": {}}
    for k, vals in sorted(agg.items()):
        vals = np.array(vals)
        out["per_quantity"][k] = {
            "n_docs_usable": int(len(vals)),
            "mean_within_doc_variance": float(np.mean(vals)),
            "median_within_doc_variance": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)), "p75": float(np.percentile(vals, 75)),
        }
        print(f"{k}: n_usable={len(vals)}  mean_within_doc_var={np.mean(vals):.5f}  "
              f"median={np.median(vals):.5f}", flush=True)

    out_path = os.path.join(OUT_DIR, "t2_step2_real_text_bracket.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[t2-step2] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
