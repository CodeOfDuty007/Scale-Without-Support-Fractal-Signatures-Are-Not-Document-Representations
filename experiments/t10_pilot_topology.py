#!/usr/bin/env python3
"""
T10 PILOT -- second representation family (persistent homology on embedding
trajectories), CPU-only, small scale, purpose-built to answer two questions
before any full-corpus / SLURM commitment: (1) is the pipeline even correct
and does it produce a valid document-level feature, (2) what is the real
per-document wall-clock cost on this hardware, so a full run's feasibility
can be judged from measurement rather than guesswork.

Representation: instead of the token-level log-probability score sequence
(what Family A's H/S are computed from), embed OVERLAPPING WORD WINDOWS of
each document's raw text with a small sentence embedding model
(all-MiniLM-L6-v2, CPU-friendly, already used for this role elsewhere in
the pipeline). The resulting sequence of window
embeddings is a trajectory through embedding space -- a genuinely different
representation of the same document, not a variant of the fractal one.

Topological features: build a Vietoris-Rips complex on each document's
embedding-window point cloud (gudhi), compute persistence diagrams for H0
(connected components) and H1 (loops), and reduce each to scalar
document-level features (total persistence, persistence entropy, bar count,
max persistence) -- the topological analogue of H/S.

Reliability check: SAME interleave methodology as T0/T1 (even-indexed
windows -> half A, odd-indexed -> half B; Pearson r pooled and within-stratum
by (Label, domain, temp), via reliability_lib's existing, already-validated
pooled_and_within_stats/demean_by_stratum) -- this is the only way to know
whether the new representation carries document-level signal at all, using
the exact same yardstick already applied to the fractal features throughout
this run.

Sampling: gpu_manifest_shards/*.parquet already carries a `text` column (no
need to rescan gagle.zip). A fixed-seed random subset of shards is read in
full, then a random subset of rows from each is kept, until the target doc
count is reached -- avoids reading all 721 shards for a pilot.
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow.parquet as pq

from fractalsig import paths as _paths  # noqa: E402
from experiments.reliability_lib import pooled_and_within_stats  # noqa: E402

SHARD_DIR = _paths.MANIFEST_SHARD_DIR_STR
OUT_DIR = _paths.STATS_DIR_STR
LOG_EVERY = 200

TARGET_N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
N_SHARDS_TO_READ = 60  # ~60 shards x ~310 docs/shard >> TARGET_N_DOCS, gives room to filter
SEED = 20260908

WINDOW_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
STRIDE_WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MIN_WINDOWS = 8  # need enough points for a meaningful point cloud + a non-trivial split half
RUN_TAG = f"w{WINDOW_WORDS}_s{STRIDE_WORDS}"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256


def make_windows(text):
    """Returns (texts, starts): starts[i] is the word-index start of window i,
    needed to measure inter-half word-position overlap directly (the same
    check that caught block_bootstrap's overlap in T1 -- overlapping windows
    at stride < window_words let adjacent windows land in opposite
    even/odd halves while sharing most of their words, which is a content
    leak between "independent" halves, not a point-count effect)."""
    words = text.split()
    texts, starts = [], []
    i = 0
    while i + WINDOW_WORDS <= len(words):
        texts.append(" ".join(words[i:i + WINDOW_WORDS]))
        starts.append(i)
        i += STRIDE_WORDS
    return texts, starts


def half_word_overlap(starts):
    """Jaccard overlap between the word positions covered by half A
    (even window index) and half B (odd window index), given each window
    covers [start, start+WINDOW_WORDS). Returns None if either half is empty."""
    if len(starts) < 2:
        return None
    max_pos = max(starts) + WINDOW_WORDS
    cov_a = np.zeros(max_pos, dtype=bool)
    cov_b = np.zeros(max_pos, dtype=bool)
    for idx, s in enumerate(starts):
        (cov_a if idx % 2 == 0 else cov_b)[s:s + WINDOW_WORDS] = True
    union = cov_a | cov_b
    if not union.any():
        return None
    inter = cov_a & cov_b
    return float(inter.sum()) / float(union.sum())


def sample_docs():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.parquet")))
    chosen_files = rng.choice(files, size=min(N_SHARDS_TO_READ, len(files)), replace=False)
    docs = []
    for f in chosen_files:
        t = pq.read_table(f, columns=["id", "domain", "Label", "temp", "prompt_type", "text"])
        d = t.to_pydict()
        n = len(d["id"])
        idx = rng.permutation(n)
        for i in idx:
            docs.append({"id": d["id"][i], "domain": d["domain"][i], "Label": d["Label"][i],
                         "temp": d["temp"][i], "prompt_type": d["prompt_type"][i], "text": d["text"][i]})
        if len(docs) >= TARGET_N_DOCS * 2:  # oversample; MIN_WINDOWS filter drops some
            break
    rng.shuffle(docs)
    return docs[:TARGET_N_DOCS * 2]  # filtered down to TARGET_N_DOCS after window-count check


def persistence_features(points):
    """points: (n, d) float32 array. Returns dict of scalar topological features."""
    import gudhi
    if len(points) < 3:
        return None
    rc = gudhi.RipsComplex(points=points)
    st = rc.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    h0 = st.persistence_intervals_in_dimension(0)
    h1 = st.persistence_intervals_in_dimension(1)

    def _finite_lengths(intervals):
        lens = []
        for b, d in intervals:
            if np.isfinite(d):
                lens.append(d - b)
        return np.array(lens, dtype=np.float64)

    def _entropy(lens):
        if len(lens) == 0 or lens.sum() <= 0:
            return 0.0
        p = lens / lens.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h0_lens = _finite_lengths(h0)
    h1_lens = _finite_lengths(h1)
    return {
        "H0_total_persistence": float(h0_lens.sum()),
        "H0_entropy": _entropy(h0_lens),
        "H0_n_bars": int(len(h0_lens)),
        "H1_total_persistence": float(h1_lens.sum()),
        "H1_entropy": _entropy(h1_lens),
        "H1_n_bars": int(len(h1_lens)),
        "H1_max_persistence": float(h1_lens.max()) if len(h1_lens) else 0.0,
    }


def main():
    t0 = time.time()
    print(f"[t10-pilot] target_n_docs={TARGET_N_DOCS} window_words={WINDOW_WORDS} "
          f"stride_words={STRIDE_WORDS} min_windows={MIN_WINDOWS} embed_model={EMBED_MODEL}",
          flush=True)

    docs = sample_docs()
    print(f"[t10-pilot] sampled {len(docs)} candidate docs, {time.time()-t0:.0f}s", flush=True)

    # build all windows for all docs up-front, one flat list for batched embedding
    doc_windows = []  # list of list-of-window-strings, same order as `docs`
    doc_starts = []   # matching list of window start-word-indices, for overlap measurement
    n_kept = 0
    for doc in docs:
        wins, starts = make_windows(doc["text"])
        if len(wins) >= MIN_WINDOWS:
            doc_windows.append(wins)
            doc_starts.append(starts)
            n_kept += 1
        else:
            doc_windows.append(None)
            doc_starts.append(None)
        if n_kept >= TARGET_N_DOCS:
            break
    docs = docs[:len(doc_windows)]
    kept_idx = [i for i, w in enumerate(doc_windows) if w is not None]
    print(f"[t10-pilot] {len(kept_idx)}/{len(docs)} docs have >= {MIN_WINDOWS} windows "
          f"(the rest are too short), {time.time()-t0:.0f}s", flush=True)

    overlaps = [half_word_overlap(doc_starts[i]) for i in kept_idx]
    overlaps = [o for o in overlaps if o is not None]
    mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")
    print(f"[t10-pilot] mean inter-half word-position overlap (Jaccard): {mean_overlap:.4f} "
          f"over {len(overlaps)} docs -- directly measured, not assumed "
          f"(w={WINDOW_WORDS} s={STRIDE_WORDS}: analytically, adjacent windows "
          f"share max(0, w-s) of w words -> ~{max(0, WINDOW_WORDS-STRIDE_WORDS)/WINDOW_WORDS:.2f} "
          f"pairwise fraction at closest spacing)", flush=True)

    flat_windows = []
    owner = []  # index into kept_idx position for each flat window
    for pos, i in enumerate(kept_idx):
        for w in doc_windows[i]:
            flat_windows.append(w)
            owner.append(pos)
    print(f"[t10-pilot] {len(flat_windows)} total windows to embed, {time.time()-t0:.0f}s", flush=True)

    from sentence_transformers import SentenceTransformer
    t_embed0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    embeds = model.encode(flat_windows, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    t_embed = time.time() - t_embed0
    print(f"[t10-pilot] embedding done: {len(flat_windows)} windows, {t_embed:.0f}s "
          f"({t_embed/len(flat_windows)*1000:.2f} ms/window), {time.time()-t0:.0f}s total", flush=True)

    # regroup embeddings by document
    per_doc_embeds = [[] for _ in kept_idx]
    for e, pos in zip(embeds, owner):
        per_doc_embeds[pos].append(e)

    t_ph0 = time.time()
    rows = []
    for pos, i in enumerate(kept_idx):
        pts = np.asarray(per_doc_embeds[pos], dtype=np.float32)
        n = len(pts)
        half_a_idx = np.arange(0, n, 2)
        half_b_idx = np.arange(1, n, 2)
        if len(half_a_idx) < 3 or len(half_b_idx) < 3:
            continue
        fa = persistence_features(pts[half_a_idx])
        fb = persistence_features(pts[half_b_idx])
        if fa is None or fb is None:
            continue
        d = docs[i]
        rows.append({"id": d["id"], "Label": d["Label"], "domain": d["domain"], "temp": d["temp"],
                     "n_windows": n, "half_a": fa, "half_b": fb})
        if (pos + 1) % LOG_EVERY == 0:
            print(f"[t10-pilot] persistence computed for {pos+1}/{len(kept_idx)} docs, "
                  f"{time.time()-t_ph0:.0f}s", flush=True)
    t_ph = time.time() - t_ph0
    print(f"[t10-pilot] persistence homology done: {len(rows)} usable docs, {t_ph:.0f}s "
          f"({t_ph/max(len(rows),1)*1000:.2f} ms/doc), {time.time()-t0:.0f}s total", flush=True)

    json.dump(rows, open(os.path.join(OUT_DIR, f"t10_pilot_topology_raw_{RUN_TAG}.json"), "w"))

    feat_keys = ["H0_total_persistence", "H0_entropy", "H0_n_bars",
                 "H1_total_persistence", "H1_entropy", "H1_n_bars", "H1_max_persistence"]
    reliability = {}
    for key in feat_keys:
        a_vals = [r["half_a"][key] for r in rows]
        b_vals = [r["half_b"][key] for r in rows]
        strata = [(r["Label"], r["domain"], r["temp"]) for r in rows]
        s = pooled_and_within_stats(a_vals, b_vals, strata)
        if s is not None:
            reliability[key] = s
            print(f"[t10-pilot] {key}: rho_pooled={s['rho_pooled']:.4f} "
                  f"rho_within={s['rho_within']:.4f} n={s['n']}", flush=True)

    out = {
        "config": {"target_n_docs": TARGET_N_DOCS, "window_words": WINDOW_WORDS,
                   "stride_words": STRIDE_WORDS, "min_windows": MIN_WINDOWS,
                   "embed_model": EMBED_MODEL, "seed": SEED},
        "mean_inter_half_word_overlap_jaccard": mean_overlap,
        "n_docs_sampled": len(docs), "n_docs_with_enough_windows": len(kept_idx),
        "n_docs_usable_for_reliability": len(rows),
        "n_windows_total": len(flat_windows),
        "timing_sec": {"embedding": t_embed, "persistence_homology": t_ph, "total": time.time() - t0,
                        "ms_per_window_embed": t_embed / len(flat_windows) * 1000,
                        "ms_per_doc_ph": t_ph / max(len(rows), 1) * 1000},
        "reliability": reliability,
    }
    out_path = os.path.join(OUT_DIR, f"t10_pilot_topology_{RUN_TAG}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)
    print(f"[t10-pilot] TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
