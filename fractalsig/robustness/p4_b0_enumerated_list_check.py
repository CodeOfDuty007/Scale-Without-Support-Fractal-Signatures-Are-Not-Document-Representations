#!/usr/bin/env python3
"""B0 (2026-08-27): closes the enumerated-list degeneration
question left unresolved by Test 7. Test 7's coarse proxy (>=3 numbered
lines) gave 5.7-10.3% pooled but conflates two cases: real degeneration
("(1) X occurred... (2) X resulted... (3) X is significant...", same
content restated with an incrementing number, which defeats the 5-gram
repeat filter by construction) vs legitimate structure (billsum/big_patent
are naturally numbered by domain).

Decisive test (specified): for every doc with >=3 numbered lines in
one of the 4 constrained generation types, compute pairwise embedding
cosine similarity BETWEEN the numbered items themselves.
  - high inter-item similarity -> same content restated -> real degeneration
  - low inter-item similarity  -> distinct content per item -> legit structure

Pre-registered threshold: mean pairwise cosine >= 0.80 = degenerate.
(Rationale: near-paraphrase restatements of the same subject/entity score
high under MiniLM embeddings; distinct legitimate list items -- e.g.
separate patent claims or separate bill provisions -- share topical
vocabulary but diverge in predicate/content, empirically well under 0.80
for MiniLM sentence embeddings on this kind of text.) Reports the full
distribution per type/domain, not just the binary call, so the threshold
choice is visible and checkable.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = _paths.WORKSPACE_STR
D = lambda *p: os.path.join(HERE, "data", *p)

CONSTRAINED_TYPES = ["paraphrase_light", "paraphrase_heavy", "style_transfer", "summarize_then_expand"]
NUMBERED_LINE_RE = re.compile(r"^\s*\(?(\d{1,2})[\.\)]\s+", re.MULTILINE)
MIN_NUMBERED_LINES = 3
DEGENERATE_THRESHOLD = 0.80


def extract_numbered_items(text):
    """Split text into (start_idx, item_text) per numbered line, item_text
    running to the next numbered line or end of text."""
    matches = list(NUMBERED_LINE_RE.finditer(text))
    if len(matches) < MIN_NUMBERED_LINES:
        return None
    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item_text = text[start:end].strip()
        if item_text:
            items.append(item_text)
    return items if len(items) >= MIN_NUMBERED_LINES else None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    print(f"[b0] device={device}", flush=True)

    docs = []
    for path in ("p4_edits_batch1.jsonl", "p4_edits_batch2.jsonl"):
        for l in open(D(path)):
            docs.append(json.loads(l))
    print(f"[b0] {len(docs)} docs loaded", flush=True)

    candidates = []  # (id, domain, vtype, items)
    for doc in docs:
        for vtype in CONSTRAINED_TYPES:
            v = doc["variants"].get(vtype)
            if v is None or not v.get("text"):
                continue
            items = extract_numbered_items(v["text"])
            if items is not None:
                candidates.append((doc["id"], doc["domain"], vtype, items))

    print(f"[b0] {len(candidates)} docs have >={MIN_NUMBERED_LINES} numbered lines "
          f"(matches Test 7's coarse proxy population)", flush=True)

    results = []
    for i, (doc_id, domain, vtype, items) in enumerate(candidates):
        emb = embedder.encode(items, convert_to_numpy=True, show_progress_bar=False)
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim_matrix = norm @ norm.T
        n = len(items)
        iu = np.triu_indices(n, k=1)
        mean_pairwise = float(sim_matrix[iu].mean())
        results.append({"id": doc_id, "domain": domain, "variant": vtype,
                         "n_items": n, "mean_pairwise_cosine": mean_pairwise,
                         "degenerate": mean_pairwise >= DEGENERATE_THRESHOLD})
        if (i + 1) % 200 == 0:
            print(f"[b0]   {i+1}/{len(candidates)}", flush=True)

    with open(D("p4_b0_enumerated_list_check.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'type':22}{'domain':18}{'n_flagged':>10}{'n_degenerate':>14}{'rate':>8}{'mean_cos':>10}")
    by_type_domain = defaultdict(list)
    for r in results:
        by_type_domain[(r["variant"], r["domain"])].append(r)
    summary = {}
    for (vtype, domain), rs in sorted(by_type_domain.items()):
        n = len(rs)
        n_deg = sum(1 for r in rs if r["degenerate"])
        rate = n_deg / n
        mean_cos = float(np.mean([r["mean_pairwise_cosine"] for r in rs]))
        print(f"{vtype:22}{domain:18}{n:10}{n_deg:14}{rate:8.3f}{mean_cos:10.4f}")
        summary[f"{vtype}/{domain}"] = {"n_flagged": n, "n_degenerate": n_deg,
                                          "degenerate_rate": rate, "mean_cosine": mean_cos}

    # overall by type
    print(f"\n{'type':22}{'n_flagged':>10}{'n_degenerate':>14}{'rate_of_all_candidates':>24}")
    n_docs_by_type = defaultdict(int)
    for doc in docs:
        for vtype in CONSTRAINED_TYPES:
            v = doc["variants"].get(vtype)
            if v is not None and v.get("text"):
                n_docs_by_type[vtype] += 1
    by_type = defaultdict(list)
    for r in results:
        by_type[r["variant"]].append(r)
    for vtype in CONSTRAINED_TYPES:
        rs = by_type.get(vtype, [])
        n_flagged = len(rs)
        n_deg = sum(1 for r in rs if r["degenerate"])
        total = n_docs_by_type[vtype]
        rate_of_all = n_deg / total if total else 0
        print(f"{vtype:22}{n_flagged:10}{n_deg:14}{rate_of_all:24.4f}")
        summary[f"{vtype}/OVERALL"] = {"n_flagged": n_flagged, "n_degenerate": n_deg,
                                         "n_total_docs_this_type": total,
                                         "degenerate_rate_of_all_docs": rate_of_all}

    with open(D("p4_b0_enumerated_list_summary.json"), "w") as f:
        json.dump({"threshold": DEGENERATE_THRESHOLD, "min_numbered_lines": MIN_NUMBERED_LINES,
                    "summary": summary}, f, indent=2)
    print("\nwritten to data/p4_b0_enumerated_list_check.jsonl, data/p4_b0_enumerated_list_summary.json")


if __name__ == "__main__":
    main()
