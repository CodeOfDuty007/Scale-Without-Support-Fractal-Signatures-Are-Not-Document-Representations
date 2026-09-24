"""F-5(a) characterization: which 7,397 LLM rows did process_doc() drop, and how
do they differ from the retained population.

Single pass over gagle.zip (LLM side only -- human ids need X-01's dedup logic,
out of scope here and irrelevant to this question). Mirrors
build_p1_feature_store_cpu.process_doc's drop test exactly (reuses extract_docs
and slice_scores rather than reimplementing them, per the project's own
"verify the code does what the description says, don't reimplement" rule).

A doc is dropped iff slice_scores() returns None for BOTH scorers -- i.e.
GAGLE's own precomputed score array is under MIN_TOKENS=320 for both. This is
independent of the document's own tokenization length (that's F-5(b), a
different mechanism already ablated).

Outputs:
  data/f5a_dropped_llm_rows.jsonl   -- full doc (id, text, raw_scores, metadata)
                                        for exactly the dropped rows, so a later
                                        self-scoring pass doesn't need a second
                                        zip read.
  data/f5a_retained_llm_stats.jsonl -- lightweight per-row stats (no text) for
                                        every retained LLM row, for the
                                        dropped-vs-retained distribution comparison.
  data/f5a_summary.json             -- counts, the 7,397 check, and the
                                        comparison distributions.
"""
from fractalsig import paths as _paths  # noqa: E402
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

from fractalsig.pipeline.build_p0_local import slice_scores, MIN_TOKENS  # noqa: E402
from fractalsig.pipeline.build_p1_feature_store_cpu import extract_docs, SCORERS  # noqa: E402

ZIP_PATH = os.environ.get("GAGLE_ZIP_PATH", _paths.GAGLE_ZIP_STR)
MEMBER_PATTERN_PREFIX = "gagle3/"
OUT_DROPPED = os.path.join(_paths.WORKSPACE_STR, "data", "f5a_dropped_llm_rows.jsonl")
OUT_RETAINED_STATS = os.path.join(_paths.WORKSPACE_STR, "data", "f5a_retained_llm_stats.jsonl")
OUT_SUMMARY = os.path.join(_paths.WORKSPACE_STR, "data", "f5a_summary.json")


def is_dropped(raw_scores):
    sliced = {sc: slice_scores(raw_scores.get(sc)) for sc in SCORERS}
    return not any(v is not None for v in sliced.values())


def main():
    zf = zipfile.ZipFile(ZIP_PATH)
    members = sorted(m for m in zf.namelist() if m.startswith(MEMBER_PATTERN_PREFIX) and m.endswith(".jsonl"))
    print(f"[f5a] {len(members)} members to scan", flush=True)

    n_dropped = 0
    n_retained = 0
    json_errors = 0

    dropped_by_domain = Counter()
    retained_by_domain = Counter()
    dropped_by_prompt_type = Counter()
    retained_by_prompt_type = Counter()
    dropped_by_label = Counter()
    retained_by_label = Counter()
    dropped_word_counts = []
    retained_word_counts = []

    fdrop = open(OUT_DROPPED, "w")
    fret = open(OUT_RETAINED_STATS, "w")

    for mi, member in enumerate(members, 1):
        split = member.split("/")[1]
        with zf.open(member) as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    json_errors += 1
                    continue
                for doc in extract_docs(entry, split):
                    if doc["label"] == "Human":
                        continue  # F-5(a) is LLM-only; human ids need X-01 dedup, out of scope
                    wc = len(doc["text"].split()) if doc.get("text") else 0
                    dropped = is_dropped(doc["raw_scores"])
                    if dropped:
                        n_dropped += 1
                        dropped_by_domain[doc["domain"]] += 1
                        dropped_by_prompt_type[doc["prompt_type"]] += 1
                        dropped_by_label[doc["label"]] += 1
                        dropped_word_counts.append(wc)
                        rec = dict(doc)
                        rec["raw_scores_lens"] = {sc: (len(doc["raw_scores"].get(sc)) if doc["raw_scores"].get(sc) else None)
                                                   for sc in SCORERS}
                        fdrop.write(json.dumps(rec) + "\n")
                    else:
                        n_retained += 1
                        retained_by_domain[doc["domain"]] += 1
                        retained_by_prompt_type[doc["prompt_type"]] += 1
                        retained_by_label[doc["label"]] += 1
                        retained_word_counts.append(wc)
                        fret.write(json.dumps({
                            "id": doc["id"], "domain": doc["domain"], "prompt_type": doc["prompt_type"],
                            "temp": doc["temp"], "label": doc["label"], "word_count": wc,
                        }) + "\n")
        if mi % 100 == 0 or mi == len(members):
            print(f"[f5a] {mi}/{len(members)} members  dropped={n_dropped}  retained={n_retained}", flush=True)

    fdrop.close()
    fret.close()

    def pctiles(xs):
        if not xs:
            return {}
        s = sorted(xs)
        n = len(s)
        return {
            "n": n, "min": s[0], "max": s[-1],
            "p10": s[int(0.10 * n)], "p25": s[int(0.25 * n)],
            "median": s[int(0.50 * n)], "p75": s[int(0.75 * n)], "p90": s[int(0.90 * n)],
            "mean": sum(s) / n,
        }

    summary = {
        "zip_path": ZIP_PATH,
        "n_members": len(members),
        "json_errors": json_errors,
        "n_dropped_llm": n_dropped,
        "n_retained_llm": n_retained,
        "n_total_llm": n_dropped + n_retained,
        "expected_dropped_from_f5a": 7397,
        "matches_7397": n_dropped == 7397,
        "dropped_word_count_dist": pctiles(dropped_word_counts),
        "retained_word_count_dist": pctiles(retained_word_counts),
        "dropped_by_domain": dict(dropped_by_domain),
        "retained_by_domain": dict(retained_by_domain),
        "dropped_by_prompt_type": dict(dropped_by_prompt_type),
        "retained_by_prompt_type": dict(retained_by_prompt_type),
        "dropped_by_label": dict(dropped_by_label),
        "retained_by_label": dict(retained_by_label),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
