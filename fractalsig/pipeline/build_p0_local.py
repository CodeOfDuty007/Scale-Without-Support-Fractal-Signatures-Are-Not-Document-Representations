#!/usr/bin/env python3
"""
Single-pass, resumable P0 corpus builder reading gagle.zip LOCALLY (no
network in the read loop). Supersedes build_p0_corpus.py, whose streaming
.shard() approach was reading byte-ranges into ibomohsin/gagle's single
remote gagle.zip and stalled unrecoverably at a fixed byte offset -- no
--num-shards value fixed it, because there was only ever one file.

SCHEMA (verified against the dataset's own GAGLE_Dataset.ipynb, which
outranks the andsteing/gagle HF Space's decompiled bundle.js as a
reference since it's the authors' own derivable code, not a minified UI
bundle):
  - Zip members live at gagle3/{split}/{model}_{temp}_{dataset}_{prompt_type}.jsonl
    -- each member IS one (model, temp, domain, prompt_type) stratification
    cell. This is why checkpointing is done per-member: it's the dataset's
    own natural unit of work, not an arbitrary reshard.
  - Each line is one JSON record with scores[SCORER]["llm_text"] /
    ["original_text"] as PLAIN LISTS of per-token floats (confirmed live,
    both via datasets.load_dataset streaming and via the notebook's own
    pretty_print_array usage -- NOT the nested token_ids/token_texts/
    token_scores dict shown in the HF dataset-card README, which appears to
    describe a different auto-generated viewer schema).

WHY Holder_Exp (S) IS NOT IN THE PER-DOC OUTPUT
  Validated against synthetic fBm with known Hurst exponent: per-document
  (~400 token) S is biased -0.26 to -0.48 regardless of bootstrap draw
  count -- there is not enough information in one document for
  p_epsilon(tau) to resolve a power law. S is only recoverable pooled
  across many documents (bias -0.004 to -0.065, R^2 0.91-0.995), which is
  also how the reference notebook itself applies it (concatenated across a
  document range, never per document). So per-doc rows carry Hurst_Exp
  only; corpus-level S/H per (domain, label) cell is computed once, after
  all members are processed, from pooled raw scores.

WHY calc_H's per-doc row IS kept
  calc_H (this repo's version: non-overlapping len//n chunks) was checked
  against the reference's get_hurst_exponent (samples_per_doc random
  overlapping windows) on synthetic fBm across 30 trials at H=0.3/0.5/
  0.65/0.8: mean bias between the two agreed within ~0.01 at every H,
  well inside each estimator's own run-to-run noise. The chunking
  difference changes variance, not bias -- cosmetic, not a defect.

CHECKPOINTING
  data/p0_local_member_checkpoint.txt: one line per FULLY processed zip
  member, fsynced after each member completes. A member is the atomic unit
  of work -- if the process dies mid-member, the whole member is redone on
  restart (cheap: it's a local read of at most a few thousand JSON lines),
  rather than tracking per-document progress within a member.

  Corpus-level pooling (raw score arrays grouped by (domain, label)) lives
  only in memory during a run. On resume, already-checkpointed members are
  replayed (read + accumulate only, no per-doc compute or output-file
  writes) to rebuild the in-memory pools before continuing with new
  members. This avoids persisting a second multi-GB raw-score cache to
  disk -- replay of local, already-downloaded members is fast.

USAGE
    python -m fractalsig.pipeline.build_p0_local                 # full run (resumable)
    python -m fractalsig.pipeline.build_p0_local --limit-members 2   # smoke test: only process
                                                    # the first N not-yet-done
                                                    # members, then stop
                                                    # (checkpoints normally,
                                                    # safe to continue after)
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import fnmatch
import json
import os
import sys
import zipfile

import numpy as np

from fractalsig.features.fractal_math import generate_trajectory, calc_H, calc_S

ZIP_PATH = os.environ.get("GAGLE_ZIP_PATH", _paths.GAGLE_ZIP_STR)
MEMBER_PATTERN = "gagle3/*/*.jsonl"

DATA_DIR = os.path.join(_paths.WORKSPACE_STR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p0_local_full_results.jsonl")
MEMBER_CKPT_PATH = os.path.join(DATA_DIR, "p0_local_member_checkpoint.txt")
MANIFEST_PATH = os.path.join(DATA_DIR, "p0_local_manifest.json")
CORPUS_SH_PATH = os.path.join(DATA_DIR, "p0_corpus_level_SH.json")

MIN_TOKENS = 320
WARMUP_DROP = 64
CLIP_TO = 464
SCORER = "gemma-2b"


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_line_set(path):
    s = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    s.add(line)
    return s


def append_line(path, value):
    with open(path, "a") as f:
        f.write(str(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def slice_scores(raw):
    if not raw:
        return None
    if len(raw) >= CLIP_TO:
        sliced = raw[WARMUP_DROP:CLIP_TO]
    else:
        sliced = raw
    if len(sliced) < MIN_TOKENS:
        return None
    return sliced


def extract_rows(entry, split):
    """Yield 0-2 row dicts (llm side, human side) from one raw GAGLE record.

    split (train/val/test) is threaded through and still recorded on every
    output row (needed downstream to reconstruct GAGLE's own partition),
    but is deliberately NOT part of the human dedup key (see below) -- X-01,
    2026-08-22. The comment that used to justify including split here
    ("dataset_id can name a different article across splits") was checked
    directly against gagle.zip and found false: the internal verification notes
    ran a full 720-member census and found all 4,474 duplicated
    (domain, dataset_id) pairs are byte-identical text across splits, not
    coincidental id collisions. Keying on split as well as domain+dataset_id
    let the same real article survive dedup as up to 3 separate rows
    (human N inflated ~2.34x; P3 would train and test on the same
    articles). See the id-construction comment on the human yield below for
    how dropping split still avoids re-introducing the OTHER thing split
    was accidentally also guarding against (deviant-source rows winning).
    """
    domain = entry.get("dataset_name")
    scores_block = entry.get("scores", {}).get(SCORER, {})

    try:
        llm_raw = scores_block["llm_text"]
    except (KeyError, TypeError):
        llm_raw = None
    llm_sliced = slice_scores(llm_raw)
    if llm_sliced is not None:
        yield {
            "id": str(entry.get("_id")),
            "split": split,
            "domain": domain,
            "label": entry.get("model", "Unknown_LLM"),
            "prompt_type": entry.get("prompt_type"),
            "temp": entry.get("temperature"),
            "scores": llm_sliced,
        }

    # Human side recurs across every prompt_type/temp file for the same
    # domain, AND across splits (X-01) -- key on (domain, dataset_id) only,
    # so the corpus-level Human pool and the per-doc output both dedup to
    # one row per real article, corpus-wide. domain MUST stay part of the
    # key: GAGLE's dataset_id numbering is not domain-scoped -- verified
    # directly against the real zip that 239 of 11,457 (split, dataset_id)
    # keys are shared between big_patent and wikipedia (e.g.
    # ('test', '00002-of-00256__21') names a different real article in
    # each). Without domain in the key, whichever domain's row was
    # processed first would silently steal the id and the other domain's
    # actual human doc would be wrongly skipped as a "duplicate".
    #
    # Dropping split from the key means "first occurrence wins" now decides
    # WHICH of up to 3 duplicate copies survives -- and that must not be an
    # arbitrary one, because Defect C found two source-member families whose
    # gemma-2b score array deviates from the majority (summary_keywords
    # members, and mistral_7b_* members: the internal verification notes
    # section 4). This is handled WITHOUT extra branching logic here: the
    # caller (process_member/main) iterates zip members via sorted(namelist),
    # and "gagle3/{split}/{model}_{temp}_{domain}_{prompt_type}.jsonl" sorts
    # gemma_2b_* before mistral_7b_* and continue_it before summary_keywords
    # (alphabetically first/last respectively among the 8 prompt types) for
    # every (split, domain) group that contains them. So the first member to
    # offer a given (domain, dataset_id) human doc is always a continue_it,
    # gemma_2b-sourced one when one exists in the corpus for that article --
    # which section 1.3 of the same report confirms is true for all 5,000
    # articles (0 need excluding). main() asserts this invariant in code
    # after the run (human_source_member check) rather than trusting the
    # sort order silently -- see X-01.
    try:
        human_raw = scores_block["original_text"]
    except (KeyError, TypeError):
        human_raw = None
    human_sliced = slice_scores(human_raw)
    if human_sliced is not None:
        yield {
            "id": f"{domain}_{entry.get('dataset_id')}_human",
            "split": split,
            "domain": domain,
            "label": "Human",
            "prompt_type": "continue",
            "temp": "Human",
            "scores": human_sliced,
        }


def is_deviant_member(member):
    """True if this member's source family carries the corrupted gemma-2b
    human score array Defect C identified (summary_keywords-sourced or any
    mistral_7b_*-sourced member) -- the internal verification notes
    section 4."""
    stem = member.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.endswith("summary_keywords") or "mistral_7b" in member.rsplit("/", 1)[-1]


def process_member(zf, member, groups, human_best, stats, write_output=True):
    """Returns a list of LLM-side output rows (empty if write_output=False).
    Human-side rows are NOT returned or written here -- see main()'s final
    flush and its docstring note for why (X-01: picking the best of several
    duplicate copies requires having seen all of them first, which isn't
    known until the whole corpus has been scanned).

    LLM rows are buffered and returned rather than written directly so the
    caller can write+fsync+checkpoint a member atomically: if the process
    is killed partway through a member, nothing from that member has been
    written yet, so redoing it on restart can't produce duplicate rows in
    the output file.
    """
    split = member.split("/")[1]  # gagle3/{split}/{model}_{temp}_{dataset}_{prompt_type}.jsonl
    out_rows = []
    with zf.open(member) as fh:
        for line_bytes in fh:
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            stats["raw_entries_seen"] += 1
            for row in extract_rows(entry, split):
                if row["label"] == "Human":
                    deviant = is_deviant_member(member)
                    existing = human_best.get(row["id"])
                    if existing is None:
                        human_best[row["id"]] = {"row": row, "member": member, "deviant": deviant}
                    else:
                        stats["skipped_dup_human"] += 1
                        if existing["deviant"] and not deviant:
                            # A cleaner copy of this article showed up later in
                            # sort order than a deviant one did -- upgrade.
                            human_best[row["id"]] = {"row": row, "member": member, "deviant": deviant}
                    continue

                # LLM side: no dedup needed (LLM rows are corpus-unique --
                # 218,834 rows, 218,832 distinct texts), stream as before.
                groups[(row["domain"], row["label"])].append(np.asarray(row["scores"], dtype=np.float64))
                if write_output:
                    trajectory, increment = generate_trajectory(row["scores"])
                    H, H_r2 = calc_H(increment)
                    out_rows.append({
                        "id": row["id"],
                        "split": row["split"],
                        "domain": row["domain"],
                        "Label": row["label"],
                        "prompt_type": row["prompt_type"],
                        "temp": row["temp"],
                        "Hurst_Exp": float(H),
                        "Hurst_R2": float(H_r2),
                    })
                    stats["written"] += 1
    return out_rows


def compute_corpus_level_SH(groups):
    results = {}
    for (domain, label), score_lists in sorted(groups.items()):
        if len(score_lists) < 10:
            continue
        # Build increments first, concatenate, compute H, then free before
        # building trajectories for calc_S -- avoids holding both the
        # per-doc trajectory list and increment list in memory at once,
        # which matters most on the largest cells after hours of a run.
        increments = [generate_trajectory(s)[1] for s in score_lists]
        all_incr = np.concatenate(increments)
        H, H_r2 = calc_H(all_incr)
        del increments, all_incr
        trajectories = [generate_trajectory(s)[0] for s in score_lists]
        S, S_r2 = calc_S(trajectories)
        n_tokens = sum(len(s) for s in score_lists)
        results[f"{domain}|{label}"] = {
            "domain": domain,
            "label": label,
            "n_docs": len(score_lists),
            "n_tokens": n_tokens,
            "S": float(S),
            "S_R2": float(S_r2),
            "H": float(H),
            "H_R2": float(H_r2),
        }
        print(f"[corpus-level] {domain:20s} {label:12s} n_docs={len(score_lists):6d} "
              f"S={S:.4f} S_R2={S_r2:.3f} H={H:.4f} H_R2={H_r2:.3f}")
    return results


def main(limit_members=None, force_corpus_level=False):
    if not os.path.exists(ZIP_PATH):
        print(f"[abort] {ZIP_PATH} not found. Run download_gagle_zip.py first and let it "
              f"report [done] before running this.")
        sys.exit(1)

    _ensure_data_dir()
    done_members = load_line_set(MEMBER_CKPT_PATH)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        all_members = sorted(m for m in zf.namelist() if fnmatch.fnmatch(m, MEMBER_PATTERN))
        print(f"[info] {len(all_members)} members match {MEMBER_PATTERN} in the zip; "
              f"{len(done_members)} already checkpointed done.")

        groups = collections.defaultdict(list)
        # X-01: id -> {"row":..., "member":..., "deviant": bool}. Human rows
        # are decided by explicit preference (prefer a non-deviant source --
        # Defect C section 4), not "first member wins" -- a given article's
        # clean copy is not guaranteed to sort before its deviant copy (this
        # was verified false empirically 2026-08-22: a first-seen-wins run
        # left 677/5000 survivors sourced from summary_keywords). So every
        # candidate must be seen before any human row is finalized, which
        # means human output can only be written once the WHOLE corpus has
        # been scanned -- see the flush after the member loop below.
        human_best = {}
        stats = collections.Counter()

        # Replay already-done members to rebuild the in-memory corpus-level
        # pools (raw scores are never persisted to disk a second time --
        # see module docstring) AND human_best (needed even on resume, since
        # a better human candidate could be sitting in an earlier, already-
        # checkpointed member). No LLM output rows are (re)written for these.
        already_done_sorted = [m for m in all_members if m in done_members]
        if already_done_sorted:
            print(f"[resume] replaying {len(already_done_sorted)} already-done members "
                  f"to rebuild in-memory pools (no output rewritten, no network involved)...")
            for i, member in enumerate(already_done_sorted, 1):
                process_member(zf, member, groups=groups, human_best=human_best,
                                stats=stats, write_output=False)
                if i % 20 == 0:
                    print(f"[resume] replayed {i}/{len(already_done_sorted)}")

        remaining = [m for m in all_members if m not in done_members]
        if limit_members:
            remaining = remaining[:limit_members]
            print(f"[smoke-test] limiting to first {limit_members} not-yet-done members")

        with open(OUT_PATH, "a") as fout:
            for i, member in enumerate(remaining, 1):
                # Rows are buffered by process_member and written here in one
                # shot, after the member is fully read -- if the process
                # dies mid-member, nothing for that member has hit disk yet,
                # so redoing it on restart can't duplicate output rows. This
                # applies to LLM rows only; human rows are flushed separately
                # below, once, after every member (fresh or replayed) is in.
                rows = process_member(zf, member, groups=groups, human_best=human_best,
                                       stats=stats, write_output=True)
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                append_line(MEMBER_CKPT_PATH, member)
                done_members.add(member)
                print(f"[member complete] {i}/{len(remaining)} ({member}) "
                      f"written={stats['written']} raw_seen={stats['raw_entries_seen']}")

            complete = (len(done_members) >= len(all_members)) and not limit_members
            coverage_ok = set(done_members) == set(all_members)

            if complete and coverage_ok:
                print(f"\n[info] all members scanned -- flushing {len(human_best)} "
                      f"deduplicated human rows now that every candidate has been seen...")
                for id_, entry in sorted(human_best.items()):
                    row = entry["row"]
                    groups[(row["domain"], row["label"])].append(
                        np.asarray(row["scores"], dtype=np.float64))
                    trajectory, increment = generate_trajectory(row["scores"])
                    H, H_r2 = calc_H(increment)
                    fout.write(json.dumps({
                        "id": row["id"],
                        "split": row["split"],
                        "domain": row["domain"],
                        "Label": row["label"],
                        "prompt_type": row["prompt_type"],
                        "temp": row["temp"],
                        "Hurst_Exp": float(H),
                        "Hurst_R2": float(H_r2),
                    }) + "\n")
                    stats["written"] += 1
                fout.flush()
                os.fsync(fout.fileno())
            else:
                print(f"\n[incomplete] {len(remaining)} members left unprocessed this run -- "
                      f"human rows are NOT written yet (dedup needs the full corpus scanned "
                      f"first). Rerun this exact command to continue; human rows will be "
                      f"written on the run that completes coverage.")

        # X-01 invariant: no surviving human row should be sourced from a
        # deviant-score member (Defect C section 4: summary_keywords or
        # mistral_7b_* members carry a gemma-2b array that disagrees with
        # the majority variant). This is checked, not assumed.
        deviant_survivors = {
            id_: entry["member"] for id_, entry in human_best.items() if entry["deviant"]
        }
        if deviant_survivors:
            print(f"\n[X-01 INVARIANT VIOLATION] {len(deviant_survivors)}/"
                  f"{len(human_best)} surviving human rows are sourced from a "
                  f"deviant member (summary_keywords or mistral_7b_*). Examples: "
                  f"{list(deviant_survivors.items())[:10]}")
        else:
            print(f"\n[X-01 invariant holds] 0/{len(human_best)} surviving human "
                  f"rows sourced from summary_keywords or mistral_7b_* members.")

        manifest = {
            "members_total": len(all_members),
            "members_done": len(done_members),
            "members_missing": sorted(set(all_members) - set(done_members))[:20],
            "raw_entries_seen_this_run": stats["raw_entries_seen"],
            "docs_written_this_run": stats["written"],
            "skipped_dup_human_this_run": stats["skipped_dup_human"],
            "json_errors_this_run": stats["json_errors"],
            "human_rows_total": len(human_best),
            "x01_deviant_survivors": len(deviant_survivors),
            "x01_deviant_survivor_ids": list(deviant_survivors.keys())[:20],
            "complete": bool(complete),
            "member_coverage_gap_free": bool(coverage_ok),
        }
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        print("\n=== SUMMARY ===")
        for k, v in manifest.items():
            print(f"  {k}: {v}")

        if complete and coverage_ok:
            print("\n[complete] All members done. Computing corpus-level S/H per (domain, label)...")
            corpus_results = compute_corpus_level_SH(groups)
            with open(CORPUS_SH_PATH, "w") as f:
                json.dump(corpus_results, f, indent=2)
            print(f"[done] corpus-level S/H written to {CORPUS_SH_PATH}")
        elif force_corpus_level:
            print("\n[smoke-test] --force-corpus-level set: computing corpus-level S/H on PARTIAL "
                  "data (not all members done) purely to exercise the code path. NOT a real result.")
            corpus_results = compute_corpus_level_SH(groups)
            print(f"[smoke-test] {len(corpus_results)} (domain,label) cells computed on partial data "
                  f"-- not written to {CORPUS_SH_PATH} since the run is incomplete.")
        else:
            print("\n[incomplete] Rerun this exact command to continue -- already-done members "
                  "are skipped (after a quick in-memory replay). Corpus-level S/H is only "
                  "computed once every member is done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-members", type=int, default=None,
                    help="Smoke-test: only process the first N not-yet-done members, then stop.")
    p.add_argument("--force-corpus-level", action="store_true",
                    help="Smoke-test: compute corpus-level S/H even if not all members are done. "
                         "Use with --limit-members to exercise that code path without a full run.")
    args = p.parse_args()
    main(limit_members=args.limit_members, force_corpus_level=args.force_corpus_level)
