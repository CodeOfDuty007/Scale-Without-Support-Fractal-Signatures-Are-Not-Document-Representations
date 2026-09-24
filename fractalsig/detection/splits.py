#!/usr/bin/env python3
"""
Frozen split definitions (X-04, the overdue D3 gate).

THE GROUPING UNIT IS (domain, dataset_id) -- NOT the row, and NOT the id.
  Every GAGLE record derives from one source article. That article appears
  once as a Human row and ~44 more times as LLM continuations at various
  (generator, temp, prompt_type). Splitting by row would put an article's
  human original in train and its own machine continuation in test, which is
  leakage of exactly the signal the detector is supposed to learn.

  X-01/D-07 already established that `split` must NOT be part of an article's
  identity (4,474/4,474 cross-split copies are byte-identical), so GAGLE's own
  train/test/val labels are ignored here and the split is rebuilt from scratch.

  dataset_id is NOT a column -- it is embedded in the ids, in two different
  formats, and joining them is the load-bearing step:
      Human : big_patent_00000-of-00256__0_human   -> 00000-of-00256__0
      LLM   : big-p-gemma-2b-extend-0.0-00004-of-00256--35 -> 00004-of-00256__35
  Note the separator differs ('__' vs '--'). Verified 2026-08-23 across the
  whole post-X-01 store: 5,000 human groups, 5,000 LLM groups, 5,000 shared,
  **0 human-only, 0 LLM-only, 0 unparsed ids**, 100% aligned in all 5 domains.
  If that assertion ever fails, the split is silently wrong -- so it is checked
  at runtime, not assumed.

HOW THE JOIN WAS ACTUALLY VERIFIED (read this before trusting it)
  An earlier version of this docstring cited "5,000 human groups, 5,000 LLM
  groups, 5,000 shared, 0 orphans" as proof the join is correct. **That proof
  is void.** Set cardinality is invariant under ANY bijection: a within-shard
  permutation of the LLM index space would produce identical counts while
  putting ~30% of articles' human rows in train and their own continuations in
  test. The `no_human`/`no_llm` assertion in build() has the same blind spot.
  Counting cannot establish that group G denotes the same ARTICLE on both
  sides; only content can.

  Independently verified 2026-08-23 by correlating each human row's features
  against the mean of its ~44 LLM continuations, against a WITHIN-SHARD
  permutation null (a global null is too weak -- shards are topically
  clustered, so it would pass on topic alone):

      Sent_Len_Mean          r=+0.1870   null -0.0078 +/- 0.0057   z=+34.4
      TTR                    r=+0.6361   null +0.4908 +/- 0.0062   z=+23.4
      Bigram_Repetition_Rate r=+0.5861   null +0.4100 +/- 0.0073   z=+24.0

  Sent_Len_Mean is decisive because its null is ~0, so it carries no
  shard-topic confound. Injecting a synthetic mis-join gives r=0.1699 at 10%,
  0.1394 at 25%, -0.0089 at 100%; the observed value sits at the 0% point, so
  a mis-join fraction of even 10% is excluded. THE '__N' AND '--N' INDEX
  SPACES ARE THE SAME COORDINATE.

THREE SCHEMES
  random : group-disjoint train/val/test. Assignment is a hash of the group
           key -- NOT stratified by domain. An earlier version of this file
           carried a per-domain loop that looked like stratification; it was
           dead code (removing it changed 0/5000 assignments) and it has been
           removed rather than left to imply a guarantee that does not exist.
           Per-domain balance is therefore an OUTCOME, not a constraint: it
           lands within +/-1.5 sigma (worst: scientific_papers val 11.4%,
           wikipedia val 8.9%) and cannot be tightened without changing the
           scheme. summarize() reports it as an observation.
  LODO   : leave-one-domain-out. Tests cross-domain generalization, which is
           Part II's headline negative result (deep fractal residuals HURT it).
  LOGO   : leave-one-generator-out. Tests whether the detector transfers to an
           unseen generator.

  LODO and LOGO are ALSO group-disjoint. For LOGO this is not automatic and is
  the easy thing to get wrong: an article's Human row is shared by every
  generator, so holding out generator B while leaving that article's Human row
  in train leaks the article. Groups are therefore partitioned FIRST, and the
  generator holdout is applied within that partition.

FROZEN
  Deterministic given SPLIT_SEED. Writing the manifest is the freeze; from then
  on downstream code should read the manifest, not recompute. Re-running with
  the same seed and inputs reproduces it exactly (asserted by --verify).
"""
from fractalsig import paths as _paths  # noqa: E402
import argparse
import collections
import hashlib
import json
import os
import re
import sys

SPLIT_SEED = 20260823
DEFAULT_STORE = os.path.join(_paths.WORKSPACE_STR,
                             "data", "p1_feature_store_cpu.jsonl")
DEFAULT_MANIFEST = os.path.join(_paths.WORKSPACE_STR,
                                "data", "splits_v1.json")

TEST_FRAC, VAL_FRAC = 0.20, 0.10
HUMAN_LABEL = "Human"

# Trailing '<NNNNN>-of-<NNNNN>--<n>' on LLM ids; '__' variant on human ids.
# Left-anchored with (?<!\d) so a 6-digit shard field cannot silently alias onto
# a 5-digit one ('...-100004-of-00256--35' must NOT become '00004-of-00256__35').
# All 218,834 live ids use exactly 5 digits, so this is latent-only today.
_LLM_DS = re.compile(r'(?<!\d)(\d{5}-of-\d{5})--(\d+)$')

# LLM ids carry a domain abbreviation. group_key() deliberately takes `domain`
# from the COLUMN, not from this map -- but the two must agree, and if they ever
# disagree the failure is SILENT and dangerous: 131 dataset_id strings occur in
# two domains (78 big_patent/wikipedia, 53 newsroom/scientific_papers), so a
# mislabeled LLM row lands inside a DIFFERENT article's perfectly valid group
# instead of becoming an orphan the alignment check would catch. Asserted below.
_LLM_DOMAIN_ABBREV = {
    "big-p": "big_patent", "bills": "billsum", "newsr": "newsroom",
    "scien": "scientific_papers", "wikip": "wikipedia",
}


def group_key(row):
    """(domain, dataset_id) for any row, human or LLM. None if unparseable."""
    domain, rid = row["domain"], row["id"]
    if row["Label"] == HUMAN_LABEL:
        pre, suf = domain + "_", "_human"
        if rid.startswith(pre) and rid.endswith(suf):
            return (domain, rid[len(pre):-len(suf)])
        return None
    m = _LLM_DS.search(rid)
    return (domain, f"{m.group(1)}__{m.group(2)}") if m else None


def _bucket(group, seed):
    """Deterministic uniform [0,1) from the group key -- independent of row
    order, dict iteration order, and numpy version. A hash-based assignment is
    reproducible anywhere; np.random.shuffle over a set is not."""
    h = hashlib.sha256(f"{seed}|{group[0]}|{group[1]}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


REQUIRED = ("id", "domain", "Label")


def load_rows(path):
    """Reads only the three fields the split depends on.

    NOTE what this implies for --verify: no feature column is read, so the
    frozen split stays valid across a feature recompute (F-4 rewrote every S
    value without changing an id) -- but equally, --verify can NOT detect that
    the features changed. It validates the partition, not the data.
    """
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            missing = [k for k in REQUIRED if k not in d]
            if missing:
                # Previously these keys were filtered with `if k in d`, so a row
                # missing Label died later with a bare KeyError inside
                # group_key() instead of saying what was wrong.
                raise SystemExit(f"[FATAL] {path}:{ln} lacks {missing}; "
                                 f"the split cannot be built from this store.")
            rows.append({k: d[k] for k in REQUIRED})
    return rows


def build(rows, seed=SPLIT_SEED):
    groups = {}
    unparsed = []
    for r in rows:
        g = group_key(r)
        if g is None:
            unparsed.append(r["id"])
            continue
        groups.setdefault(g, {"human": 0, "llm": collections.Counter()})
        if r["Label"] == HUMAN_LABEL:
            groups[g]["human"] += 1
        else:
            groups[g]["llm"][r["Label"]] += 1
    if unparsed:
        raise SystemExit(f"[FATAL] {len(unparsed)} ids did not yield a group key, "
                         f"e.g. {unparsed[:3]}. The id format changed -- fix "
                         f"group_key() rather than dropping rows.")

    # Catch the SILENT failure mode: a mislabeled `domain` on an LLM row cannot
    # produce an orphan (131 dataset_ids exist in two domains), so it would land
    # inside another article's valid group and nothing else here would notice.
    bad_dom = []
    for r in rows:
        if r["Label"] == HUMAN_LABEL:
            continue
        abbrev = r["id"].split("-", 1)[0]
        expect = _LLM_DOMAIN_ABBREV.get(abbrev)
        if expect is not None and expect != r["domain"]:
            bad_dom.append((r["id"], r["domain"], expect))
    if bad_dom:
        raise SystemExit(
            f"[FATAL] {len(bad_dom)} LLM rows whose id abbreviation contradicts "
            f"their domain column, e.g. {bad_dom[:3]}. These would silently join "
            f"into the WRONG article's group -- 131 dataset_ids are shared across "
            f"domains. Fix the store, do not proceed.")

    # The alignment invariant this whole scheme rests on.
    no_human = [g for g, v in groups.items() if v["human"] == 0]
    no_llm = [g for g, v in groups.items() if not v["llm"]]
    if no_human or no_llm:
        raise SystemExit(f"[FATAL] group alignment broken: {len(no_human)} groups "
                         f"with no human row, {len(no_llm)} with no LLM rows. "
                         f"Group-disjoint splitting would not prevent leakage.")

    # --- random, group-disjoint --------------------------------------------
    # Purely hash-based: assignment depends only on the group key, so it is
    # independent of row order, dict order and numpy version. There is NO
    # per-domain stratification (see the docstring); a loop that appeared to
    # provide it was dead code and was removed.
    assign = {}
    for g in sorted(groups):
        u = _bucket(g, seed)
        assign[g] = ("test" if u < TEST_FRAC
                     else "val" if u < TEST_FRAC + VAL_FRAC
                     else "train")

    generators = sorted({lab for v in groups.values() for lab in v["llm"]})
    domains = sorted({g[0] for g in groups})

    return {
        "seed": seed,
        "n_groups": len(groups),
        "domains": domains,
        "generators": generators,
        "test_frac": TEST_FRAC,
        "val_frac": VAL_FRAC,
        "random": {f"{g[0]}|{g[1]}": assign[g] for g in sorted(groups)},
        "lodo": {
            "_recipe": "For held-out domain D: TEST = every group whose domain "
                       "is D. TRAIN/VAL = groups from the OTHER domains only.",
            "_validation_warning":
                "Do NOT reuse the 'random' val assignment for LODO early "
                "stopping. Those val groups are drawn from all domains, so "
                "~1/5 of them belong to D itself and would leak the held-out "
                "domain into model selection. Use `lodo_val` below, which is "
                "the random-split val groups RESTRICTED to non-D domains.",
            "folds": {d: {"test_domains": [d],
                          "train_domains": [x for x in domains if x != d],
                          "n_test_groups": sum(1 for g in groups if g[0] == d),
                          "n_val_groups_leak_free": sum(
                              1 for g in groups
                              if g[0] != d and assign[g] == "val")}
                      for d in domains},
            "lodo_val": {d: sorted(f"{g[0]}|{g[1]}" for g in groups
                                   if g[0] != d and assign[g] == "val")
                         for d in domains},
        },
        "logo": {
            "_recipe": "For held-out generator G: TEST = G's rows in TEST "
                       "groups + those groups' human rows. TRAIN = the other "
                       "generators' rows in TRAIN groups + those groups' human "
                       "rows. Group partition FIRST, generator holdout second.",
            "_leak_warning":
                "Do NOT read this as 'hold out all of G's rows globally'. An "
                "article's human row is shared by every generator, so a global "
                "generator holdout leaves the held-out article's own human "
                "original in train. The group partition is what prevents it.",
            "_val_note": "The 509 'val' groups are unused by this recipe. Use "
                         "them as the LOGO validation set (same generator "
                         "restriction as TRAIN) rather than discarding them.",
            "_caveats": [
                "Test-fold class balance differs from the random split: ~4.26% "
                "human (gemma fold) and ~4.48% (mistral fold) vs 2.23%. "
                "Thresholds and calibration do NOT transfer between schemes.",
                "The two folds are not interchangeable: prompt_type mix differs "
                "(cot is 9.94% of gemma rows but 4.61% of mistral's, because "
                "mistral cot generation dropped ~68% of articles). A "
                "cross-generator transfer claim from these folds is confounded "
                "by prompt_type composition and must say so.",
            ],
            "folds": {gen: {"test_generator": gen,
                            "train_generators": [x for x in generators if x != gen]}
                      for gen in generators},
        },
        "_group_stats": {
            "rows_total": len(rows),
            "human_rows": sum(v["human"] for v in groups.values()),
            "llm_rows": sum(sum(v["llm"].values()) for v in groups.values()),
        },
    }


def summarize(man):
    a = man["random"]
    c = collections.Counter(a.values())
    print(f"  groups: {man['n_groups']:,}   domains: {len(man['domains'])}   "
          f"generators: {', '.join(man['generators'])}")
    print(f"  random split (groups): " +
          "  ".join(f"{k}={v:,} ({100*v/len(a):.1f}%)" for k, v in sorted(c.items())))
    per = collections.defaultdict(collections.Counter)
    for k, v in a.items():
        per[k.split("|", 1)[0]][v] += 1
    print("  per-domain group counts (OBSERVED, not constrained -- assignment\n        is hash-based; see docstring. Reported so drift is visible):")
    for d in sorted(per):
        t = sum(per[d].values())
        print(f"    {d:<20} " + "  ".join(
            f"{s}={per[d][s]:>4} ({100*per[d][s]/t:>4.1f}%)"
            for s in ("train", "val", "test")))
    print(f"  LODO folds: {len(man['domains'])}   LOGO folds: {len(man['generators'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--out", default=DEFAULT_MANIFEST)
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and assert byte-identical to the frozen manifest")
    args = ap.parse_args()

    rows = load_rows(args.store)
    man = build(rows)
    # Fingerprint of the INPUT identity space (not the features -- see
    # load_rows). Recorded IN the manifest so --verify has something external to
    # check against; previously the freeze sha256 was only printed to stdout, so
    # a rebuild compared the file to itself and would pass if both had drifted.
    man["_input_fingerprint"] = hashlib.sha256(
        "\n".join(sorted(f'{r["id"]}|{r["domain"]}|{r["Label"]}' for r in rows)
                  ).encode()).hexdigest()

    print("=" * 74)
    print("SPLITS v1 (X-04 / D3 gate)")
    print("=" * 74)
    summarize(man)

    blob = json.dumps(man, indent=2, sort_keys=True)
    if args.verify:
        if not os.path.exists(args.out):
            print(f"[FAIL] no frozen manifest at {args.out}")
            return 1
        frozen = json.load(open(args.out))
        ok = True

        same = hashlib.sha256(open(args.out).read().encode()).hexdigest() == \
            hashlib.sha256(blob.encode()).hexdigest()
        print(f"\n  [{'PASS' if same else 'FAIL'}] rebuild byte-identical to frozen manifest")
        ok &= same

        fp_ok = frozen.get("_input_fingerprint") == man["_input_fingerprint"]
        print(f"  [{'PASS' if fp_ok else 'FAIL'}] store identity space matches the "
              f"fingerprint recorded at freeze time")
        ok &= fp_ok

        # Assert the PARTITION properties a reader assumes 'verify' covers.
        a = frozen["random"]
        mapped = 0
        straddle = collections.defaultdict(set)
        for r in rows:
            g = group_key(r)
            k = f"{g[0]}|{g[1]}"
            if k not in a:
                print(f"  [FAIL] row {r['id']} maps to unassigned group {k}")
                ok = False
                break
            mapped += 1
            straddle[k].add(a[k])
        print(f"  [{'PASS' if mapped == len(rows) else 'FAIL'}] every store row "
              f"({mapped:,}/{len(rows):,}) maps to an assigned group")
        ok &= mapped == len(rows)

        multi = [k for k, v in straddle.items() if len(v) > 1]
        print(f"  [{'PASS' if not multi else 'FAIL'}] no group spans >1 split "
              f"({len(multi)} offenders)")
        ok &= not multi

        for d, keys in frozen.get("lodo", {}).get("lodo_val", {}).items():
            bad = [k for k in keys if k.split("|", 1)[0] == d]
            if bad:
                print(f"  [FAIL] lodo_val[{d}] contains {len(bad)} groups from the "
                      f"held-out domain itself")
                ok = False
                break
        else:
            print(f"  [PASS] every lodo_val fold excludes its own held-out domain")

        print(f"\n[{'PASS' if ok else 'FAIL'}] splits verification")
        return 0 if ok else 1

    if os.path.exists(args.out):
        print(f"\n[abort] {args.out} exists. The split is FROZEN -- "
              f"move it aside deliberately (never delete, D-10) to re-freeze.")
        return 1
    with open(args.out, "w") as fh:
        fh.write(blob)
    print(f"\n[frozen] {args.out}")
    print(f"[sha256] {hashlib.sha256(blob.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
