#!/usr/bin/env python3
"""T1: run every deterministic split-half construction in one process.

The corpus is loaded once and reused across constructions
(``run_construction.run`` caches it in a module-level dict). Constructions that
already have a statistics file in the output directory are skipped unless
``--force`` is given, so an interrupted run resumes where it stopped.

Usage:
    python -m experiments.run_all_t1_deterministic [--workers N] [--force]
"""
import argparse
import os

from fractalsig import paths as _paths
from fractalsig.runtime import default_workers
from experiments.reliability_lib import WINDOW_SCALE
from experiments.run_construction import run

CONSTRUCTIONS = ["interleave", "contiguous", "block_4", "block_8", "block_16",
                 "block_32", "block_bootstrap", "matched_sweep"]
SEED = 20260908


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--force", action="store_true", help="recompute constructions that already have output")
    args = ap.parse_args()
    _paths.STATS_DIR.mkdir(parents=True, exist_ok=True)

    for c in CONSTRUCTIONS:
        prefix = f"t0_t1_{c}"
        stats_path = os.path.join(_paths.STATS_DIR_STR, f"{prefix}_stats.json")
        if os.path.exists(stats_path) and not args.force:
            print(f"[driver] skip {c}: {stats_path} exists (use --force to recompute)", flush=True)
            continue
        window_scale = WINDOW_SCALE if c == "matched_sweep" else None
        print(f"\n{'=' * 80}\n[driver] construction={c}\n{'=' * 80}", flush=True)
        run(c, args.workers, prefix, seed=SEED, window_scale=window_scale, save_long_parquet=True)
    print("\n[driver] all deterministic T1 constructions done", flush=True)


if __name__ == "__main__":
    main()
