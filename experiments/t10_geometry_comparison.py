#!/usr/bin/env python3
"""T10: compare split-half reliability of the topological features across window geometries.

For each pilot run ``t10_pilot_topology_raw_<tag>.json`` (written by
``t10_pilot_topology``), computes per H1 feature:

* ``raw_r``      Pearson r between the two interleaved halves,
* ``partial_r``  the same correlation after partialling out the document's window
                 count (linear, with intercept), which removes the length confound,
* ``*_SB``       Spearman-Brown projection to full length, 2r / (1 + r).

Writes ``t10_pilot_geometry_comparison.json``.

Usage:
    python -m experiments.t10_geometry_comparison [TAG ...]   # default: w40_s40 w15_s15 w10_s10
"""
import json
import os
import sys

import numpy as np

from fractalsig import paths as _paths

FEATURES = ["H1_total_persistence", "H1_entropy", "H1_n_bars", "H1_max_persistence"]
DEFAULT_TAGS = ["w40_s40", "w15_s15", "w10_s10"]


def spearman_brown(r: float) -> float:
    return 2.0 * r / (1.0 + r)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of x and y after OLS-residualising both on [1, z]."""
    design = np.column_stack([np.ones_like(z), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def summarise(rows: list[dict]) -> dict:
    n_windows = np.array([r["n_windows"] for r in rows], dtype=float)
    out = {"mean_n_windows": float(n_windows.mean()), "n_docs": len(rows), "features": {}}
    for feat in FEATURES:
        a = np.array([r["half_a"][feat] for r in rows], dtype=float)
        b = np.array([r["half_b"][feat] for r in rows], dtype=float)
        raw = float(np.corrcoef(a, b)[0, 1])
        part = partial_corr(a, b, n_windows)
        out["features"][feat] = {"raw_r": raw, "raw_r_SB": spearman_brown(raw),
                                 "partial_r": part, "partial_r_SB": spearman_brown(part)}
    return out


def main() -> None:
    tags = sys.argv[1:] or DEFAULT_TAGS
    result = {}
    for tag in tags:
        path = os.path.join(_paths.STATS_DIR_STR, f"t10_pilot_topology_raw_{tag}.json")
        if not os.path.exists(path):
            raise SystemExit(f"[t10-geometry] missing {path}; run experiments.t10_pilot_topology first")
        with open(path) as fh:
            result[tag] = summarise(json.load(fh))
        print(f"[t10-geometry] {tag}: n_docs={result[tag]['n_docs']} "
              f"mean_n_windows={result[tag]['mean_n_windows']:.2f}", flush=True)
    out_path = os.path.join(_paths.STATS_DIR_STR, "t10_pilot_geometry_comparison.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[written] {out_path}", flush=True)


if __name__ == "__main__":
    main()
