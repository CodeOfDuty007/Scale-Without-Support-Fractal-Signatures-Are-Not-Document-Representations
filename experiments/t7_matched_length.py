#!/usr/bin/env python3
"""
T7 -- matched-length analysis, built on the already-computed T0 interleave
per-document long-format data (outputs/stats/t0_t1_interleave_features.parquet),
which already carries n_tokens per (document, scorer). No rerun of the
feature computation needed.

Bands: 320-360 / 361-400 / 401-464 usable tokens (the plan's own bands).
Reports reliability (rho_pooled, rho_within) per band per headline feature.

KEY FINDING, reported rather than assumed: GAGLE's human-side documents are
gated to EXACTLY 400 usable tokens with zero variance (confirmed directly:
std=0 across all 40,665+ human (doc,scorer,feature) rows), while LLM
documents range 320-463. This means the 320-360 and 401-464 bands are
LLM-ONLY by construction -- there is no possible human/machine length-
matched comparison outside the 361-400 band, which is itself dominated by
the fixed human value. This is a stronger and more specific version of the
plan's own warning ("length correlates with label... match, don't
stratify") than a generic confound -- it is a hard structural fact about
how build_p0_local.py's CLIP_TO=464 gate interacts with human vs. LLM
document lengths, not a soft correlation to adjust for.

Given this structural fact, a length-MATCHED classification model (human
vs. machine, restricted to the same token-count range) is not meaningfully
achievable within GAGLE's own gated score arrays -- the human side simply
does not vary in usable length. This is reported as the finding and
limitation, rather than forcing a match that the data cannot support.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.reliability_lib import pooled_and_within_stats  # noqa: E402

from fractalsig import paths as _paths  # noqa: E402

DATA_DIR = _paths.STATS_DIR_STR
BANDS = [(320, 360), (361, 400), (401, 464)]
HEADLINE = ["gemma_S", "gemma_H", "mistral_S", "mistral_H"]


def main():
    table = pq.read_table(os.path.join(DATA_DIR, "t0_t1_interleave_features.parquet"))
    df = table.to_pandas()

    # ---- structural finding: token-count distribution by Label ----
    by_label = df.groupby("Label")["n_tokens"].agg(["count", "mean", "std", "min", "max"])
    print("[t7] n_tokens by Label (across all doc,scorer,feature long-format rows):", flush=True)
    print(by_label, flush=True)

    out = {"bands": BANDS, "n_tokens_by_label": json.loads(by_label.to_json(orient="index")),
           "structural_finding": (
               "Human documents are gated to EXACTLY 400 usable tokens (std=0); "
               "LLM documents range 320-463. Bands 320-360 and 401-464 are LLM-only "
               "by construction; only 361-400 contains both classes, dominated by the "
               "fixed human value of 400."
           ), "by_band": {}}

    for lo, hi in BANDS:
        band_label = f"{lo}-{hi}"
        band_df = df[(df["n_tokens"] >= lo) & (df["n_tokens"] <= hi)]
        band_out = {"n_rows_all_features": int(len(band_df))}
        label_counts = band_df.drop_duplicates(["id", "scorer"])["Label"].value_counts().to_dict()
        band_out["label_counts_unique_doc_scorer"] = {str(k): int(v) for k, v in label_counts.items()}
        for feat_full in HEADLINE:
            sname, feat = feat_full.split("_", 1)
            sub = band_df[(band_df["scorer"] == sname) & (band_df["feature"] == feat)]
            if len(sub) < 100:
                band_out[feat_full] = {"n": len(sub), "note": "insufficient rows in this band"}
                continue
            strata = list(zip(sub["Label"], sub["domain"], sub["temp"]))
            s = pooled_and_within_stats(sub["half_a"].to_numpy(), sub["half_b"].to_numpy(), strata)
            band_out[feat_full] = s
            print(f"[t7] band={band_label} {feat_full}: n={s['n']:,} rho_pooled={s['rho_pooled']:.4f} "
                  f"rho_within={s['rho_within']:.4f}", flush=True)
        out["by_band"][band_label] = band_out

    out_path = os.path.join(DATA_DIR, "t7_matched_length.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\n[written] {out_path}", flush=True)


if __name__ == "__main__":
    main()
