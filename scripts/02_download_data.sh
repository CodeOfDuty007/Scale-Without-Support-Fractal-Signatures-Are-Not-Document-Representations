#!/usr/bin/env bash
# Step 2 -- download the GAGLE archive (10.6 GB; resumable, re-run after any interruption).
# Requires a Hugging Face account that has accepted the dataset terms: `hf auth login`.
# RAID (used from step 10) is obtained separately; see docs/DATA.md.
source "$(dirname "$0")/common.sh"
"$PY" -c "from fractalsig import paths; paths.ensure_dirs()"
run 02_download fractalsig.pipeline.download_gagle_zip
