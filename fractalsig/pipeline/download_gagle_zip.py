#!/usr/bin/env python3
"""Resumable download of the GAGLE archive (``ibomohsin/gagle``, gagle.zip, 10.6 GB).

The file is streamed to ``<GAGLE_ZIP>.part`` and resumed with an HTTP Range
request from the partial file's on-disk size, so an interrupted download can be
restarted at any time without losing progress. A read timeout turns silent
stalls into retryable exceptions. After transfer the file is size-checked against
the Hub metadata and CRC-checked (zipfile.testzip); a corrupt file is deleted.

Destination: ``$FRACTALSIG_GAGLE_ZIP`` (default ``workspace/gagle.zip``).
Authentication: uses the token from ``hf auth login`` if present.

Usage:
    python -m fractalsig.pipeline.download_gagle_zip
"""
from fractalsig import paths as _paths  # noqa: E402
import os
import sys
import time
import zipfile

import requests
from huggingface_hub import HfApi, get_token

REPO_ID = "ibomohsin/gagle"
FILENAME = "gagle.zip"
DEST_DIR = str(_paths.GAGLE_ZIP.parent)
DEST_PATH = _paths.GAGLE_ZIP_STR
PART_PATH = DEST_PATH + ".part"

URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FILENAME}"

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 60          # no bytes received for this long on an open request -> retry
MAX_RETRIES = 500
BACKOFF_BASE = 3
BACKOFF_CAP = 60
CHUNK_SIZE = 1024 * 1024
PROGRESS_EVERY = 200 * 1024 * 1024


def expected_size():
    info = HfApi().repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    for s in info.siblings:
        if s.rfilename == FILENAME:
            return s.size
    raise RuntimeError(f"{FILENAME} not found in repo listing")


def auth_headers():
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def current_part_size():
    return os.path.getsize(PART_PATH) if os.path.exists(PART_PATH) else 0


def download_once(exp_size, headers):
    """One attempt: resume from current_part_size() to exp_size, or raise."""
    local_size = current_part_size()
    if local_size >= exp_size:
        return
    req_headers = dict(headers)
    req_headers["Range"] = f"bytes={local_size}-"
    mode = "ab" if local_size > 0 else "wb"

    with requests.get(URL, headers=req_headers, stream=True,
                       timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"unexpected status {resp.status_code}: {resp.text[:200]}")
        if local_size > 0 and resp.status_code == 200:
            # We asked for bytes=<local_size>- but got a full 200 instead of a
            # 206 partial response: the server ignored our Range header and is
            # about to send the WHOLE file from byte zero. Appending that to
            # an existing partial would silently corrupt it -- refuse instead.
            raise RuntimeError(
                f"server ignored Range request (status 200, expected 206) with "
                f"{local_size} bytes already on disk -- refusing to append and corrupt {PART_PATH}"
            )
        last_progress_print = local_size
        with open(PART_PATH, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                local_size += len(chunk)
                if local_size - last_progress_print >= PROGRESS_EVERY:
                    f.flush()
                    os.fsync(f.fileno())
                    print(f"[progress] {local_size}/{exp_size} bytes "
                          f"({100 * local_size / exp_size:.1f}%)")
                    last_progress_print = local_size
            f.flush()
            os.fsync(f.fileno())


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    exp_size = expected_size()
    print(f"[info] expected size: {exp_size} bytes")
    headers = auth_headers()
    if not headers:
        print("[warn] no HF token found (run `hf auth login`) -- proceeding unauthenticated, "
              "subject to lower rate limits.")

    if os.path.exists(DEST_PATH) and os.path.getsize(DEST_PATH) == exp_size:
        print(f"[skip] {DEST_PATH} already exists at the expected size; verifying only.")
    else:
        attempt = 0
        while current_part_size() < exp_size:
            attempt += 1
            if attempt > MAX_RETRIES:
                print("[abort] exceeded max retries")
                return 1
            try:
                print(f"[attempt {attempt}] resuming from {current_part_size()} bytes")
                download_once(exp_size, headers)
            except Exception as e:
                wait = min(BACKOFF_BASE * attempt, BACKOFF_CAP)
                print(f"[retry] attempt={attempt} {type(e).__name__}: {e} -- "
                      f"retrying in {wait}s (resume point: {current_part_size()} bytes, safe)")
                time.sleep(wait)

        os.rename(PART_PATH, DEST_PATH)
        print(f"[transferred] {DEST_PATH}")

    # Verify before trusting it.
    actual_size = os.path.getsize(DEST_PATH)
    if actual_size != exp_size:
        print(f"[FAIL] size mismatch: expected {exp_size}, got {actual_size}. Deleting and re-download needed.")
        os.remove(DEST_PATH)
        return 1

    print("[verify] size OK, running zipfile.testzip() (checks CRCs of every member, takes a while)...")
    with zipfile.ZipFile(DEST_PATH) as zf:
        bad_file = zf.testzip()
    if bad_file is not None:
        print(f"[FAIL] corrupt member in zip: {bad_file}. Deleting and re-download needed.")
        os.remove(DEST_PATH)
        return 1

    print(f"[done] downloaded and verified: {DEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
