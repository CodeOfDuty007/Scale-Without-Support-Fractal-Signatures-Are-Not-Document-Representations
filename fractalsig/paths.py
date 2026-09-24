"""Single source of truth for every filesystem location used by the code.

All locations resolve from environment variables, with defaults relative to the
repository root, so no script contains a machine-specific path.

Environment variables (all optional):

    FRACTALSIG_WORKSPACE   Root for downloaded and intermediate data.
                           Default: <repo>/workspace
    FRACTALSIG_GAGLE_ZIP   Path to the GAGLE archive (GAGLE_ZIP_PATH is also honoured).
                           Default: $FRACTALSIG_WORKSPACE/gagle.zip
    GAGLE_MANIFEST_DIR     Parquet shards written by the CPU feature pass and
                           read by the GPU pass / experiments.
                           Default: $FRACTALSIG_WORKSPACE/gpu_manifest_shards
    FRACTALSIG_OUTPUT_DIR  Root for regenerated statistics, figures and logs.
                           Default: <repo>/outputs
    FRACTALSIG_STATS_DIR   Overrides the statistics directory only (e.g. point
                           it at reference_outputs/stats to re-render figures
                           from the shipped statistics).

Importing this module has no side effects; call :func:`ensure_dirs` to create
the writable directories.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


REPO_ROOT: Path = Path(__file__).resolve().parents[1]

WORKSPACE: Path = _env_path("FRACTALSIG_WORKSPACE", REPO_ROOT / "workspace")
GAGLE_ZIP: Path = _env_path("FRACTALSIG_GAGLE_ZIP", _env_path("GAGLE_ZIP_PATH", WORKSPACE / "gagle.zip"))
DATA_DIR: Path = WORKSPACE / "data"
FEATURE_STORE_DIR: Path = WORKSPACE / "feature_store"
MANIFEST_SHARD_DIR: Path = _env_path("GAGLE_MANIFEST_DIR", WORKSPACE / "gpu_manifest_shards")
ALIGNMENT_PROBE_DIR: Path = WORKSPACE / "alignment_probe"

OUTPUT_DIR: Path = _env_path("FRACTALSIG_OUTPUT_DIR", REPO_ROOT / "outputs")
STATS_DIR: Path = _env_path("FRACTALSIG_STATS_DIR", OUTPUT_DIR / "stats")
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
LOG_DIR: Path = OUTPUT_DIR / "logs"

REFERENCE_DIR: Path = REPO_ROOT / "reference_outputs"
REFERENCE_STATS_DIR: Path = REFERENCE_DIR / "stats"
REFERENCE_FIGURES_DIR: Path = REFERENCE_DIR / "figures"

# String forms for modules that build paths with os.path.join.
WORKSPACE_STR: str = str(WORKSPACE)
GAGLE_ZIP_STR: str = str(GAGLE_ZIP)
DATA_DIR_STR: str = str(DATA_DIR)
FEATURE_STORE_DIR_STR: str = str(FEATURE_STORE_DIR)
MANIFEST_SHARD_DIR_STR: str = str(MANIFEST_SHARD_DIR)
STATS_DIR_STR: str = str(STATS_DIR)
FIGURES_DIR_STR: str = str(FIGURES_DIR)
LOG_DIR_STR: str = str(LOG_DIR)

_WRITABLE = (WORKSPACE, DATA_DIR, FEATURE_STORE_DIR, MANIFEST_SHARD_DIR,
             STATS_DIR, FIGURES_DIR, LOG_DIR)


def ensure_dirs() -> None:
    """Create every writable directory (idempotent)."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable summary of the resolved configuration."""
    rows = [
        ("REPO_ROOT", REPO_ROOT), ("WORKSPACE", WORKSPACE), ("GAGLE_ZIP", GAGLE_ZIP),
        ("DATA_DIR", DATA_DIR), ("FEATURE_STORE_DIR", FEATURE_STORE_DIR),
        ("MANIFEST_SHARD_DIR", MANIFEST_SHARD_DIR), ("STATS_DIR", STATS_DIR),
        ("FIGURES_DIR", FIGURES_DIR), ("LOG_DIR", LOG_DIR),
        ("REFERENCE_DIR", REFERENCE_DIR),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


if __name__ == "__main__":
    print(describe())
