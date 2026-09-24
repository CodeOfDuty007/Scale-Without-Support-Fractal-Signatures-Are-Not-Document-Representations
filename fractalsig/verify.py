"""Compare regenerated statistics against the reference outputs shipped with the repo.

Every numeric leaf in each JSON file is compared with ``math.isclose`` under a
relative and an absolute tolerance; strings, booleans and structure are checked
for equality. Keys that record wall-clock time are ignored.

Both estimators are deterministic (calc_S enumerates start offsets exhaustively
at document length and uses a seeded sampler otherwise), so the default
tolerance only absorbs floating-point summation-order differences across
platforms. ``PROFILES`` allows a per-file statistical tolerance for any output
that is Monte-Carlo by design; it is empty for the shipped reference outputs.

Usage:
    python -m fractalsig.verify                       # all files present in outputs/stats
    python -m fractalsig.verify t2_step1_noise_floor  # selected files (stem or name)
    python -m fractalsig.verify --rtol 1e-6 --atol 1e-9 --strict

Exit status: 0 if every compared file matches, 1 on any mismatch, 2 if nothing
was compared.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fractalsig import paths

IGNORED_KEYS = frozenset({"runtime_sec", "elapsed_sec", "wall_clock_sec", "timestamp"})

# file name -> list of (leaf-key regex, rule). Rules:
#   ("z", sd_key, n_key, z_max): |new - ref| <= z_max * ref[sd_key] / sqrt(n)
#       where sd_key is a sibling key and n_key a top-level key of the file.
#   ("atol", value): |new - ref| <= value
PROFILES: dict[str, list[tuple[str, tuple]]] = {}


@dataclass
class FileReport:
    name: str
    n_numeric: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _walk(ref: Any, new: Any, path: str, key: str = "",
          parent: dict | None = None) -> Iterator[tuple[str, str, dict | None, Any, Any]]:
    """Yield (path, leaf_key, ref_parent, ref_leaf, new_leaf); structural differences are leaves."""
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in sorted(set(ref) | set(new)):
            if k in IGNORED_KEYS:
                continue
            sub = f"{path}.{k}" if path else str(k)
            if k not in ref or k not in new:
                yield sub, str(k), ref, ref.get(k, "<missing>"), new.get(k, "<missing>")
            else:
                yield from _walk(ref[k], new[k], sub, str(k), ref)
    elif isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            yield f"{path}[len]", key, parent, len(ref), len(new)
            return
        for i, (r, n) in enumerate(zip(ref, new)):
            yield from _walk(r, n, f"{path}[{i}]", key, parent)
    else:
        yield path, key, parent, ref, new


def _rule_for(profile: list[tuple[str, tuple]], leaf_key: str) -> tuple | None:
    for pattern, rule in profile:
        if re.search(pattern, leaf_key):
            return rule
    return None


def _within(rule: tuple | None, r: float, n: float, parent: dict | None, root: dict,
            rtol: float, atol: float) -> bool:
    if rule is None:
        return math.isclose(r, n, rel_tol=rtol, abs_tol=atol)
    if rule[0] == "atol":
        return abs(r - n) <= rule[1]
    if rule[0] == "z":
        _, sd_key, n_key, z_max = rule
        try:
            se = float(parent[sd_key]) / math.sqrt(float(root[n_key]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return math.isclose(r, n, rel_tol=rtol, abs_tol=atol)
        return abs(r - n) <= z_max * se
    raise ValueError(f"unknown tolerance rule {rule!r}")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_file(ref_path: Path, new_path: Path, rtol: float, atol: float, strict: bool) -> FileReport:
    report = FileReport(ref_path.name)
    try:
        ref = json.loads(ref_path.read_text())
        new = json.loads(new_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report.mismatches.append(f"unreadable: {exc}")
        return report
    profile = PROFILES.get(ref_path.name, [])
    root = ref if isinstance(ref, dict) else {}
    for leaf_path, leaf_key, parent, r, n in _walk(ref, new, ""):
        if _is_number(r) and _is_number(n):
            report.n_numeric += 1
            r_f, n_f = float(r), float(n)
            if math.isnan(r_f) and math.isnan(n_f):
                continue
            if not _within(_rule_for(profile, leaf_key), r_f, n_f, parent, root, rtol, atol):
                report.mismatches.append(f"{leaf_path}: reference={r!r} regenerated={n!r}")
        elif r != n and (strict or not (isinstance(r, str) and isinstance(n, str))):
            report.mismatches.append(f"{leaf_path}: reference={r!r} regenerated={n!r}")
    return report


def _resolve(names: list[str], new_dir: Path) -> list[str]:
    if not names:
        return sorted(p.name for p in new_dir.glob("*.json"))
    return [n if n.endswith(".json") else f"{n}.json" for n in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="JSON stems/names to compare (default: all regenerated files)")
    ap.add_argument("--reference-dir", type=Path, default=paths.REFERENCE_STATS_DIR)
    ap.add_argument("--new-dir", type=Path, default=paths.STATS_DIR)
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance (default 1e-6)")
    ap.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance (default 1e-9)")
    ap.add_argument("--strict", action="store_true", help="also require free-text string fields to match")
    ap.add_argument("--max-show", type=int, default=10, help="mismatches printed per file")
    args = ap.parse_args(argv)

    if not args.new_dir.is_dir():
        print(f"[verify] regenerated statistics directory not found: {args.new_dir}")
        return 2
    compared, failed, skipped = 0, 0, []
    for name in _resolve(args.files, args.new_dir):
        ref_path, new_path = args.reference_dir / name, args.new_dir / name
        if not ref_path.exists() or not new_path.exists():
            skipped.append(name)
            continue
        rep = compare_file(ref_path, new_path, args.rtol, args.atol, args.strict)
        compared += 1
        status = "PASS" if rep.ok else "FAIL"
        print(f"[{status}] {name}  ({rep.n_numeric} numeric values, {len(rep.mismatches)} mismatches)")
        for line in rep.mismatches[: args.max_show]:
            print(f"        {line}")
        if len(rep.mismatches) > args.max_show:
            print(f"        ... {len(rep.mismatches) - args.max_show} more")
        failed += not rep.ok
    for name in skipped:
        print(f"[SKIP] {name}  (no counterpart in reference or regenerated set)")
    if compared == 0:
        print("[verify] nothing compared")
        return 2
    print(f"[verify] {compared - failed}/{compared} files match "
          f"(rtol={args.rtol:g}, atol={args.atol:g})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
