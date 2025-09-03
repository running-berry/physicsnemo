#!/usr/bin/env python3
"""
check_rwrf_npz_missing.py

Parallel scanner to report missing (and optionally unreadable) RWRF NPZ files
for a date range and hour list.

- Filename pattern: "{variable}_{YYYYMMDD}{HH}.npz"
- Uses ProcessPoolExecutor for concurrency (configurable with --workers)
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logger = logging.getLogger("check_rwrf_npz")

# ---------- Variables list (defaults) ----------
DEFAULT_VARIABLES = [
    "u10","v10","t2m","sp","msl","tcwv",
    "u50","u100","u150","u200","u250","u300","u400","u500","u600","u700","u850","u925","u1000",
    "v50","v100","v150","v200","v250","v300","v400","v500","v600","v700","v850","v925","v1000",
    "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
    "t50","t100","t150","t200","t250","t300","t400","t500","t600","t700","t850","t925","t1000",
    "q50","q100","q150","q200","q250","q300","q400","q500","q600","q700","q850","q925","q1000",
    "qpepre","lsm","orog",
]

# ---------- CLI helpers ----------
def parse_hours(hours: str) -> List[str]:
    parts = [p.strip() for p in hours.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid hour '{p}' (must be 0..23).")
        v = int(p)
        if not (0 <= v <= 23):
            raise ValueError(f"Invalid hour '{p}' (must be 0..23).")
        out.append(f"{v:02d}")
    if not out:
        raise ValueError("No hours parsed from --hours.")
    return out

def parse_variables(vars_str: Optional[str]) -> List[str]:
    if not vars_str:
        return DEFAULT_VARIABLES
    parts = [p.strip() for p in vars_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("No variables parsed from --variables.")
    return parts

def date_range_inclusive(start_ymd_slash: str, end_ymd_slash: str) -> List[str]:
    fmt = "%Y/%m/%d"
    start = datetime.strptime(start_ymd_slash, fmt).date()
    end = datetime.strptime(end_ymd_slash, fmt).date()
    if start > end:
        start, end = end, start
    days = (end - start).days + 1
    return [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]

# ---------- Model ----------
@dataclass(frozen=True)
class ExpectedFile:
    variable: str
    yyyymmdd: str
    hh: str
    path: Path

def expected_files(root: Path, variables: Iterable[str], yyyymmdds: Iterable[str], hours: Iterable[str]) -> List[ExpectedFile]:
    return [ExpectedFile(v, d, h, root / f"{v}_{d}{h}.npz")
            for v in variables for d in yyyymmdds for h in hours]

# ---------- Worker ----------
def _probe_file(args: Tuple[str, str, str, str, bool]) -> Tuple[str, str, str, str, str]:
    """
    Worker that checks a single file.

    Returns tuple: (status, variable, yyyymmdd, hh, path)
      status in {"ok", "missing", "unreadable"}
    """
    variable, yyyymmdd, hh, path_str, strict = args
    p = Path(path_str)
    logger.info("checking %s", p)
    if not p.exists():
        return ("missing", variable, yyyymmdd, hh, path_str)
    if not strict:
        return ("ok", variable, yyyymmdd, hh, path_str)
    # Strict: try to open to ensure valid NPZ
    try:
        import numpy as np
        with np.load(p, allow_pickle=False) as _:
            pass
        return ("ok", variable, yyyymmdd, hh, path_str)
    except Exception:
        return ("unreadable", variable, yyyymmdd, hh, path_str)

# ---------- Parallel check ----------
def check_files_mp(files: List[ExpectedFile], strict: bool, workers: int) -> Tuple[List[ExpectedFile], List[ExpectedFile]]:
    logger.info("Checking %d expected files (strict=%s, workers=%d)...", len(files), strict, workers)

    tasks = [(f.variable, f.yyyymmdd, f.hh, str(f.path), strict) for f in files]
    missing: List[ExpectedFile] = []
    unreadable: List[ExpectedFile] = []

    # NOTE: per-file DEBUG logging only (INFO would be too noisy)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_probe_file, t) for t in tasks]
        for fut in as_completed(futures):
            status, v, d, h, path_str = fut.result()
            if status == "missing":
                ef = ExpectedFile(v, d, h, Path(path_str))
                missing.append(ef)
                logger.debug("Missing: %s", path_str)
            elif status == "unreadable":
                ef = ExpectedFile(v, d, h, Path(path_str))
                unreadable.append(ef)
                logger.debug("Unreadable: %s", path_str)
            else:
                logger.debug("OK: %s", path_str)

    logger.info("Check complete. Missing=%d, Unreadable=%d", len(missing), len(unreadable))
    return missing, unreadable

# ---------- Reporting ----------
def print_summary(
    root: Path,
    variables: List[str],
    yyyymmdds: List[str],
    hours: List[str],
    missing: List[ExpectedFile],
    unreadable: List[ExpectedFile],
) -> None:
    total_expected = len(variables) * len(yyyymmdds) * len(hours)
    print("=" * 72)
    print(f"Root: {root}")
    print(f"Variables: {len(variables)}")
    print(f"Dates: {yyyymmdds[0]} .. {yyyymmdds[-1]}  (count={len(yyyymmdds)})")
    print(f"Hours: {','.join(hours)}  (count={len(hours)})")
    print(f"Total expected files: {total_expected}")
    print(f"Missing: {len(missing)}")
    print(f"Unreadable (strict): {len(unreadable)}")
    print("=" * 72)

    if missing:
        by_dt: Dict[Tuple[str, str], List[str]] = {}
        for f in missing:
            by_dt.setdefault((f.yyyymmdd, f.hh), []).append(f.variable)
        print("\nTop missing slots (date/hour) with variable counts:")
        for (d, h), vars_ in sorted(by_dt.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]:
            print(f"  {d} {h}: {len(vars_)} vars missing")

def export_csv(csv_path: Path, rows: List[ExpectedFile], label: str) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "variable", "yyyymmdd", "hh", "path"])
        for r in rows:
            w.writerow([label, r.variable, r.yyyymmdd, r.hh, str(r.path)])

# ---------- Main ----------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check missing RWRF NPZ files over a date/hour range (parallel).")
    p.add_argument("--root", required=True, help="Directory containing NPZ files.")
    p.add_argument("--start-date", required=True, help="YYYY/MM/DD")
    p.add_argument("--end-date", required=True, help="YYYY/MM/DD")
    p.add_argument("--hours", required=True, help="Comma-separated hours, e.g. 00,06,12,18 or 00,01,...,23")
    p.add_argument("--variables", default=None, help="Comma-separated variable IDs. Defaults to built-in list.")
    p.add_argument("--csv-out", default=None, help="Optional CSV path to write a detailed missing list.")
    p.add_argument("--csv-unreadable-out", default=None, help="Optional CSV path for unreadable (when --strict).")
    p.add_argument("--strict", action="store_true", help="Try opening each NPZ to detect corrupt/unreadable files.")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Process pool size.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logger level.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s"
    )

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found or not a directory: {root}", file=sys.stderr)
        return 2

    try:
        hours = parse_hours(args.hours)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        variables = parse_variables(args.variables)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    yyyymmdds = date_range_inclusive(args.start_date, args.end_date)
    files = expected_files(root, variables, yyyymmdds, hours)

    missing, unreadable = check_files_mp(files, strict=args.strict, workers=args.workers)

    print_summary(root, variables, yyyymmdds, hours, missing, unreadable)

    if args.csv_out:
        export_csv(Path(args.csv_out), missing, label="missing")
        print(f"Wrote missing CSV: {args.csv_out}")
    if args.csv_unreadable_out and args.strict:
        export_csv(Path(args.csv_unreadable_out), unreadable, label="unreadable")
        print(f"Wrote unreadable CSV: {args.csv_unreadable_out}")

    # Non-zero exit if anything is missing or unreadable
    return 1 if (missing or unreadable) else 0

if __name__ == "__main__":
    raise SystemExit(main())