#!/usr/bin/env python3
"""
run_rwrf_lite.py

Command-line wrapper for RWRFLite (mp or base).

NO CONFIG IMPORTS — all inputs are provided via CLI.

Features
- Paths: --rwrf-src, --qpepre-src, --qpepre-nc-src, --npz-folder
- Date range: --start-date, --end-date (inclusive, YYYY/MM/DD)
- Hours: --hours "00,06,12,18" (required)
- Concurrency: --workers
- Overwrite: --overwrite
- Logging: --log-level, --log-file
- Writes run metadata JSON into the output folder
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from datetime import datetime
from typing import List
import multiprocessing

from rwrf_lite_mp import RWRFLite, create_datestrs  
import time

def setup_logging(level: str, log_file: str | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
        handlers=handlers,
    )
logger = logging.getLogger("rwrf")
setup_logging("DEBUG", "rwrf_qpe_nc_2_npz_log.txt")

def parse_hours(hours_str: str) -> List[str]:
    """Parse required comma-separated hours into zero-padded HH list."""
    parts = [p.strip() for p in hours_str.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("`--hours` is required and must be non-empty, e.g. 00,06,12,18")
    hrs = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid hour '{p}': must be 0..23")
        v = int(p)
        if not (0 <= v <= 23):
            raise ValueError(f"Invalid hour '{p}': must be 0..23")
        hrs.append(f"{v:02d}")
    return hrs


def ensure_dir(p: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str, log_file: str | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
        handlers=handlers,
    )


def dump_metadata(meta_path: pathlib.Path, payload: dict):
    try:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to write metadata: %s", e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RWRFLite over a configurable date/hour range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rwrf-src", required=True, help="Path to raw RWRF source directory")
    parser.add_argument("--qpepre-src", required=True, help="Path to QPEPRE source directory")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY/MM/DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY/MM/DD)")
    parser.add_argument("--hours", required=True, help="Comma-separated hours, e.g. '00,06,12,18'")
    parser.add_argument("--qpepre-nc-src", required=True, help="Temporary working directory for NetCDFs")
    parser.add_argument("--npz-folder", required=True, help="Output NPZ cache directory root")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Max workers for ProcessPoolExecutor")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args(argv)

    # Start timer
    t0 = time.perf_counter()

    # Resolve hours and dates
    try:
        hr_strs = parse_hours(args.hours)
    except ValueError as e:
        logger.error(str(e))
        return 2
    date_strs = create_datestrs([args.start_date, args.end_date])
    if not date_strs:
        logger.error("Empty date range after parsing; check --start-date/--end-date.")
        return 2

    # Prepare directories
    qpere_nc_src = ensure_dir(args.qpepre_nc_src)
    out_root = ensure_dir(args.npz_folder)
    out_dir = ensure_dir(out_root)

    # Instantiate pipeline (all config comes from CLI here)
    rwrf_lite = RWRFLite(
        qpepre_src=args.qpepre_src,
        rwrf_src=args.rwrf_src,
        qpepre_nc_src=str(args.qpepre_nc_src),
        npz_folder=str(out_dir),
        overwrite=args.overwrite,
    )


    logger.debug("Dates: %s", date_strs)
    logger.debug("Hours: %s", hr_strs)
    logger.debug("RWRF src: %s | QPEPRE src: %s", args.rwrf_src, args.qpepre_src)
    logger.debug("qpepre-nc-src: %s | npz-folder: %s", qpere_nc_src, out_dir)

    # Persist run metadata
    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rwrf_src": str(args.rwrf_src),
        "qpepre_src": str(args.qpepre_src),
        "tmp_src": str(qpere_nc_src),
        "npz_folder": str(out_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "hours": hr_strs,
        "workers": args.workers,
        "overwrite": args.overwrite,
        "cmdline": " ".join(sys.argv),
    }
    dump_metadata(out_dir / "run_metadata.json", run_meta)

    # Run
    try:
        rwrf_lite.process_files_from_date_list(date_strs, hr_strs, workers=args.workers)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1

    elapsed = time.perf_counter() - t0
    mins, secs = divmod(elapsed, 60)
    logger.info("Total runtime: %.1f seconds (%.1f minutes)", elapsed, mins + secs/60)
    logger.info("Completed successfully. Output: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())