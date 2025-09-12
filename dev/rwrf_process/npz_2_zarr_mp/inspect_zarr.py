#!/usr/bin/env python3
"""
zarr_inspect.py

Inspect a StormCast Zarr dataset and log per-time, per-channel stats.

- Supports Zarr written by build_stormcast_zarr_mp.py (variables "HighRes" or "LowRes")
- Computes nan-aware stats: min, max, mean, std
- Logs to file (and stdout) with progress messages
- Optional CSV export

Usage:
  python zarr_inspect.py /path/to/HighRes/exp_train.zarr --var HighRes --log zarr_stats.log --csv stats.csv
  python zarr_inspect.py /path/to/LowRes/exp_train.zarr  --var LowRes  --log zarr_stats.log
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import xarray as xr
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


LOG = logging.getLogger("zarr_inspect")


def setup_logging(log_path: Optional[str], level: str = "INFO"):
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    LOG.debug("Logger initialized (level=%s, file=%s)", level, log_path)


def auto_var_name(ds: xr.Dataset, user_var: Optional[str]) -> str:
    """Pick the data variable to inspect."""
    if user_var:
        if user_var not in ds.data_vars:
            raise ValueError(f"--var {user_var!r} not in dataset data_vars {list(ds.data_vars)}")
        return user_var
    # If only one data var, use it
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    # Prefer common names
    for name in ("HighRes", "LowRes", "HighRes_invariants"):
        if name in ds.data_vars:
            return name
    raise ValueError(f"Multiple data_vars found; please choose with --var from {list(ds.data_vars)}")


def compute_stats(arr: np.ndarray):
    """Return (nanmin, nanmax, nanmean, nanstd) for a numeric array."""
    # Flatten to avoid axis issues; keep NaN-safe ops
    return (
        float(np.nanmin(arr)),
        float(np.nanmax(arr)),
        float(np.nanmean(arr)),
        float(np.nanstd(arr)),
    )


def inspect_zarr(zarr_path: str, var_name: Optional[str], csv_out: Optional[str], only_first_n_hours: Optional[int]):
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(f"Zarr directory not found: {zarr_path}")

    LOG.info("Opening Zarr: %s", zarr_path)
    # Use consolidated metadata (your writer consolidates it)
    ds = xr.open_zarr(zarr_path, consolidated=True)

    # Pick variable
    vname = auto_var_name(ds, var_name)
    da = ds[vname]
    LOG.info("Using variable: %s", vname)

    # Basic coords sanity
    time = da.coords.get("time", None)
    channel = da.coords.get("channel", None)
    y = da.coords.get("y", None)
    x = da.coords.get("x", None)

    LOG.info("Dims: %s | shape=%s", da.dims, tuple(da.shape))
    if channel is not None:
        LOG.info("Channels (%d): %s", channel.size, list(map(str, channel.values)))
    if time is not None:
        LOG.info("Time range: %s → %s (len=%d)", str(time.values[0]), str(time.values[-1]), time.size)
    if y is not None and x is not None:
        LOG.info("Spatial grid: y=%d, x=%d", y.size, x.size)

    # Determine how many hours to scan
    T = int(da.sizes.get("time", 1))
    C = int(da.sizes.get("channel", 1))
    scan_T = min(T, only_first_n_hours) if (only_first_n_hours and T) else T
    LOG.info("Scanning %d hour(s) across %d channel(s)", scan_T, C)

    # Optional CSV rows
    rows = []

    # Iterate hour by hour to keep memory bounded
    for ti in range(scan_T):
        # Select one time slice: shape (channel, y, x)
        if "time" in da.dims:
            slice_t = da.isel(time=ti)
            t_val = str(slice_t.coords["time"].values)
        else:
            slice_t = da
            t_val = "NA"

        # Compute per-channel stats
        for ci in range(C):
            if "channel" in slice_t.dims:
                slice_tc = slice_t.isel(channel=ci).to_numpy()  # (y, x)
                c_name = str(slice_t.coords["channel"].values[ci])
            else:
                # Invariant or single-channel dataset
                slice_tc = slice_t.to_numpy()
                c_name = "0"

            try:
                vmin, vmax, vmean, vstd = compute_stats(slice_tc)
                LOG.info(
                    "[time=%s | ch=%s] min=%.6g max=%.6g mean=%.6g std=%.6g",
                    t_val, c_name, vmin, vmax, vmean, vstd
                )
                if csv_out:
                    rows.append({
                        "time": t_val,
                        "channel": c_name,
                        "min": vmin,
                        "max": vmax,
                        "mean": vmean,
                        "std": vstd,
                    })
            except ValueError:
                # All-NaN slice
                LOG.info("[time=%s | ch=%s] ALL NaNs", t_val, c_name)
                if csv_out:
                    rows.append({
                        "time": t_val,
                        "channel": c_name,
                        "min": np.nan,
                        "max": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                    })

        if (ti + 1) % 24 == 0 or (ti + 1) == scan_T:
            LOG.info("Progress: %d/%d hour(s) processed", ti + 1, scan_T)

    # Optional CSV
    if csv_out:
        if not HAS_PANDAS:
            LOG.warning("pandas is not installed; cannot write CSV. Install pandas or omit --csv")
        else:
            Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            pd.DataFrame(rows).to_csv(csv_out, index=False)
            LOG.info("Wrote CSV summary to %s", csv_out)

    # Global stats (all time, per channel) — optional extra
    if "time" in da.dims and "channel" in da.dims:
        LOG.info("Computing global per-channel stats across ALL %d hour(s) ...", T)
        # Reduce along time, y, x; keep channel
        # We’ll loop channels to avoid loading everything into RAM
        for ci in range(C):
            c_name = str(da.coords["channel"].values[ci])
            sl = da.isel(channel=ci)
            arr = sl.to_numpy()  # shape: (time, y, x) or (y, x) if no time
            try:
                vmin, vmax, vmean, vstd = compute_stats(arr)
                LOG.info("[GLOBAL | ch=%s] min=%.6g max=%.6g mean=%.6g std=%.6g",
                         c_name, vmin, vmax, vmean, vstd)
            except ValueError:
                LOG.info("[GLOBAL | ch=%s] ALL NaNs", c_name)

    LOG.info("Done.")


def main():
    ap = argparse.ArgumentParser(description="Inspect a StormCast Zarr file and log per-time, per-channel stats.")
    ap.add_argument("zarr_path", help="Path to the Zarr store directory (e.g., .../HighRes/exp_train.zarr)")
    ap.add_argument("--var", default=None, help="Data variable to inspect (e.g., HighRes or LowRes). If omitted, auto-select.")
    ap.add_argument("--log", default="zarr_inspect.log", help="Log file path (default: zarr_inspect.log)")
    ap.add_argument("--log-level", default="DEBUG", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--csv", default=None, help="Optional CSV path to save per-time, per-channel stats")
    ap.add_argument("--first-n-hours", type=int, default=None, help="Limit to first N hours to scan (for quick checks)")
    args = ap.parse_args()

    setup_logging(args.log, args.log_level)

    try:
        inspect_zarr(args.zarr_path, args.var, args.csv, args.first_n_hours)
    except Exception as e:
        LOG.exception("Inspection failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()