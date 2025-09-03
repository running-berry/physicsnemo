#!/usr/bin/env python3
#
# build_stormcast_zarr_mp.py
#
# Build HighRes/LowRes Zarr datasets (and invariants) from NPZ caches using fully
# configurable CLI args. Supports two splits (train/valid) with one or multiple
# date ranges each, and uses multiprocessing to speed up per-timestamp loading.
#
# NEW: When a file is missing at a target timestamp, we now substitute the
#      NEAREST-IN-TIME available file for that *same variable* (±1h, ±2h, ...).
#      If no file exists for that variable anywhere in the split, we fall back
#      to zeros with the correct shape (from --var-dummy template).
#
# Example:
#   python build_stormcast_zarr_mp.py \
#     --cache-highres ../data/cache/rwrf/train \
#     --cache-lowres  ../data/cache/era5/train \
#     --data-base ../data \
#     --experiment-name stormcast_small \
#     --train-ranges "2019/08/01:2019/08/31" \
#     --valid-ranges "2019/09/01:2019/09/07" \
#     --lon-min 118 --lon-max 123.5 --lat-min 21.5 --lat-max 26.8 \
#     --domain-size 256,256 \
#     --split both --workers 16 --log-level INFO
#
# Notes:
# - If you omit --var-lowres / --var-highres / --invariants, built-in defaults are used.
# - Invariants are taken from the HighRes cache (first timestamp in the split where all exist).
# - Missing NPZs are replaced by the nearest-in-time available sample for that var; only if no
#   samples exist at all do we fall back to zeros (dummy template).

import argparse
import os
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import xarray as xr
import zarr

# You must provide these utilities in PYTHONPATH:
#   util_extract.extract_region(npz_path, var, lon_min, lon_max, lat_min, lat_max, domain_size)
#   util_extract.interp_to_domain(lon, lat, data, domain_size, method="linear")
import util_extract as u1

LOG = logging.getLogger("npz2zarr")

# -------------------- Logging -------------------- #
def setup_logging(level: str, log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
        handlers=handlers,
    )
    LOG.debug("Logger initialized (level=%s, file=%s)", level, log_file)

setup_logging("DEBUG", "/workspace/phy")
# -------------------- Defaults -------------------- #
DEFAULT_LOWRES_VARS = [
    # Provided low-res (ERA5-like) channel list
    "mslp", "sp", "t2m", "u10", "v10", "tp", "tcwv",
    "q1000", "q850", "q500", "q250",
    "t1000", "t850", "t500", "t250",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
]

DEFAULT_HIGHRES_VARS = [
    # Provided high-res (RWRF) channel list
    "u10","v10","t2m","sp","msl","tcwv",
    "u50","u100","u150","u200","u250","u300","u400","u500","u600","u700","u850","u925","u1000",
    "v50","v100","v150","v200","v250","v300","v400","v500","v600","v700","v850","v925","v1000",
    "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
    "t50","t100","t150","t200","t250","t300","t400","t500","t600","t700","t850","t925","t1000",
    "q50","q100","q150","q200","q250","q300","q400","q500","q600","q700","q850","q925","q1000",
    "qpepre","lsm","orog",
]

DEFAULT_INVARIANTS = ["lsm", "orog"]
DEFAULT_DUMMY_VAR = "t2m"

# -------------------- Logging -------------------- #
def setup_logging(level: str, log_file: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
        handlers=handlers,
    )
    LOG.debug("Logger initialized (level=%s, file=%s)", level, log_file)

# -------------------- CLI helpers -------------------- #
def parse_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    vals = [p.strip() for p in s.split(",") if p.strip()]
    LOG.debug("Parsed CSV '%s' -> %s", s, vals)
    return vals

def parse_domain_size(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--domain-size must be like H,W (e.g., 256,256)")
    H = int(parts[0]); W = int(parts[1])
    if H <= 0 or W <= 0:
        raise ValueError("domain-size must be positive integers")
    LOG.debug("Parsed domain size '%s' -> (%d, %d)", s, H, W)
    return (H, W)

def make_hourly_datetimes(start_slash: str, end_slash: str) -> np.ndarray:
    base = np.datetime64(start_slash.replace("/", "-") + "T00:00:00")
    end  = np.datetime64(end_slash.replace("/", "-")) + np.timedelta64(23, "h")
    total_hours = int((end - base) / np.timedelta64(1, "h")) + 1
    arr = base + np.arange(total_hours, dtype=np.int64) * np.timedelta64(1, "h")
    LOG.debug("Hourly datetimes %s -> %s (count=%d)", start_slash, end_slash, len(arr))
    return arr

def parse_ranges(ranges: str) -> List[Tuple[str, str]]:
    """
    ranges string like: "YYYY/MM/DD:YYYY/MM/DD,YYYY/MM/DD:YYYY/MM/DD"
    returns list of (start, end)
    """
    out: List[Tuple[str, str]] = []
    for chunk in ranges.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad range '{chunk}', expected A:B")
        a, b = [q.strip() for q in chunk.split(":", 1)]
        out.append((a, b))
    if not out:
        raise ValueError("No date ranges parsed.")
    LOG.debug("Parsed ranges '%s' -> %s", ranges, out)
    return out

def concat_hourly(ranges: List[Tuple[str, str]]) -> np.ndarray:
    blocks = [make_hourly_datetimes(a, b) for a, b in ranges]
    out = np.concatenate(blocks) if blocks else np.array([], dtype="datetime64[h]")
    LOG.info("Constructed datetime grid from %d range(s): total %d hours", len(ranges), len(out))
    return out

# -------------------- Data helpers -------------------- #
def npz_path_for(cache_dir: str, var: str, dt: np.datetime64) -> str:
    yy, mm, dd, hh = np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
    return os.path.join(cache_dir, f"{var}_{yy}{mm}{dd}{hh}.npz")

def try_build_dummy(
    datetime_array: np.ndarray,
    cache_dir: str,
    dummy_var: str,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the first timestamp where dummy_var NPZ exists and use it to infer shape.
    Returns: (dummy_data, lon_grid, lat_grid, times)
    """
    LOG.info("Searching for dummy template var='%s' in %d timestamps under %s",
             dummy_var, len(datetime_array), cache_dir)
    for dt in list(datetime_array):
        candidate = npz_path_for(cache_dir, dummy_var, dt)
        if os.path.exists(candidate):
            LOG.info("Using %s as dummy template", candidate)
            real_arr, lon_grid, lat_grid, times = u1.extract_region(
                candidate, dummy_var, lon_min, lon_max, lat_min, lat_max,
                domain_size=domain_size,
            )
            Ny_full, Nx_full = lat_grid.shape
            Ny_tgt, Nx_tgt = domain_size
            if Ny_full < Ny_tgt or Nx_full < Nx_tgt:
                LOG.debug("Template smaller than target; interpolating to %s", domain_size)
                real_arr, lon_grid, lat_grid = u1.interp_to_domain(
                    lon_grid, lat_grid, real_arr, domain_size, method="linear"
                )
            dummy_arr = np.full_like(real_arr, fill_value=0.0)
            LOG.debug("Dummy template shape=%s", tuple(dummy_arr.shape))
            return dummy_arr, lon_grid, lat_grid, times
    raise FileNotFoundError(
        "Could not build dummy: no NPZ found for --var-dummy across the provided datetimes."
    )

# --------- Nearest-available substitution helpers ---------- #
def build_availability_map(cache_dir: str, var_list: List[str], dts: np.ndarray) -> Dict[str, np.ndarray]:
    """
    For each variable, return a sorted array of indices (into dts) where a file exists.
    """
    avail: Dict[str, List[int]] = {v: [] for v in var_list}
    for i, dt in enumerate(dts):
        for v in var_list:
            if os.path.exists(npz_path_for(cache_dir, v, dt)):
                avail[v].append(i)
    return {v: np.array(ixs, dtype=int) for v, ixs in avail.items()}

def nearest_index(target_idx: int, sorted_indices: np.ndarray) -> Optional[int]:
    """
    Given a sorted array of available indices, return the index (into that array)
    of the nearest available element to target_idx, breaking ties by choosing the earlier one.
    Returns None if sorted_indices is empty.
    """
    if sorted_indices.size == 0:
        return None
    pos = np.searchsorted(sorted_indices, target_idx)
    if pos == 0:
        return sorted_indices[0]
    if pos == sorted_indices.size:
        return sorted_indices[-1]
    before = sorted_indices[pos - 1]
    after  = sorted_indices[pos]
    # tie-break: prefer 'before' if equal distance
    return before if (target_idx - before) <= (after - target_idx) else after

def load_or_nearest(
    cache_dir: str,
    var: str,
    dt_idx: int,
    all_datetimes: np.ndarray,
    avail_indices_for_var: np.ndarray,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
    dummy_data: np.ndarray,
) -> Tuple[np.ndarray, bool, Optional[int]]:
    """
    Try to load var at dt_idx; if missing, substitute the nearest-in-time available
    timestamp for that var. Returns (data, was_substituted, src_idx_or_None).
    If no sample exists for this var anywhere, returns (dummy, True, None).
    """
    # 1) direct hit?
    dt = all_datetimes[dt_idx]
    path = npz_path_for(cache_dir, var, dt)
    if os.path.exists(path):
        data, lon_grid, lat_grid, _ = u1.extract_region(
            path, var, lon_min, lon_max, lat_min, lat_max, domain_size=domain_size
        )
        data, lon_grid, lat_grid = u1.interp_to_domain(lon_grid, lat_grid, data, domain_size, method="linear")
        return data, False, dt_idx

    # 2) nearest substitution
    near_idx = nearest_index(dt_idx, avail_indices_for_var)
    if near_idx is None:
        # no sample for this var at all -> zeros fallback
        return dummy_data.copy(), True, None

    dt2 = all_datetimes[near_idx]
    path2 = npz_path_for(cache_dir, var, dt2)
    data, lon_grid, lat_grid, _ = u1.extract_region(
        path2, var, lon_min, lon_max, lat_min, lat_max, domain_size=domain_size
    )
    data, lon_grid, lat_grid = u1.interp_to_domain(lon_grid, lat_grid, data, domain_size, method="linear")
    return data, True, near_idx

# -------------------- Multiprocessing worker -------------------- #
def _load_timestamp_block(
    idx: int,
    all_datetimes: np.ndarray,
    cache_dir: str,
    var_list: List[str],
    avail_map: Dict[str, np.ndarray],   # var -> sorted indices where file exists
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
    dummy_data: np.ndarray,
) -> tuple:
    """
    Worker: Load all variables for one timestamp, using nearest-in-time substitution
    if missing. Returns (index, stacked_array, substitutions_log).
    """
    channel_arr = None
    subs_msgs: List[str] = []

    for var in var_list:
        data, substituted, src_idx = load_or_nearest(
            cache_dir, var, idx, all_datetimes, avail_map[var],
            lon_min, lon_max, lat_min, lat_max, domain_size, dummy_data
        )
        if substituted:
            tgt_str = np.datetime_as_string(all_datetimes[idx], unit="h")
            if src_idx is None:
                subs_msgs.append(f"{var} {tgt_str} <- ZERO (no samples for var)")
            else:
                src_str = np.datetime_as_string(all_datetimes[src_idx], unit="h")
                subs_msgs.append(f"{var} {tgt_str} <- nearest {src_str}")

        channel_arr = data.copy() if channel_arr is None else np.concatenate((channel_arr, data), axis=0)

    if channel_arr.ndim != 3:
        raise RuntimeError(f"Unexpected per-timestamp shape: {channel_arr.shape}")
    return idx, channel_arr, subs_msgs

# -------------------- Split builder (MP) -------------------- #
def build_split(
    split_name: str,
    datetime_array: np.ndarray,
    cache_highres: str,
    cache_lowres: str,
    vars_highres: List[str],
    vars_lowres: List[str],
    dummy_var: str,
    invariants: List[str],
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
    zarr_base: str,
    experiment_name: str,
    workers: int,
) -> None:
    """
    Assemble HighRes/LowRes 4D arrays over a datetime grid, compute stats, and
    write Zarr stores. Also write invariants (from the first available timestamp
    of the split) once under zarr_base/invariants/.
    """
    t_split0 = time.perf_counter()
    LOG.info("[%s] Building split with %d timestamps | HR vars=%d | LR vars=%d | workers=%d",
             split_name, len(datetime_array), len(vars_highres), len(vars_lowres), workers)

    # ---- Dummy template from HighRes cache ----
    dummy_data, dummy_lon_grid, dummy_lat_grid, _ = try_build_dummy(
        datetime_array, cache_highres, dummy_var, lon_min, lon_max, lat_min, lat_max, domain_size
    )

    def build_level(fname: str, cache_dir: str, var_list: List[str]):
        t0 = time.perf_counter()
        LOG.info("[%s] %s: precomputing availability map ...", split_name, fname)
        avail_map = build_availability_map(cache_dir, var_list, datetime_array)
        missing_all = sum(1 for v in var_list if avail_map[v].size == 0)
        if missing_all:
            LOG.warning("[%s] %s: %d/%d vars have NO samples; will fall back to zeros.",
                        split_name, fname, missing_all, len(var_list))

        LOG.info("[%s] %s: assembling with multiprocessing (vars=%d, workers=%d)",
                 split_name, fname, len(var_list), workers)

        stats_dir = Path(zarr_base) / fname / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        T = len(datetime_array)
        C = len(var_list)
        H, W = domain_size

        # Preallocate final array
        data_arr = np.empty((T, C, H, W), dtype=np.float32)
        subs_log: List[str] = []

        # Fan out
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _load_timestamp_block,
                    idx, datetime_array, cache_dir, var_list, avail_map,
                    lon_min, lon_max, lat_min, lat_max, domain_size, dummy_data
                )
                for idx in range(T)
            ]

            for i, fut in enumerate(as_completed(futures), 1):
                idx, block, subs = fut.result()
                data_arr[idx] = block
                if subs:
                    subs_log.extend(subs)
                if i % 500 == 0 or i == T:
                    LOG.info("[%s] %s: collected %d/%d", split_name, fname, i, T)

        LOG.info("[%s] %s: stacked array shape=%s | substitutions applied=%d",
                 split_name, fname, tuple(data_arr.shape), len(subs_log))

        # Stats
        LOG.info("[%s] %s: computing stats ...", split_name, fname)
        means = np.nanmean(data_arr, axis=(0, 2, 3)).astype(np.float32)
        stds  = np.nanstd (data_arr, axis=(0, 2, 3)).astype(np.float32)
        np.save(stats_dir / f"{experiment_name}_{split_name}_means.npy", means)
        np.save(stats_dir / f"{experiment_name}_{split_name}_stds.npy",  stds)
        LOG.info("[%s] %s: saved stats under %s", split_name, fname, stats_dir)

        # Zarr write
        ds = xr.Dataset(
            { f"{fname}": (["time", "channel", "y", "x"], data_arr) },
            coords={
                "time": datetime_array,
                "channel": var_list,
                "latitude": (["y", "x"], dummy_lat_grid),
                "longitude": (["y", "x"], dummy_lon_grid),
            }
        )
        enc = {f"{fname}": {"dtype": "float32", "compressor": None}}
        out_store = Path(zarr_base) / fname / f"{experiment_name}_{split_name}.zarr"
        out_store.parent.mkdir(parents=True, exist_ok=True)

        LOG.info("[%s] %s: writing Zarr -> %s", split_name, fname, out_store)
        t_z0 = time.perf_counter()
        ds.to_zarr(str(out_store), mode="w", consolidated=True, encoding=enc, zarr_format=2)
        zarr.consolidate_metadata(str(out_store))
        t_z1 = time.perf_counter()
        LOG.info("[%s] %s: Zarr write complete in %.1fs", split_name, fname, t_z1 - t_z0)

        t1 = time.perf_counter()
        LOG.info("[%s] %s: total elapsed %.1fs", split_name, fname, t1 - t0)

    # Build HighRes / LowRes
    build_level("HighRes", cache_highres, vars_highres)
    build_level("LowRes",  cache_lowres,  vars_lowres)

    # Invariants
    LOG.info("[%s] Building invariants ...", split_name)
    inv_arr = None
    dt0 = None
    for dt in list(datetime_array):
        ok_all = True
        for var in invariants:
            if not os.path.exists(npz_path_for(cache_highres, var, dt)):
                ok_all = False
                break
        if ok_all:
            dt0 = dt
            break
    if dt0 is None:
        raise FileNotFoundError("No timestamp where all invariants exist in HighRes cache.")
    LOG.info("[%s] Using %s for invariants extraction", split_name, np.datetime_as_string(dt0, unit="h"))

    for var in invariants:
        path = npz_path_for(cache_highres, var, dt0)
        dt_data, lon_grid, lat_grid, _ = u1.extract_region(
            path, var, lon_min, lon_max, lat_min, lat_max, domain_size=domain_size
        )
        dt_data, lon_grid, lat_grid = u1.interp_to_domain(
            lon_grid, lat_grid, dt_data, domain_size, method="linear"
        )
        inv_arr = dt_data.copy() if inv_arr is None else np.concatenate((inv_arr, dt_data), axis=0)

    inv_ds = xr.Dataset(
        { "HighRes_invariants": (["channel", "y", "x"], inv_arr) },
        coords={
            "channel": invariants,
            "latitude": (["y", "x"], lat_grid),
            "longitude": (["y", "x"], lon_grid),
        }
    )
    inv_enc = {"HighRes_invariants": {"dtype": "float32", "compressor": None}}
    inv_store = Path(zarr_base) / "invariants" / "invariants.zarr"
    inv_store.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("[%s] Writing invariants Zarr -> %s", split_name, inv_store)
    t_iz0 = time.perf_counter()
    inv_ds.to_zarr(str(inv_store), mode="w", consolidated=True, encoding=inv_enc, zarr_format=2)
    zarr.consolidate_metadata(str(inv_store))
    t_iz1 = time.perf_counter()
    LOG.info("[%s] Invariants write complete in %.1fs | shape=%s",
             split_name, t_iz1 - t_iz0, tuple(inv_arr.shape))

    t_split1 = time.perf_counter()
    LOG.info("[%s] Split complete in %.1fs", split_name, t_split1 - t_split0)

# -------------------- Main -------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Build HighRes/LowRes Zarr datasets for train/valid splits from NPZ caches (multiprocessing, nearest-time substitution)."
    )
    # Variables (optional; fall back to defaults if omitted)
    ap.add_argument("--var-lowres",  default=None,
                    help="Comma-separated vars for LowRes. If omitted, built-in defaults are used.")
    ap.add_argument("--var-highres", default=None,
                    help="Comma-separated vars for HighRes. If omitted, built-in defaults are used.")
    ap.add_argument("--var-dummy",   default=DEFAULT_DUMMY_VAR,
                    help=f"Var used to infer dummy shape (default: {DEFAULT_DUMMY_VAR}).")
    ap.add_argument("--invariants",  default=None,
                    help=f"Comma-separated invariant vars (default: {','.join(DEFAULT_INVARIANTS)}).")

    # Region / domain
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--domain-size", required=True, help="H,W (e.g., 256,256)")

    # Paths
    ap.add_argument("--cache-highres", required=True, help="Folder containing HighRes NPZ files (e.g., rwrf cache).")
    ap.add_argument("--cache-lowres",  required=True, help="Folder containing LowRes NPZ files (e.g., era5 cache).")
    ap.add_argument("--zarr-base",     required=True, help="Output base directory for Zarr stores and stats.")
    ap.add_argument("--experiment-name", required=True, help="Base name used in output Zarrs (e.g., 'stormcast_small').")

    # Splits & ranges
    ap.add_argument("--train-ranges", required=True,
                    help="Comma-separated date ranges A:B (YYYY/MM/DD:YYYY/MM/DD).")
    ap.add_argument("--valid-ranges", required=True,
                    help="Comma-separated date ranges A:B (YYYY/MM/DD:YYYY/MM/DD) for validation.")
    ap.add_argument("--split", choices=["train", "valid", "both"], default="both",
                    help="Which split(s) to build.")

    # MP + Logging
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of worker processes for multiprocessing.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--log-file", default=None, help="Optional path to write logs.")

    args = ap.parse_args()
    setup_logging(args.log_level, args.log_file)

    # Resolve variable lists (CLI overrides defaults if provided)
    vars_lr  = parse_csv(args.var_lowres)  or DEFAULT_LOWRES_VARS
    vars_hr  = parse_csv(args.var_highres) or DEFAULT_HIGHRES_VARS
    inv_vars = parse_csv(args.invariants)  or DEFAULT_INVARIANTS
    dummy_var = args.var_dummy
    domain_size = parse_domain_size(args.domain_size)

    LOG.info("Starting build experiment=%s split=%s workers=%d", args.experiment_name, args.split, args.workers)
    LOG.info("HighRes vars=%d | LowRes vars=%d | Invariants=%s | Dummy=%s",
             len(vars_hr), len(vars_lr), ",".join(inv_vars), dummy_var)
    LOG.info("HighRes cache: %s | LowRes cache: %s", args.cache_highres, args.cache_lowres)

    train_dt = concat_hourly(parse_ranges(args.train_ranges))
    valid_dt = concat_hourly(parse_ranges(args.valid_ranges))

    # Run selected splits
    if args.split in ("train", "both"):
        LOG.info("=== Building TRAIN split ===")
        build_split(
            split_name="train",
            datetime_array=train_dt,
            cache_highres=args.cache_highres,
            cache_lowres=args.cache_lowres,
            vars_highres=vars_hr,
            vars_lowres=vars_lr,
            dummy_var=dummy_var,
            invariants=inv_vars,
            lon_min=args.lon_min, lon_max=args.lon_max,
            lat_min=args.lat_min, lat_max=args.lat_max,
            domain_size=domain_size,
            zarr_base=args.zarr_base,
            experiment_name=args.experiment_name,
            workers=args.workers,
        )

    if args.split in ("valid", "both"):
        LOG.info("=== Building VALID split ===")
        build_split(
            split_name="valid",
            datetime_array=valid_dt,
            cache_highres=args.cache_highres,
            cache_lowres=args.cache_lowres,
            vars_highres=vars_hr,
            vars_lowres=vars_lr,
            dummy_var=dummy_var,
            invariants=inv_vars,  # invariants are saved in a common store, reused
            lon_min=args.lon_min, lon_max=args.lon_max,
            lat_min=args.lat_min, lat_max=args.lat_max,
            domain_size=domain_size,
            zarr_base=args.zarr_base,
            experiment_name=args.experiment_name,
            workers=args.workers,
        )

    LOG.info("All done.")

if __name__ == "__main__":
    main()