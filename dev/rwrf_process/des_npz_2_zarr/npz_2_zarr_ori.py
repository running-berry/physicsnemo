#!/usr/bin/env python3
"""
Build HighRes/LowRes Zarr datasets (and invariants) from NPZ caches,
**preserving the original orientation** of your NPZ data.

Features:
- Fast spatial windowing + optional resizing via util_extract.extract_region()
- Choose resampling: --resample downsample|interp (default: downsample)
- Optional float32 casting to reduce memory (default: on)
- Nearest-in-time substitution when a timestamp is missing (per variable)
- Streaming per-channel mean/std stats
- NO orientation changes: data and grids are written exactly as returned
  by util_extract (except for size-only interpolation when needed)

You must provide util_extract on PYTHONPATH:
  - util_extract.extract_region(npz_path, var, lon_min, lon_max, lat_min, lat_max,
                                domain_size=(H,W), resample="downsample"|"interp",
                                cast_float32=True)
      -> returns (data, lon_grid, lat_grid, times)
  - util_extract.interp_to_domain(lon_grid, lat_grid, data, domain_size, method="linear")
      -> returns (resized_data, resized_lon_grid, resized_lat_grid)
"""

import argparse
import os
import re
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import xarray as xr
import zarr

import util_extract as u1

LOG = logging.getLogger("npz2zarr")

# -------------------- Defaults -------------------- #
DEFAULT_LOWRES_VARS = [
    "mslp", "sp", "t2m", "u10", "v10", "tcwv",
    "q1000", "q850", "q500", "q250",
    "t1000", "t850", "t500", "t250",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
]

DEFAULT_HIGHRES_VARS = [
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
        raise ValueError("--domain-size must be like H,W (e.g., 224,128)")
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

# --------- Dummy/template helpers ---------- #
def try_build_dummy(
    datetime_array: np.ndarray,
    cache_dir: str,
    dummy_var: str,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
    resample: str,
    cast_float32: bool,
    cache_index: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a zero-filled template with the correct (1,H,W) shape and grids,
    using the same extraction/resampling path as real data.
    """
    LOG.info("Searching for dummy template var='%s' under %s", dummy_var, cache_dir)

    # Prefer an exact grid timestamp
    for dt in list(datetime_array):
        candidate = npz_path_for(cache_dir, dummy_var, dt)
        if os.path.exists(candidate):
            LOG.info("Using %s as dummy template", candidate)
            real_arr, lon_grid, lat_grid, times = u1.extract_region(
                candidate, dummy_var, lon_min, lon_max, lat_min, lat_max,
                domain_size=domain_size, resample=resample, cast_float32=cast_float32
            )
            # normalize to (1,H,W)
            real_arr = _ensure_3d(real_arr)
            return np.zeros_like(real_arr), lon_grid, lat_grid, times

    # Else: any file for dummy_var from the prebuilt index
    if cache_index is None:
        cache_index = scan_cache_index(cache_dir)
    avail_any = cache_index.get(dummy_var, np.array([], dtype="datetime64[h]"))
    if avail_any.size:
        dt = avail_any[0]
        candidate = npz_path_for(cache_dir, dummy_var, dt)
        LOG.info("Using %s as dummy template (first available in cache index)", candidate)
        real_arr, lon_grid, lat_grid, times = u1.extract_region(
            candidate, dummy_var, lon_min, lon_max, lat_min, lat_max,
            domain_size=domain_size, resample=resample, cast_float32=cast_float32
        )
        real_arr = _ensure_3d(real_arr)
        return np.zeros_like(real_arr), lon_grid, lat_grid, times

    raise FileNotFoundError("Could not build dummy: no NPZ found for --var-dummy.")
    
# -------------------- Filenames & data helpers -------------------- #
def npz_path_for(cache_dir: str, var: str, dt: np.datetime64) -> str:
    yy, mm, dd, hh = np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
    return os.path.join(cache_dir, f"{var}_{yy}{mm}{dd}{hh}.npz")

_RX_VAR_TS = re.compile(r"^(?P<var>.+)_(?P<ts>\d{10})\.npz$")  # var_YYYYMMDDHH.npz

def scan_cache_index(cache_dir: str) -> Dict[str, np.ndarray]:
    """
    Scan cache once and build: { var_name -> sorted np.array(datetime64[h]) }.
    """
    index: Dict[str, List[np.datetime64]] = {}
    try:
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            m = _RX_VAR_TS.match(entry.name)
            if not m:
                continue
            var = m.group("var")
            stamp = m.group("ts")  # YYYYMMDDHH
            dt = np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:00").astype("datetime64[h]")
            index.setdefault(var, []).append(dt)
    except FileNotFoundError:
        pass

    out: Dict[str, np.ndarray] = {}
    for var, lst in index.items():
        arr = np.sort(np.unique(np.array(lst, dtype="datetime64[h]")))
        out[var] = arr
    LOG.info("Cache scan complete: %d vars indexed in '%s'", len(out), cache_dir)
    return out

# --------- Basic shape helper (no orientation change) ---------- #
def _ensure_3d(data: np.ndarray) -> np.ndarray:
    """
    Normalize data shapes to (channels, H, W).
    Accepts (H,W), (1,H,W), (C,H,W), or (T,1,H,W)->(T,H,W).
    """
    if data.ndim == 2:
        return data[None, :, :]
    if data.ndim == 3:
        return data
    if data.ndim == 4 and data.shape[1] == 1:
        # squeeze singleton middle dim
        return data[:, 0, :, :]
    raise ValueError(f"Unexpected data shape (cannot normalize to 3D): {data.shape}")

# --------- Nearest-available substitution helpers ---------- #
def nearest_dt(target_dt: np.datetime64, sorted_dts: np.ndarray) -> Optional[np.datetime64]:
    if sorted_dts.size == 0:
        return None
    diffs = np.abs(sorted_dts.astype("datetime64[h]") - target_dt.astype("datetime64[h]"))
    imin = int(np.argmin(diffs))
    best = sorted_dts[imin]
    ties = np.where(diffs == diffs[imin])[0]
    if ties.size > 1:
        best = sorted_dts[int(np.min(ties))]
    return best

def load_or_nearest_by_cache(
    cache_dir: str,
    var: str,
    target_dt: np.datetime64,
    avail_dts_for_var: np.ndarray,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
    resample: str,
    cast_float32: bool,
    dummy_data: np.ndarray,
) -> Tuple[np.ndarray, bool, Optional[np.datetime64]]:
    """
    Try loading var at target_dt; if missing, substitute nearest-in-time.
    Always returns 3D array (C,H,W) with H,W == domain_size by resizing if needed.
    Orientation is preserved: we never flip or reorder.
    """
    Ht, Wt = domain_size

    def _extract_exact(path: str) -> np.ndarray:
        arr, lon_g, lat_g, _ = u1.extract_region(
            path, var, lon_min, lon_max, lat_min, lat_max,
            domain_size=domain_size, resample=resample, cast_float32=cast_float32
        )
        arr3 = _ensure_3d(arr)
        # Enforce requested size if needed
        if arr3.shape[-2] != Ht or arr3.shape[-1] != Wt:
            arr3, _, _ = u1.interp_to_domain(lon_g, lat_g, arr3, domain_size, method="linear")
        return arr3  # preserve original orientation

    # Exact timestamp available?
    path = npz_path_for(cache_dir, var, target_dt)
    if os.path.exists(path):
        return _extract_exact(path), False, target_dt

    # Nearest-in-time fallback
    near_dt = nearest_dt(target_dt, avail_dts_for_var)
    if near_dt is None:
        # Fallback to zeros (dummy) with correct size only; orientation is irrelevant for zeros
        dum = _ensure_3d(dummy_data)
        if dum.shape[-2:] != (Ht, Wt):
            lon_lin = np.linspace(lon_min, lon_max, dum.shape[-1])
            lat_lin = np.linspace(lat_min, lat_max, dum.shape[-2])
            lon_g = np.broadcast_to(lon_lin, (dum.shape[-2], dum.shape[-1]))
            lat_g = np.broadcast_to(lat_lin[:, None], (dum.shape[-2], dum.shape[-1]))
            dum, _, _ = u1.interp_to_domain(lon_g, lat_g, dum, domain_size, method="linear")
        return dum.copy(), True, None

    path2 = npz_path_for(cache_dir, var, near_dt)
    return _extract_exact(path2), True, near_dt

# -------------------- Multiprocessing worker -------------------- #
def _load_timestamp_block(
    idx: int,
    all_datetimes: np.ndarray,
    cache_dir: str,
    var_list: List[str],
    avail_map: Dict[str, np.ndarray],
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    domain_size: Tuple[int, int],
    resample: str,
    cast_float32: bool,
    dummy_data: np.ndarray,
) -> tuple:
    """
    Worker: Load all variables for one timestamp using nearest-in-time substitution.
    Returns (index, stacked_channels(C,H,W), substitutions_log).
    """
    channel_arr = None
    subs_msgs: List[str] = []

    tgt_dt = all_datetimes[idx]
    for var in var_list:
        LOG.info(f"loading timestamp block for {var} at {tgt_dt}")
        data3, substituted, src_dt = load_or_nearest_by_cache(
            cache_dir, var, tgt_dt, avail_map.get(var, np.array([], dtype="datetime64[h]")),
            lon_min, lon_max, lat_min, lat_max, domain_size, resample, cast_float32, dummy_data
        )

        if substituted:
            tgt_str = np.datetime_as_string(tgt_dt, unit="h")
            if src_dt is None:
                subs_msgs.append(f"{var} {tgt_str} <- ZERO (no samples for var)")
            else:
                src_str = np.datetime_as_string(src_dt, unit="h")
                subs_msgs.append(f"{var} {tgt_str} <- nearest {src_str}")

        data3 = _ensure_3d(data3)
        channel_arr = data3.copy() if channel_arr is None else np.concatenate((channel_arr, data3), axis=0)

    if channel_arr is None or channel_arr.ndim != 3:
        raise RuntimeError(f"Unexpected per-timestamp shape: {None if channel_arr is None else channel_arr.shape}")
    return idx, channel_arr, subs_msgs

# -------------------- Split builder -------------------- #
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
    resample: str,
    cast_float32: bool,
    zarr_base: str,
    experiment_name: str,
    workers: int,
) -> None:
    """
    Assemble HighRes/LowRes 4D arrays over a datetime grid, compute stats, and write Zarr stores.
    Orientation is preserved throughout.
    """
    t_split0 = time.perf_counter()
    LOG.info("[%s] Building split with %d timestamps | HR vars=%d | LR vars=%d | workers=%d",
             split_name, len(datetime_array), len(vars_highres), len(vars_lowres), workers)

    # ---- Pre-scan cache once per level ----
    LOG.info("[%s] Scanning HighRes cache index ...", split_name)
    hr_index = scan_cache_index(cache_highres)
    LOG.info("[%s] Scanning LowRes cache index ...", split_name)
    lr_index = scan_cache_index(cache_lowres)

    # ---- Dummy template from HighRes cache ----
    dummy_data, dummy_lon_grid, dummy_lat_grid, _ = try_build_dummy(
        datetime_array, cache_highres, dummy_var,
        lon_min, lon_max, lat_min, lat_max, domain_size,
        resample=resample, cast_float32=cast_float32, cache_index=hr_index
    )

    # --- Ensure dummy grids are exactly (H, W) (no flips, no reorientation) ---
    H, W = domain_size
    if dummy_lat_grid.shape != (H, W) or dummy_lon_grid.shape != (H, W):
        LOG.warning("[%s] Dummy grids shape %s (lat) / %s (lon) != (%d,%d); regenerating simple rectilinear coords.",
                    split_name, dummy_lat_grid.shape, dummy_lon_grid.shape, H, W)
        lon_lin = np.linspace(lon_min, lon_max, W, dtype=np.float32 if cast_float32 else np.float64)
        lat_lin = np.linspace(lat_min, lat_max, H, dtype=np.float32 if cast_float32 else np.float64)
        dummy_lon_grid = np.broadcast_to(lon_lin, (H, W))
        dummy_lat_grid = np.broadcast_to(lat_lin[:, None], (H, W))

    # Helper to build each level
    def build_level(fname: str, cache_dir: str, var_list: List[str], cache_index: Dict[str, np.ndarray]):
        t0 = time.perf_counter()
        LOG.info("[%s] %s: preparing availability map ...", split_name, fname)
        avail_map: Dict[str, np.ndarray] = {v: cache_index.get(v, np.array([], dtype="datetime64[h]")) for v in var_list}
        missing_all = sum(1 for v in var_list if avail_map[v].size == 0)
        if missing_all:
            LOG.warning("[%s] %s: %d/%d vars have NO samples; will fall back to zeros.",
                        split_name, fname, missing_all, len(var_list))

        stats_dir = Path(zarr_base) / fname / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        T = len(datetime_array)
        C = len(var_list)

        data_arr = np.empty((T, C, H, W), dtype=np.float32 if cast_float32 else np.float64)
        subs_log: List[str] = []

        # streaming stats accumulators (per-channel)
        sum_   = np.zeros((C,), dtype=np.float64)
        sumsq  = np.zeros((C,), dtype=np.float64)
        count  = np.zeros((C,), dtype=np.float64)

        def _accumulate(block3: np.ndarray):
            m = np.isfinite(block3)
            cnt = m.reshape(C, -1).sum(axis=1, dtype=np.float64)
            b = np.where(m, block3, 0.0)
            s  = b.reshape(C, -1).sum(axis=1, dtype=np.float64)
            ss = (b * b).reshape(C, -1).sum(axis=1, dtype=np.float64)
            return cnt, s, ss

        # Fan out
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _load_timestamp_block,
                    idx, datetime_array, cache_dir, var_list, avail_map,
                    lon_min, lon_max, lat_min, lat_max, domain_size, resample, cast_float32, dummy_data
                )
                for idx in range(T)
            ]

            for i, fut in enumerate(as_completed(futures), 1):
                idx, block, subs = fut.result()  # block: (C,H,W), orientation preserved
                data_arr[idx] = block
                cnt, s, ss = _accumulate(block)
                count += cnt
                sum_  += s
                sumsq += ss
                if subs:
                    subs_log.extend(subs)
                if i % 500 == 0 or i == T:
                    LOG.info("[%s] %s: collected %d/%d", split_name, fname, i, T)

        LOG.info("[%s] %s: stacked array shape=%s | substitutions=%d",
                 split_name, fname, tuple(data_arr.shape), len(subs_log))

        # finalize streaming stats
        LOG.info("[%s] %s: computing stats (streaming) ...", split_name, fname)
        safe_count = np.maximum(count, 1.0)  # avoid div-by-zero
        means_f64 = np.divide(sum_, safe_count, out=np.zeros_like(sum_), where=count > 0)
        ex2 = np.divide(sumsq, safe_count, out=np.zeros_like(sumsq), where=count > 0)
        var = np.maximum(ex2 - means_f64**2, 0.0)
        means = means_f64.astype(np.float32)
        stds  = np.sqrt(var).astype(np.float32)

        np.save(stats_dir / "means.npy", means)
        np.save(stats_dir / "stds.npy",  stds)
        LOG.info("[%s] %s: saved stats under %s", split_name, fname, stats_dir)

        # Write Zarr (coords = dummy grids; orientation preserved)
        ds = xr.Dataset(
            { f"{fname}": (["time", "channel", "y", "x"], data_arr) },
            coords={
                "time": datetime_array,
                "channel": var_list,
                "latitude":  (["y", "x"], dummy_lat_grid),
                "longitude": (["y", "x"], dummy_lon_grid),
            }
        )
        enc = {f"{fname}": {"dtype": "float32" if cast_float32 else "float64", "compressor": None}}
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

    # Build HighRes / LowRes using the prebuilt indexes
    build_level("HighRes", cache_highres, vars_highres, hr_index)
    build_level("LowRes",  cache_lowres,  vars_lowres,  lr_index)

    # Invariants (pulled from HighRes cache at one timestamp where all exist)
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

    # Extract/resize invariants and stack (orientation preserved)
    H, W = domain_size
    for var in invariants:
        LOG.info("processing invariant %s", var)
        path = npz_path_for(cache_highres, var, dt0)
        dt_data, lon_grid, lat_grid, _ = u1.extract_region(
            path, var, lon_min, lon_max, lat_min, lat_max,
            domain_size=domain_size, resample=resample, cast_float32=cast_float32
        )
        dt_data = _ensure_3d(dt_data)
        if dt_data.shape[-2:] != (H, W):
            # IMPORTANT: keep whatever orientation util_extract returns; only resize if needed
            dt_data, lon_grid, lat_grid = u1.interp_to_domain(lon_grid, lat_grid, dt_data, domain_size, method="linear")
        inv_arr = dt_data.copy() if inv_arr is None else np.concatenate((inv_arr, dt_data), axis=0)

    # Write invariants with the same dummy grids
    inv_ds = xr.Dataset(
        { "HighRes_invariants": (["channel", "y", "x"], inv_arr) },
        coords={
            "channel":   invariants,
            "latitude":  (["y", "x"], dummy_lat_grid),
            "longitude": (["y", "x"], dummy_lon_grid),
        }
    )
    inv_enc = {"HighRes_invariants": {"dtype": "float32" if cast_float32 else "float64", "compressor": None}}
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
        description="Build HighRes/LowRes Zarr datasets from NPZ caches (fast slicing + optional resampling), preserving original orientation."
    )
    # Variables (optional; fall back to defaults if omitted)
    ap.add_argument("--var-lowres",  default=None,
                    help="Comma-separated vars for LowRes. If omitted, defaults are used.")
    ap.add_argument("--var-highres", default=None,
                    help="Comma-separated vars for HighRes. If omitted, defaults are used.")
    ap.add_argument("--var-dummy",   default=DEFAULT_DUMMY_VAR,
                    help=f"Var used to infer dummy shape (default: {DEFAULT_DUMMY_VAR}).")
    ap.add_argument("--invariants",  default=None,
                    help=f"Comma-separated invariant vars (default: {','.join(DEFAULT_INVARIANTS)}).")

    # Region / domain
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--domain-size", required=True, help="H,W (e.g., 224,128)")

    # Paths
    ap.add_argument("--cache-highres", required=True, help="Folder with HighRes NPZ files (e.g., rwrf cache).")
    ap.add_argument("--cache-lowres",  required=True, help="Folder with LowRes NPZ files (e.g., era5 cache).")
    ap.add_argument("--zarr-base",     required=True, help="Output base directory for Zarr stores and stats.")
    ap.add_argument("--experiment-name", required=True, help="Base name used in output Zarrs.")

    # Splits & ranges
    ap.add_argument("--train-ranges", required=True,
                    help="Comma-separated date ranges A:B (YYYY/MM/DD:YYYY/MM/DD).")
    ap.add_argument("--valid-ranges", required=True,
                    help="Comma-separated date ranges A:B (YYYY/MM/DD:YYYY/MM/DD) for validation.")
    ap.add_argument("--split", choices=["train", "valid", "both"], default="both",
                    help="Which split(s) to build.")

    # Performance / IO
    ap.add_argument("--resample", choices=["downsample","interp"], default="downsample",
                    help="Resize method when domain_size differs from raw grid. 'downsample' is faster.")
    ap.add_argument("--cast-float32", dest="cast_float32", action="store_true", default=True,
                    help="Cast outputs to float32 (default: on).")
    ap.add_argument("--no-cast-float32", dest="cast_float32", action="store_false",
                    help="Disable float32 cast; keep original dtype.")

    # MP + Logging
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of worker processes for multiprocessing.")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--log-file", default=None, help="Optional path to write logs.")

    args = ap.parse_args()
    setup_logging(args.log_level, args.log_file)

    # Resolve variable lists
    vars_lr  = parse_csv(args.var_lowres)  or DEFAULT_LOWRES_VARS
    vars_hr  = parse_csv(args.var_highres) or DEFAULT_HIGHRES_VARS
    inv_vars = parse_csv(args.invariants)  or DEFAULT_INVARIANTS
    dummy_var = args.var_dummy
    domain_size = parse_domain_size(args.domain_size)

    LOG.info("Starting build experiment=%s split=%s workers=%d", args.experiment_name, args.split, args.workers)
    LOG.info("HighRes vars=%d | LowRes vars=%d | Invariants=%s | Dummy=%s | Resample=%s | float32=%s",
             len(vars_hr), len(vars_lr), ",".join(inv_vars), dummy_var, args.resample, args.cast_float32)
    LOG.info("HighRes cache: %s | LowRes cache: %s", args.cache_highres, args.cache_lowres)

    train_dt = concat_hourly(parse_ranges(args.train_ranges))
    valid_dt = concat_hourly(parse_ranges(args.valid_ranges))

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
            resample=args.resample,
            cast_float32=args.cast_float32,
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
            resample=args.resample,
            cast_float32=args.cast_float32,
            zarr_base=args.zarr_base,
            experiment_name=args.experiment_name,
            workers=args.workers,
        )

    LOG.info("All done.")

if __name__ == "__main__":
    main()