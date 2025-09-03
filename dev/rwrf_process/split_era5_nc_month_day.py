#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import xarray as xr
import datetime as dt

# ---------- per-hour split internals (reused for each monthly file) ----------

def _derive_prefix(stem: str) -> str:
    m = re.match(r"^([A-Za-z0-9]+)_[0-9]{4,}$", stem)
    return m.group(1) if m else stem.split("_", 1)[0]

def _hour_code(ts: np.datetime64) -> str:
    s = np.datetime_as_string(ts, unit="h")
    return s.replace("-", "").replace("T", "").replace(":", "")

def _build_encoding(ds: xr.Dataset, enable: bool, complevel: int = 4):
    if not enable:
        return None
    enc = {}
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.number):
            enc[name] = {"zlib": True, "complevel": complevel}
        else:
            enc[name] = {}
    return enc

def _worker_hour(args):
    (in_nc, out_dir, time_dim, engine, idx, ts_val, prefix,
     compress, complevel, overwrite) = args
    out_dir = Path(out_dir)
    code = _hour_code(ts_val)
    out_path = out_dir / f"{prefix}_{code}.nc"
    if out_path.exists() and not overwrite:
        return f"[skip] {out_path}"
    tmp_path = out_path.with_suffix(".nc.part")
    ds = xr.open_dataset(in_nc, engine=engine)
    try:
        sub = ds.isel({time_dim: slice(idx, idx + 1)}).copy()
        sub.attrs = dict(ds.attrs) if ds.attrs else {}
        note = f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z split-by-hour -> {out_path.name}"
        sub.attrs["history"] = (sub.attrs.get("history", "") + ("\n" if sub.attrs.get("history") else "") + note)
        enc = _build_encoding(sub, enable=compress, complevel=complevel)
        sub.to_netcdf(tmp_path, engine=engine, encoding=enc)
        os.replace(tmp_path, out_path)
    finally:
        ds.close()
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
    return f"[ok]   {out_path}"

def split_one_month_file(nc_path: Path, out_dir: Path, time_dim: str,
                         engine: str, workers: int, compress: bool,
                         complevel: int, overwrite: bool, prefix: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds0 = xr.open_dataset(nc_path, engine=engine)
    try:
        if time_dim not in ds0.coords and time_dim not in ds0.dims:
            raise ValueError(f"{nc_path}: time coord/dim '{time_dim}' not found")
        t = ds0[time_dim]
        if not np.issubdtype(t.dtype, np.datetime64):
            raise TypeError(f"{nc_path}: '{time_dim}' must be datetime64; got {t.dtype}")
        ts_vals = t.values
    finally:
        ds0.close()
    if prefix is None:
        prefix = _derive_prefix(nc_path.stem)

    work = [
        (str(nc_path), str(out_dir), time_dim, engine, i, ts_vals[i], prefix,
         compress, complevel, overwrite)
        for i in range(ts_vals.shape[0])
    ]
    print(f"\n==> Splitting {nc_path} -> {out_dir} ({len(work)} hours)")
    with Pool(processes=workers) as pool:
        for msg in pool.imap_unordered(_worker_hour, work, chunksize=8):
            print(msg)

# ------------------------------- recursion -----------------------------------

def find_monthly_nc_files(root: Path) -> list[Path]:
    # Match files like var_YYYYMM.nc (e.g., mslp_201909.nc)
    rx = re.compile(r".*_[0-9]{6}\.nc$")
    return [p for p in root.rglob("*.nc") if rx.match(p.name)]

def compute_out_dir(nc_path: Path, root: Path, out_root: Path | None) -> Path:
    if out_root is None:
        return nc_path.parent 
    # mirror tree under out_root
    rel = nc_path.parent.relative_to(root)
    return out_root / rel

def main():
    ap = argparse.ArgumentParser(
        description="Recursively split monthly NetCDF files into per-hour files."
    )
    ap.add_argument("root", type=Path,
                    help="Root directory to search (e.g., NCDR_StormCast/data/ERA5_nc/mslp)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Mirror outputs under this directory (default: <month_dir>/hourly)")
    ap.add_argument("--time-dim", default="valid_time", help="Datetime coord/dim name.")
    ap.add_argument("--engine", default="netcdf4",
                    choices=["netcdf4", "scipy", "h5netcdf"], help="IO engine.")
    ap.add_argument("-w", "--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="Workers per monthly file (processes).")
    ap.add_argument("--no-compress", action="store_true", help="Disable zlib compression.")
    ap.add_argument("--complevel", type=int, default=4, help="zlib compression level 0-9.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    ap.add_argument("--prefix", default=None,
                    help="Override filename prefix (defaults to token before underscore).")
    args = ap.parse_args()

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "TRUE")

    monthly_files = find_monthly_nc_files(args.root)
    if not monthly_files:
        print(f"No monthly NetCDF files found under: {args.root}")
        return

    print(f"Found {len(monthly_files)} monthly files.")
    for nc_path in sorted(monthly_files):
        out_dir = compute_out_dir(nc_path, args.root, args.out_root)
        try:
            split_one_month_file(
                nc_path=nc_path,
                out_dir=out_dir,
                time_dim=args.time_dim,
                engine=args.engine,
                workers=args.workers,
                compress=(not args.no_compress),
                complevel=args.complevel,
                overwrite=args.overwrite,
                prefix=args.prefix,
            )
        except Exception as e:
            print(f"[error] {nc_path}: {e}")

if __name__ == "__main__":
    main()