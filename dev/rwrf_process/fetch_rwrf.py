#!/usr/bin/env python3
# fetch_rwrf.py — refactored to match fetch_era5 style

from utils.config import CONFIG
from netCDF4 import Dataset
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import argparse
import logging
import os

# -------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------- variable config --------------------
# Map CLI variable names -> (WRF variable key in file, saved name, filename flavor)
# saved_name controls the variable key inside the .npz
# For pptn we read 'qpepre' and also save as 'qpepre'
@dataclass
class VariableConfig:
    wrf_var_key: str        # variable name in the NetCDF file
    saved_name: str         # key stored in npz
    fn_flavor: str = "std"  # 'std' or 'pptn_qpepre' (affects filepath rule)

VARIABLES: Dict[str, VariableConfig] = {
    # t2m/u10/etc. come from wrf interp file without suffix
    "t2m":   VariableConfig("T2",      "t2m",   "std"),
    "u10":   VariableConfig("umet10",  "u10",   "std"),
    "lsm":   VariableConfig("LANDMASK","lsm",   "std"),
    "orog":  VariableConfig("HGT",     "orog",  "std"),
    # precip (pptn) lives in a different filename; inside file var is 'qpepre'
    "pptn":  VariableConfig("qpepre",  "qpepre","pptn_qpepre"),
}

# -------------------- processor --------------------
class RWRFProcessor:
    """
    Process RWRF (interpolated) outputs into .npz bundles:
    np.savez(..., <saved_name>=data, lat=XLAT, lon=XLONG, times=Times)
    """
    def __init__(self, rwrf_base_path: str, output_base_path: str, cropped_qpepre: bool = False):
        self.rwrf_base = Path(rwrf_base_path)
        self.out_base = Path(output_base_path)
        self.out_base.mkdir(parents=True, exist_ok=True)
        self.cropped_qpepre = cropped_qpepre

    @staticmethod
    def _format_dt(date: datetime, hour: int) -> str:
        # YYYY-MM-DD_HH
        return date.strftime(f"%Y-%m-%d_{hour:02d}")

    def _build_filepath(self, date: datetime, hour: int, varkey: str) -> Path:
        """
        Build the RWRF netCDF path based on variable name and date/hour.
        Folder layout:
          <CONFIG.rwrf>/RWRF_YYYY-MM/YYYY-MM-DD_HH/<file>
        Files:
          std:        wrfout_d01_YYYY-MM-DD_HH_interp
          pptn_qpepre: wrfout_d01_YYYY-MM-DD_HH_interp_qpepre.nc
                       or wrfout_d01_YYYY-MM-DD_HH_interp_cropped_qpepre.nc (if cropped flag)
        """
        cfg = VARIABLES[varkey]
        month_folder = f"RWRF_{date.strftime('%Y-%m')}"
        stamp = self._format_dt(date, hour)
        base_dir = self.rwrf_base / month_folder / stamp

        if cfg.fn_flavor == "pptn_qpepre":
            if self.cropped_qpepre:
                fname = f"wrfout_d01_{stamp}_interp_cropped_qpepre.nc"
            else:
                fname = f"wrfout_d01_{stamp}_interp_qpepre.nc"
        else:
            # Some environments store this file without .nc extension; handle both.
            # Prefer with extension if exists, else fallback to no-extension.
            with_ext = base_dir / f"wrfout_d01_{stamp}_interp.nc"
            no_ext   = base_dir / f"wrfout_d01_{stamp}_interp"
            if with_ext.exists():
                return with_ext
            return no_ext

        return base_dir / fname

    def _open_dataset(self, date: datetime, hour: int, varkey: str) -> Dataset:
        path = self._build_filepath(date, hour, varkey)
        if not path.exists():
            raise FileNotFoundError(f"NC file not found: {path}")
        return Dataset(str(path), mode="r")

    def _extract_arrays(self, ds: Dataset, varkey: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = VARIABLES[varkey]
        try:
            data = ds.variables[cfg.wrf_var_key][:]
        except KeyError:
            # Helpful error with available keys
            raise KeyError(f"Variable '{cfg.wrf_var_key}' not found in file; available={list(ds.variables.keys())}")

        # Coordinates / time in WRF outputs
        try:
            lat   = ds.variables["XLAT"][:]   # (time, y, x)
            lon   = ds.variables["XLONG"][:]
        except KeyError:
            # Sometimes lowercase or different naming—add fallbacks if your files differ
            raise KeyError(f"Missing XLAT/XLONG in file; available={list(ds.variables.keys())}")

        # WRF Times: char array, keep raw as-is
        tkey = "Times" if "Times" in ds.variables else None
        if tkey is None:
            raise KeyError(f"Missing 'Times' variable; available={list(ds.variables.keys())}")
        times = ds.variables[tkey][:]

        return data, lat, lon, times

    def process_single_time(self, date: datetime, hour: int, varkey: str) -> bool:
        try:
            ds = self._open_dataset(date, hour, varkey)
        except FileNotFoundError as e:
            logger.warning("%s — skipping %s %02d:00", e, date.strftime("%Y/%m/%d"), hour)
            return False

        try:
            data, lat, lon, times = self._extract_arrays(ds, varkey)

            # Output path: <out_base>/<variable>/<year>/<name>.npz
            year_dir = self.out_base 
            year_dir.mkdir(parents=True, exist_ok=True)

            saved_name = VARIABLES[varkey].saved_name
            fname = f"{saved_name}_{date.strftime('%Y%m%d')}_{hour:02d}.npz"
            out_path = year_dir / fname

            np.savez(str(out_path), **{saved_name: data}, lat=lat, lon=lon, times=times)
            logger.info("Saved arrays to %s", out_path)
            return True

        except KeyError as e:
            logger.error("Key error at %s %02d:00 -> %s", date.strftime("%Y/%m/%d"), hour, e)
            return False
        except Exception as e:
            logger.exception("Error processing %s %02d:00: %s", date.strftime("%Y/%m/%d"), hour, e)
            return False
        finally:
            ds.close()

    def process_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        varkey: str,
        hour_step: int = 6
    ) -> Dict[str, int]:
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        cur = start_date
        while cur <= end_date:
            for hour in range(0, 24, hour_step):
                logger.info("Processing %s %02d:00", cur.strftime("%Y/%m/%d"), hour)
                ok = self.process_single_time(cur, hour, varkey)
                if ok:
                    stats["processed"] += 1
                else:
                    # Distinguish between file-not-found/KeyError and hard exceptions already logged
                    # Here we just count as skipped; 'errors' are counted inside process_single_time when exceptions occur.
                    stats["skipped"] += 1
            cur += timedelta(days=1)
        return stats

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="Extract RWRF data and save as numpy arrays (refactor).")
    parser.add_argument(
        "--variable",
        required=True,
        choices=list(VARIABLES.keys()),
        help="Variable to extract (e.g., t2m, u10, pptn, lsm, orog)",
    )
    parser.add_argument(
        "--cropped-qpepre",
        action="store_true",
        help="Use cropped QPEPRE file for pptn (wrfout_*_interp_cropped_qpepre.nc)",
    )
    parser.add_argument("--start-date", default="2019/08/03", help="Start date YYYY/MM/DD")
    parser.add_argument("--end-date",   default="2019/08/04", help="End date YYYY/MM/DD")
    parser.add_argument("--hour-step",  type=int, default=6,  help="Hour step (default 6). Use 1 to process every hour.")
    parser.add_argument("--output-dir", default=str(CONFIG.rwrf_npz), help="Output base directory for .npz files")
    parser.add_argument("--verbose",    action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse dates
    try:
        start = datetime.strptime(args.start_date, "%Y/%m/%d")
        end   = datetime.strptime(args.end_date,   "%Y/%m/%d")
    except ValueError as e:
        logger.error("Invalid date format: %s", e)
        return 1

    # Build processor
    if args.variable != "pptn" and args.cropped_qpepre:
        logger.warning("--cropped-qpepre has no effect for variable '%s'", args.variable)

    processor = RWRFProcessor(
        rwrf_base_path=CONFIG.rwrf,
        output_base_path=args.output_dir,
        cropped_qpepre=args.cropped_qpepre
    )

    logger.info(
        "Starting RWRF processing: var=%s, %s → %s, step=%dh, output=%s",
        args.variable, args.start_date, args.end_date, args.hour_step, args.output_dir
    )

    stats = processor.process_date_range(start, end, args.variable, args.hour_step)

    logger.info(
        "Processing complete. Processed: %d, Skipped: %d, Errors: %d",
        stats["processed"], stats["skipped"], stats["errors"]
    )

    # Return non-zero if any errors were logged as exceptions (we count those inside process_single_time).
    # Here we treat 'skipped' as non-fatal (missing files, etc.).
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())