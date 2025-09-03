from utils.config import CONFIG
from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
import numpy as np
import argparse
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Setup logger
LOGFILE = "era5_2_npz_log.txt"

# Remove existing handlers to avoid duplication when rerun in notebooks
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

@dataclass
class VariableConfig:
    """Configuration for different ERA5 variables"""
    nc_var_name: str
    file_prefix: str

VARIABLE_CONFIGS = {
    # surface / single-level
    "mslp":  VariableConfig("msl",   "mslp"),
    "sp":    VariableConfig("sp",    "sp"),
    "t2m":   VariableConfig("t2m",   "t2m"),
    "u10":   VariableConfig("u10",   "u10"),
    "v10":   VariableConfig("v10",   "v10"),
    "tp":    VariableConfig("tp",    "tp"),      # precip (folder is 'tp')
    "tcwv":  VariableConfig("tcwv",  "tcwv"),

    # humidity (q) on pressure levels
    "q1000": VariableConfig("q",     "q1000"),
    "q850":  VariableConfig("q",     "q850"),
    "q500":  VariableConfig("q",     "q500"),
    "q250":  VariableConfig("q",     "q250"),

    # temperature (t) on pressure levels
    "t1000": VariableConfig("t",     "t1000"),
    # "t925":  VariableConfig("t",     "t925"),
    "t850":  VariableConfig("t",     "t850"),
    "t500":  VariableConfig("t",     "t500"),
    "t250":  VariableConfig("t",     "t250"),

    # wind U-component on pressure levels
    "u1000": VariableConfig("u",     "u1000"),
    # "u925":  VariableConfig("u",     "u925"),
    "u850":  VariableConfig("u",     "u850"),
    # "u700":  VariableConfig("u",     "u700"),
    "u500":  VariableConfig("u",     "u500"),
    "u250":  VariableConfig("u",     "u250"),
    # "u200":  VariableConfig("u",     "u200"),

    # wind V-component on pressure levels
    "v1000": VariableConfig("v",     "v1000"),
    # "v925":  VariableConfig("v",     "v925"),
    "v850":  VariableConfig("v",     "v850"),
    # "v700":  VariableConfig("v",     "v700"),
    "v500":  VariableConfig("v",     "v500"),
    "v250":  VariableConfig("v",     "v250"),
    # "v200":  VariableConfig("v",     "v200"),

    # geopotential height (z) on pressure levels
    "z1000": VariableConfig("z",     "z1000"),
    "z850":  VariableConfig("z",     "z850"),
    "z500":  VariableConfig("z",     "z500"),
    "z250":  VariableConfig("z",     "z250"),
}




class ERA5Processor:
    """Handles ERA5 data processing and conversion to numpy arrays"""
    


    def __init__(self, era5_base_path: str, output_base_path: str, overwrite: bool):
        self.era5_base_path = Path(era5_base_path)
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True) 
        self._overwrite = overwrite
    
    def _get_file_path(self, date: datetime, hour: int, variable: str) -> Path:
        """Generate the file path for a given date, hour, and variable"""
        config = VARIABLE_CONFIGS[variable]
        hour_str = f"{hour:02d}"
        date_str = date.strftime('%Y%m%d')
        
        return self.era5_base_path / variable / date.strftime('%Y') / f"{config.file_prefix}_{date_str}{hour_str}.nc"

    
    def _load_dataset(self, date: datetime, hour: int, variable: str) -> Optional[Dataset]:
        """Load ERA5 dataset for given parameters"""
        filepath = self._get_file_path(date, hour, variable)
        
        if not filepath.exists():
            raise FileNotFoundError(f"NC file not found: {filepath}")
        
        return Dataset(str(filepath), mode="r")
    
    def _extract_data(self, ds: Dataset, variable: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract data arrays from dataset"""
        config = VARIABLE_CONFIGS[variable]  # <-- fix: use module-level dict

        # Common coordinates
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        # Some ERA5 files name this "time" instead of "valid_time"
        times_var = "valid_time" if "valid_time" in ds.variables else "time"
        times = ds.variables[times_var][:]

        # Variable-specific data
        data = ds.variables[config.nc_var_name][:]

        return data, lat, lon, times
    
    def process_single_time(self, date: datetime, hour: int, variable: str) -> bool:
        """Process a single time step and save as numpy array"""

        filename = f"{variable}_{date.strftime('%Y%m%d')}_{hour:02d}.npz"
        output_path = self.output_base_path / filename

        logger.info("START: var=%s date=%s hour=%02d", variable, date.strftime("%Y-%m-%d"), hour)

        if output_path.exists() and not self._overwrite:
            logger.info("SKIP (exists): var=%s date=%s hour=%02d path=%s",
                        variable, date.strftime("%Y-%m-%d"), hour, output_path)
            return False

        try:
            ds = self._load_dataset(date, hour, variable)
        except FileNotFoundError as e:
            logger.warning("MISSING: var=%s date=%s hour=%02d (%s)",
                           variable, date.strftime("%Y-%m-%d"), hour, e)
            return False

        try:
            logger.debug("Dataset vars for %s: %s", variable, list(ds.variables.keys()))

            # Extract data
            data, lat, lon, times = self._extract_data(ds, variable)

            # Save as .npz
            np.savez(
                str(output_path),
                **{variable: data},
                lat=lat,
                lon=lon,
                times=times
            )

            logger.info("DONE: var=%s date=%s hour=%02d saved to %s",
                        variable, date.strftime("%Y-%m-%d"), hour, output_path)
            return True

        except Exception as e:
            logger.error("FAIL: var=%s date=%s hour=%02d error=%s",
                         variable, date.strftime("%Y-%m-%d"), hour, e)
            return False
        finally:
            ds.close()
    
    from typing import List, Union

    def process_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        variables: Union[str, List[str]],
        hour_step: int = 1,
        max_workers: Optional[int] = None,
    ) -> Dict[str, int]:
        """Process a range of dates (for one or many variables) in parallel."""
        stats = {"processed": 0, "skipped": 0, "errors": 0}

        # Normalize variables -> List[str]
        if isinstance(variables, str):
            var_list = [variables]
        else:
            var_list = list(variables)

        # Build all tasks
        tasks = []
        current = start_date
        while current <= end_date:
            for hour in range(0, 24, hour_step):
                for v in var_list:
                    tasks.append((current, hour, v))
            current += timedelta(days=1)

        logger.info("Total tasks to process: %d", len(tasks))

        # Determine workers
        cpu_workers = multiprocessing.cpu_count()
        if max_workers is None:
            max_workers = cpu_workers
        else:
            max_workers = max(1, min(max_workers, cpu_workers))
        logger.info("Using up to %d workers (cpu=%d)", max_workers, cpu_workers)

        # Run tasks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_task, (self, d, h, v)) for (d, h, v) in tasks]

            for f in as_completed(futures):
                try:
                    status = f.result()
                    stats[status] += 1
                except Exception:
                    logger.exception("Unexpected future error")
                    stats["errors"] += 1

        return stats

def worker_task(args):
    processor, date, hour, variable = args
    try:
        ok = processor.process_single_time(date, hour, variable)
        return "processed" if ok else "skipped"
    except Exception as e:
        logger.exception("CRASHED worker for var=%s date=%s hour=%02d",
                         variable, date.strftime("%Y-%m-%d"), hour)
        return "errors"

def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5 data and save as numpy arrays."
    )

    all_choices = sorted(VARIABLE_CONFIGS.keys())

    parser.add_argument(
        "--variable",
        required=True,
        nargs="+",   # <-- allows multiple
        choices=all_choices,
        help="Variable(s) to extract ..."
    )
    parser.add_argument(
        "--start-date",
        default="2019/08/03",
        help="Start date (YYYY/MM/DD). Default: 2019/08/03",
    )
    parser.add_argument(
        "--end-date",
        default="2019/08/04",
        help="End date (YYYY/MM/DD). Default: 2019/08/04",
    )
    parser.add_argument(
        "--hour-step",
        type=int,
        default=6,
        help="Hour step between outputs (default 1). Use 6 for every 6 hours.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--input-dir",
        default=str(CONFIG.era5),
        help="Input ERA5 base directory (default: CONFIG.era5)"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Output base directory for .npz files (will be created if missing)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files (default: skip if exists).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers (default: all cores)."
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y/%m/%d")
        end_date = datetime.strptime(args.end_date, "%Y/%m/%d")
    except ValueError as e:
        logger.error("Invalid date format: %s", e)
        return 1
    
    # Initialize processor
    processor = ERA5Processor(args.input_dir, args.output_dir, overwrite=args.overwrite)
    
    # Process data
    logger.info("Starting processing: %s from %s to %s (step: %dh)", 
                args.variable, args.start_date, args.end_date, args.hour_step)
    
    stats = processor.process_date_range(start_date, end_date, args.variable, args.hour_step)
    
    logger.info("Summary for var=%s: processed=%d skipped=%d errors=%d",
                args.variable, stats["processed"], stats["skipped"], stats["errors"])
    
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    exit(main())