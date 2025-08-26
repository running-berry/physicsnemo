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

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VariableConfig:
    """Configuration for different ERA5 variables"""
    nc_var_name: str
    file_prefix: str


class ERA5Processor:
    """Handles ERA5 data processing and conversion to numpy arrays"""
    
    VARIABLE_CONFIGS = {
        # t1000  t2m  t925  u10  u1000  u200  u250  u500  u700  u850  u925  v10  v1000  v200  v250  v500  v700  v850  v925
        "t2m": VariableConfig("t2m", "t2m"),
        "u10": VariableConfig("u10", "u10"),
        "pptn": VariableConfig("tp", "tp"),
        "t1000": VariableConfig("t", "t1000"),
        "t925": VariableConfig("t", "t925"),
        "u1000": VariableConfig("u", "u1000"),
        "u200": VariableConfig("u", "u200"),
        "u250": VariableConfig("u", "u250"),
        "u500": VariableConfig("u", "u500"),
        "u700": VariableConfig("u", "u700"),
        "u850": VariableConfig("u", "u850"),
        "u925": VariableConfig("u", "u925"),
        "v10": VariableConfig("v10", "v10"),
        "v1000": VariableConfig("v", "v1000"),
        "v200": VariableConfig("v", "v200"),
        "v250": VariableConfig("v", "v250"),
        "v500": VariableConfig("v", "v500"),
        "v700": VariableConfig("v", "v700"),
        "v850": VariableConfig("v", "v850"),
        "v925": VariableConfig("v", "v925"),
    }
    
    def __init__(self, era5_base_path: str, output_base_path: str):
        self.era5_base_path = Path(era5_base_path)
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True) 
    
    def _get_file_path(self, date: datetime, hour: int, variable: str) -> Path:
        """Generate the file path for a given date, hour, and variable"""
        config = self.VARIABLE_CONFIGS[variable]
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
        config = self.VARIABLE_CONFIGS[variable]
        
        # Common coordinates
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        times = ds.variables["valid_time"][:]
        
        # Variable-specific data
        data = ds.variables[config.nc_var_name][:]
        
        return data, lat, lon, times
    
    def process_single_time(self, date: datetime, hour: int, variable: str) -> bool:
        """Process a single time step and save as numpy array"""
        try:
            ds = self._load_dataset(date, hour, variable)
        except FileNotFoundError as e:
            logger.warning("%s — skipping %s %02d:00", e, date.strftime('%Y/%m/%d'), hour)
            return False
        
        try:
            logger.debug("Variables in dataset: %s", list(ds.variables.keys()))
            
            # Extract data
            data, lat, lon, times = self._extract_data(ds, variable)
            
            # Ensure output directory exists
            output_dir = self.output_base_path 
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as .npz
            filename = f"{variable}_{date.strftime('%Y%m%d')}_{hour:02d}.npz"
            output_path = output_dir / filename
            
            np.savez(
                str(output_path),
                **{variable: data},
                lat=lat,
                lon=lon,
                times=times
            )
            
            logger.info("Saved arrays to %s", output_path)
            return True
            
        except Exception as e:
            logger.error("Error processing %s %02d:00: %s", date.strftime('%Y/%m/%d'), hour, e)
            return False
        finally:
            ds.close()
    
    def process_date_range(self, start_date: datetime, end_date: datetime, 
                          variable: str, hour_step: int = 1) -> Dict[str, int]:
        """Process a range of dates"""
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        
        current = start_date
        while current <= end_date:
            for hour in range(0, 24, hour_step):
                logger.info("Processing %s %02d:00", current.strftime('%Y/%m/%d'), hour)
                
                try:
                    if self.process_single_time(current, hour, variable):
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1
                except Exception:
                    logger.exception("Unexpected error for %s %02d:00", 
                                   current.strftime('%Y/%m/%d'), hour)
                    stats["errors"] += 1
            
            current += timedelta(days=1)
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5 data and save as numpy arrays."
    )
    parser.add_argument(
        "--variable",
        required=True,
        # t1000  t2m  t925  u10  u1000  u200  u250  u500  u700  u850  u925  v10  v1000  v200  v250  v500  v700  v850  v925
        choices=["t2m", "u10", "pptn", "t1000", "t925", "u1000", "u200", "u250", "u500", "u700", "u850", "u925", "v10", "v1000", "v200", "v250", "v500", "v700", "v850", "v925"],
        help="Variable to extract: t2m, u10, or pptn",
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
        "--output-dir", default=str(CONFIG.era5_npz),
        help="Output base directory for .npz files (will be created if missing)")
    
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
    processor = ERA5Processor(CONFIG.era5, args.output_dir)
    
    # Process data
    logger.info("Starting processing: %s from %s to %s (step: %dh)", 
                args.variable, args.start_date, args.end_date, args.hour_step)
    
    stats = processor.process_date_range(start_date, end_date, args.variable, args.hour_step)
    
    logger.info("Processing complete. Processed: %d, Skipped: %d, Errors: %d", 
                stats["processed"], stats["skipped"], stats["errors"])
    
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    exit(main())