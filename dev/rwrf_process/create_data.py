import logging
import os
import sys
from datetime import datetime

import numpy as np
import util_extract as u1
import xarray as xr
import yaml
import zarr
from utils import CONFIG


# Configure logging
def setup_logger():
    """Setup logger with timestamp and process information."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d: - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    # log_file = f"create_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


logger = setup_logger()

with open("../../examples/weather/stormcast/config/dataset/small.yaml", "r") as f:
    cfg_mod = yaml.safe_load(f)

with open("./config.yaml", "r") as f:
    cfg_dev = yaml.safe_load(f)

channel_vars = {
    "LowRes": cfg_dev["var_lowres"],
    "HighRes": cfg_dev["var_highres"],
    "dummy": cfg_dev["var_highres"][0] if cfg_dev["var_highres"] else "t2m",
}
invariants = cfg_dev["invariants"]
lon_min, lon_max = cfg_dev["lon_min"], cfg_dev["lon_max"]
lat_min, lat_max = cfg_dev["lat_min"], cfg_dev["lat_max"]
num_channel = len(channel_vars)
domain_size = tuple(cfg_mod["HighRes_img_size"])
train_datetime_start = cfg_mod["train_dates"][0]
train_datetime_last = cfg_mod["train_dates"][-1]
valid_datetime_start = cfg_mod["valid_dates"][0]
valid_datetime_last = cfg_mod["valid_dates"][-1]

cache_base = CONFIG.cache
data_base = "../data"
experiment_name = cfg_mod["exp_train_zarrs"][0]


def create_dummy_arr(
    fname,
    dt,
    data_path: str,
    data_var: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
):
    """
    Create a dummy data array (filled with NaNs) matching the shape
    of the real data for timestamp dt, and also return the lon/lat grids
    and time coords.
    """
    logger.debug(f"Creating dummy array for {data_var} at {dt}")

    # build the filename for this dt
    # yy, mm, dd, hh = np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
    # fn = f"{data_var}_{yy}{mm}{dd}{hh}.npz"
    # dt_path = os.path.join(data_path, fn)
    dt_path = f"{cache_base}/dummy/dummy_{fname}.npz"

    logger.debug(f"Using template file: {dt_path}")

    # grab one real sample to infer shapes
    real_arr, lon_grid, lat_grid, times = u1.extract_region(
        dt_path,
        data_var,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        domain_size=domain_size,
    )

    Ny_full, Nx_full = lat_grid.shape
    Ny_tgt, Nx_tgt = domain_size

    if Ny_full < Ny_tgt or Nx_full < Nx_tgt:
        logger.debug(f"Interpolating from {Ny_full}x{Nx_full} to {Ny_tgt}x{Nx_tgt}")
        real_arr, lon_grid, lat_grid = u1.interp_to_domain(
            lon_grid, lat_grid, real_arr, domain_size, method="linear"
        )

    dummy_arr = np.full_like(
        real_arr, fill_value=0.0
    )  # future: or use fill_value=np.nan

    logger.debug(f"Dummy array created with shape: {dummy_arr.shape}")

    return dummy_arr, lon_grid, lat_grid, times


def process_invariants(
    cache_base: str,
    data_base: str,
    experiment_name: str,
    start_date: str,
):
    """
    Process invariant data (land-sea mask, orography) and save to zarr.
    """
    logger.info("=" * 50)
    logger.info("PROCESSING INVARIANTS")
    logger.info("=" * 50)

    # Create invariants directory
    invariant_folder = f"{data_base}/invariants"
    os.makedirs(invariant_folder, exist_ok=True)
    logger.info(f"Created invariants directory: {invariant_folder}")

    # Use rwrf cache path for invariants
    cache_path = f"{cache_base}/rwrf"
    logger.info(f"Using cache path: {cache_path}")

    # Use first datetime to get invariant data
    base_date = np.datetime64(start_date.replace("/", "-") + "T00:00:00")
    dt = base_date
    logger.info(f"Using reference date: {dt}")

    invariant_arr = None
    lon_grid = None
    lat_grid = None

    for var in invariants:
        logger.info(f"Processing invariant/{len(invariants)}: {var}")
        yy, mm, dd, hh = (
            np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
        )
        dt_path = os.path.join(cache_path, f"{var}_{yy}{mm}{dd}{hh}.npz")
        logger.info(f"Processing invariant: {dt_path}")

        try:
            dt_data, lon_grid, lat_grid, times = u1.extract_region(
                dt_path,
                var,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                domain_size=domain_size,
            )  # (1, Ny, Nx)
            dt_data, lon_grid, lat_grid = u1.interp_to_domain(
                lon_grid, lat_grid, dt_data, domain_size, method="linear"
            )  # (1, Ny, Nx)
            logger.debug(f"Successfully loaded {var} with shape: {dt_data.shape}")

        except FileNotFoundError:
            logger.error(f"Invariant file not found: {dt_path}")
            raise FileNotFoundError(f"Invariant file not found: {dt_path}")

        # concatenate data
        if invariant_arr is None:  # first iteration only
            invariant_arr = dt_data.copy()
            logger.debug(
                f"Initialized invariant array with shape: {invariant_arr.shape}"
            )
        else:
            # concatenate along axis=0 (channel)
            invariant_arr = np.concatenate((invariant_arr, dt_data), axis=0)
            logger.debug(
                f"Concatenated invariant array, new shape: {invariant_arr.shape}"
            )

    # Create xarray dataset for invariants
    year_data = xr.Dataset(
        {
            "HighRes_invariants": (["channel", "y", "x"], invariant_arr),
            "channel": invariants,
            "latitude": (["y", "x"], lat_grid),
            "longitude": (["y", "x"], lon_grid),
        }
    )
    data_enc = {"HighRes_invariants": {"dtype": "float32", "compressor": None}}
    logger.info(f"Final invariant array shape: {invariant_arr.shape}")
    logger.info(f"Invariant variables: {invariants}")

    # Save invariants
    year_data.to_zarr(
        f"{data_base}/invariants/invariants.zarr",
        mode="w",
        consolidated=True,
        encoding=data_enc,
        zarr_format=2,
    )
    zarr.consolidate_metadata(f"{data_base}/invariants/invariants.zarr")

    logger.info(f"Invariants saved to {data_base}/invariants/invariants.zarr")


def process_period(
    start_date: str,
    end_date: str,
    fname: str,
    channel_vars_dict: dict,
    cache_base: str,
    data_base: str,
    domain_size: tuple,
    experiment_name: str,
    is_validation: bool = False,
):
    """
    Process a period (train or validation) and write stats + zarr.
    start_date/end_date: strings "YYYY/MM/DD" (end_date inclusive).
    fname: "HighRes" or "LowRes"
    channel_vars_dict: dict with channel variables for each resolution
    """

    period_type = "VALIDATION" if is_validation else "TRAINING"
    logger.info("=" * 50)
    logger.info(f"PROCESSING {period_type} DATA - {fname}")
    logger.info("=" * 50)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Variables: {channel_vars_dict[fname]}")

    folder_path = f"{data_base}/{fname}/stats"
    os.makedirs(folder_path, exist_ok=True)
    logger.info(f"Created stats directory: {folder_path}")

    # determine data path base
    cache_path = f"{cache_base}/rwrf" if fname == "HighRes" else f"{cache_base}/era5"
    logger.info(f"Using cache path: {cache_path}")

    base_date = np.datetime64(start_date.replace("/", "-") + "T00:00:00")
    end_date_np = np.datetime64(end_date.replace("/", "-")) + np.timedelta64(23, "h")
    total_hours = int((end_date_np - base_date) / np.timedelta64(1, "h")) + 1
    offsets = np.arange(total_hours, dtype=np.int64)
    datetime_array = base_date + offsets * np.timedelta64(1, "h")

    logger.info(f"Processing {total_hours} hourly time steps")

    # create the dummy data format by the first data
    logger.info("Creating dummy data template...")

    dummy_data, dummy_lon_grid, dummy_lat_grid, dummy_times = create_dummy_arr(
        fname,
        datetime_array[0],
        cache_path,
        channel_vars_dict["dummy"],
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )

    missing_data = []
    data_arr = None
    processed_files = 0
    missing_files = 0
    cache_path = f"{cache_base}/rwrf/"

    for dt in datetime_array:
        channel_arr = None
        for var in channel_vars_dict[fname]:
            yy, mm, dd, hh = (
                np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
            )
            dt_path = os.path.join(cache_path, f"{var}_{yy}{mm}{dd}{hh}.npz")
            logger.info(f"Processing {dt_path}")

            try:
                dt_data, lon_grid, lat_grid, times = u1.extract_region(
                    dt_path,
                    var,
                    lon_min,
                    lon_max,
                    lat_min,
                    lat_max,
                    domain_size=domain_size,
                )  # (1, Ny, Nx)
                dt_data, lon_grid, lat_grid = u1.interp_to_domain(
                    lon_grid, lat_grid, dt_data, domain_size, method="linear"
                )  # (1, Ny, Nx)
                processed_files += 1
            except FileNotFoundError:
                # create dummy data
                dt_data = dummy_data.copy()
                missing_file = f"{yy}-{mm}-{dd} {hh}:00"
                missing_data.append(missing_file)
                missing_files += 1
                lon_grid = dummy_lon_grid
                lat_grid = dummy_lat_grid
                logger.warning(f"Missing file, using dummy data: {cache_base}/dummy/dummy_{fname}.npz")

            # concatenate data
            if channel_arr is None:  # first iteration only
                channel_arr = dt_data.copy()
            else:
                # concatenate along axis=0 (channel)
                channel_arr = np.concatenate((channel_arr, dt_data), axis=0)

        if channel_arr.ndim == 3:
            # make it (time, var, y, x)
            channel_arr = channel_arr[None, :, :, :]
        # concatenate data
        if data_arr is None:  # first iteration only
            data_arr = channel_arr.copy()
        else:
            # concatenate along axis=0 (time)
            data_arr = np.concatenate((data_arr, channel_arr), axis=0)

    logger.info(f"File processing summary:")
    logger.info(f"  - Successfully processed: {processed_files}")
    logger.info(f"  - Missing files (dummy data used): {missing_files}")
    logger.info(
        f"  - Total files expected: {total_hours * len(channel_vars_dict[fname])}"
    )

    # compute mean and std over time, latitude & longitude → leaves (n_chan,)
    # data_arr shape is (n_time, n_chan, ny, nx)
    if data_arr.ndim == 4:
        means = np.nanmean(data_arr, axis=(0, 2, 3)).astype(np.float32)
        stds = np.nanstd(data_arr, axis=(0, 2, 3)).astype(np.float32)
    elif data_arr.ndim == 3:
        m = np.nanmean(data_arr).astype(np.float32)
        s = np.nanstd(data_arr).astype(np.float32)
        means = np.array([m], dtype=np.float32)
        stds = np.array([s], dtype=np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {data_arr.ndim}D.")

    means = means.astype(np.float32)
    stds = stds.astype(np.float32)

    # Only save stats for training
    if not is_validation:
        np.save(f"{folder_path}/means.npy", means)
        np.save(f"{folder_path}/stds.npy", stds)

    logger.info(f"Data shape: {data_arr.shape}")

    # Create xarray dataset
    year_data = xr.Dataset(
        {
            f"{fname}": (["time", "channel", "y", "x"], data_arr),
            "time": datetime_array,
            "channel": channel_vars_dict[fname],
            "latitude": (["y", "x"], lat_grid),
            "longitude": (["y", "x"], lon_grid),
        }
    )
    data_enc = {f"{fname}": {"dtype": "float32", "compressor": None}}
    out_name = "valid.zarr" if is_validation else "train.zarr"
    year_data.to_zarr(
        f"{data_base}/{fname}/{out_name}",
        mode="w",
        consolidated=True,
        encoding=data_enc,
        zarr_format=2,
    )
    zarr.consolidate_metadata(f"{data_base}/{fname}/{out_name}")

    logger.info(
        f"{'Validation' if is_validation else 'Data'} for {experiment_name} saved to {data_base}/{fname}/{out_name}"
    )
    logger.info(f"{period_type} data processing completed successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset zarrs and stats.")
    # parser.add_argument(
    #     "--highres-vars",
    #     choices=channel_vars["HighRes"],
    #     default="t2m",
    #     help="HighRes channel variables (comma-separated). Default: t2m",
    # )
    # parser.add_argument(
    #     "--lowres-vars",
    #     choices=channel_vars["LowRes"],
    #     default="t2m",
    #     help="LowRes channel variables (comma-separated). Default: t2m",
    # )
    parser.add_argument(
        "--train-start",
        default=train_datetime_start,
        help=f"Training start date YYYY/MM/DD (default {train_datetime_start})",
    )
    parser.add_argument(
        "--train-end",
        default=train_datetime_last,
        help=f"Training end date YYYY/MM/DD (default {train_datetime_last})",
    )
    parser.add_argument(
        "--val-start",
        default=valid_datetime_start,
        help=f"Validation start date YYYY/MM/DD (default {valid_datetime_start})",
    )
    parser.add_argument(
        "--val-end",
        default=valid_datetime_last,
        help=f"Validation end date YYYY/MM/DD (default {valid_datetime_last})",
    )
    parser.add_argument(
        "--cache-base",
        default=cache_base,
        help=f"Cache base path (default {cache_base})",
    )
    parser.add_argument(
        "--data-base", default=data_base, help=f"Data base path (default {data_base})"
    )
    parser.add_argument(
        "--process-invariants",
        action="store_true",
        help="Process invariant data (land-sea mask, orography)",
    )

    args = parser.parse_args()

    logger.info(f"train-start: {args.train_start}")
    logger.info(f"train-end: {args.train_end}")

    channel_vars_dict = channel_vars

    # Process invariants if requested
    # if args.process_invariants:
    process_invariants(
         cache_base=args.cache_base,
         data_base=args.data_base,
         experiment_name=experiment_name,
         start_date=args.train_start,
     )

    # process training period
    for fname in ["HighRes", "LowRes"]:
        process_period(
            start_date=args.train_start,
            end_date=args.train_end,
            fname=fname,
            channel_vars_dict=channel_vars_dict,
            cache_base=args.cache_base,
            data_base=args.data_base,
            domain_size=domain_size,
            experiment_name=experiment_name,
            is_validation=False,
        )

    # process validation period
    for fname in ["HighRes", "LowRes"]:
        process_period(
            start_date=args.val_start,
            end_date=args.val_end,
            fname=fname,
            channel_vars_dict=channel_vars_dict,
            cache_base=args.cache_base,
            data_base=args.data_base,
            domain_size=domain_size,
            experiment_name=experiment_name,
            is_validation=True,
        )


if __name__ == "__main__":
    main()

