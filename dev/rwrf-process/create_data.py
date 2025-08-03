import os

import numpy as np
import util_extract as u1
import xarray as xr
import yaml
import zarr

with open("../../examples/generative/stormcast/config/dataset/small.yaml", "r") as f:
    cfg = yaml.safe_load(f)

channel_vars = ["t2m", "u10"]
num_channel = len(channel_vars)
domain_size = tuple(cfg["HighRes_img_size"])
test_datetime_start = cfg["train_dates"][0]
test_datetime_last = cfg["train_dates"][-1]
cache_base = "./cache"
data_base = "../data"
experiment_name = cfg["exp_train_zarrs"][0]


def create_dummy_arr(
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
    # build the filename for this dt
    yy, mm, dd, hh = np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
    fn = f"{data_var}_{yy}{mm}{dd}_{hh}.npz"
    dt_path = os.path.join(data_path, fn)

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
        real_arr, lon_grid, lat_grid = u1.interp_to_domain(
            lon_grid, lat_grid, real_arr, domain_size, method="linear"
        )

    dummy_arr = np.full_like(
        real_arr, fill_value=0.0
    )  # future: or use fill_value=np.nan

    return dummy_arr, lon_grid, lat_grid, times


for fname in ["HighRes", "LowRes"]:
    folder_path = f"{data_base}/{fname}/stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print(f"'{folder_path}' exists, skipping stats folder generation.")

    # determine data path base
    if fname == "HighRes":
        cache_path = f"{cache_base}/rwrf/"
    elif fname == "LowRes":
        cache_path = f"{cache_base}/era5/"

    lon_min, lon_max = 121.00, 125.00
    lat_min, lat_max = 21.00, 25.00

    base_date = np.datetime64(test_datetime_start.replace("/", "-") + "T00:00:00")
    end_date = np.datetime64(test_datetime_last.replace("/", "-")) + np.timedelta64(
        23, "h"
    )
    total_hours = int((end_date - base_date) / np.timedelta64(1, "h")) + 1
    offsets = np.arange(total_hours, dtype=np.int64)
    datetime_array = base_date + offsets * np.timedelta64(1, "h")
    print(cache_path)
    # create the dummy data format by the first data
    dummy_data, dummy_lon_grid, dummy_lat_grid, dummy_times = create_dummy_arr(
        datetime_array[0],
        cache_path,
        channel_vars[0],
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )
    missing_data = []
    data_arr = None
    for dt in datetime_array:
        channel_arr = None
        for var in channel_vars:
            yy, mm, dd, hh = (
                np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
            )
            dt_path = os.path.join(cache_path, f"{var}_{yy}{mm}{dd}_{hh}.npz")
            print(f"Processing {dt_path}")

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

            except FileNotFoundError:
                # create dummy data
                dt_data = dummy_data.copy()
                missing_data.append(f"{yy}-{mm}-{dd} {hh}:00")

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

    # compute mean and std over time, latitude & longitude → leaves (n_chan,)
    # data_arr shape is (n_time, n_chan, ny, nx)

    if data_arr.ndim == 4:
        # mean/std over time, y, x → result per level
        means = np.nanmean(data_arr, axis=(0, 2, 3)).astype(np.float32)
        stds = np.nanstd(data_arr, axis=(0, 2, 3)).astype(np.float32)
    elif data_arr.ndim == 3:
        # mean/std over all values → wrap in length-1 array
        m = np.nanmean(data_arr).astype(np.float32)
        s = np.nanstd(data_arr).astype(np.float32)
        means = np.array([m], dtype=np.float32)
        stds = np.array([s], dtype=np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D array, got {data_arr.ndim}D.")

    # ensure float32 precision
    means = means.astype(np.float32)
    stds = stds.astype(np.float32)
    print(f"Check: {means}, {stds}")

    # save them
    np.save(f"{folder_path}/means.npy", means)
    np.save(f"{folder_path}/stds.npy", stds)

    data_shape = (total_hours, num_channel) + domain_size
    print(data_arr.shape)
    year_data = xr.Dataset(
        {
            f"{fname}": (["time", "channel", "y", "x"], data_arr),
            "time": datetime_array,
            "channel": channel_vars,
            "latitude": (["y", "x"], lat_grid),
            "longitude": (["y", "x"], lon_grid),
        }
    )
    data_enc = {f"{fname}": {"dtype": "float32", "compressor": None}}
    year_data.to_zarr(
        f"{data_base}/{fname}/{experiment_name}.zarr",
        mode="w",
        consolidated=True,
        encoding=data_enc,
        zarr_format=2,
    )
    zarr.consolidate_metadata(f"{data_base}/{fname}/{experiment_name}.zarr")

    print(
        f"Data for {experiment_name} saved to {data_base}/{fname}/{experiment_name}.zarr"
    )
