import os
import xarray as xr
import numpy as np

channel_vars = ["t2m"]
num_channel = len(channel_vars)
test_datetime_start = "2019/08/03"
test_datetime_last = "2019/08/03"
test_years = [2019]
cache_base = "./cache"
data_base = "../data"


def load_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    dt_data = data[channel_vars[0]]  # (t,y,x) or (t,lev,y,x)
    lat_grid = data["lat"]  # era5: (1, Nlat), rwrf: (1, Nlat, Nlon)
    lat_grid = lat_grid.squeeze()  # ensure shape is (Nlat,) or (Nlat, Nlon)
    lon_grid = data["lon"]  # era5: (1, Nlat), rwrf: (1, Nlat, Nlon)
    lon_grid = lon_grid.squeeze()  # ensure shape is (Nlon,) or (Nlat, Nlon)
    times = data["times"]

    return dt_data, lon_grid, lat_grid, times


def create_dummy_arr(
    dt,
    data_path: str,
    data_var: str,
):
    yy, mm, dd, hh = np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
    fn = f"{data_var}_{yy}{mm}{dd}_{hh}.npz"
    dt_path = os.path.join(data_path, fn)

    data, lon_grid, lat_grid, times = load_data(dt_path)

    dummy_arr = np.full_like(
        data, fill_value=0.0, dtype=np.float32
    )  # future: or use fill_value=np.nan

    return dummy_arr, lon_grid, lat_grid, times


for fname in ["HighRes", "LowRes"]:
    folder_path = f"{data_base}/{fname}/stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print(f"'{folder_path}' is existed")

    # determine data path base
    if fname == "HighRes":
        cache_path = f"{cache_base}/rwrf/"
        import fetch_rwrf as u2
    elif fname == "LowRes":
        cache_path = f"{cache_base}/era5/"
        import fetch_era5 as u2

    year = test_years[0]
    base_date = np.datetime64(test_datetime_start.replace("/", "-") + "T00:00:00")
    end_date = np.datetime64(test_datetime_last.replace("/", "-")) + np.timedelta64(
        23, "h"
    )
    total_hours = int((end_date - base_date) / np.timedelta64(1, "h")) + 1
    offsets = np.arange(total_hours, dtype=np.int64)
    datetime_array = base_date + offsets * np.timedelta64(1, "h")
    print(cache_path)
    # create the dummy data format by the first data
    dummy_data, lon_grid, lat_grid, times = create_dummy_arr(
        datetime_array[0],
        cache_path,
        channel_vars[0],
    )
    data_arr = None
    channel_var = channel_vars[0]

    for dt in datetime_array:
        yy, mm, dd, hh = (
            np.datetime_as_string(dt, unit="h").replace("T", "-").split("-")
        )
        dt_path = os.path.join(cache_path, f"{channel_var}_{yy}{mm}{dd}_{hh}.npz")
        print(f"Processing {dt_path}")

        try:
            dt_data, lon_grid, lat_grid, times = load_data(dt_path)

        except FileNotFoundError:
            # create dummy data
            dt_data = dummy_data.copy()
            print(
                "File not found, missing data for",
                f"{yy}-{mm}-{dd} {hh}:00",
                "using dummy data",
            )

        # concatenate data
        if data_arr is None:
            data_arr = dt_data.copy()
        else:
            # concatenate along axis=0 (time or channel, whichever you're stacking)
            data_arr = np.concatenate((data_arr, dt_data), axis=0)

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

    print(data_arr.shape)
    if data_arr.ndim == 3:
        # make it (time, 1, y, x)
        data_arr = data_arr[:, None, :, :]

    print(data_arr.shape)
    if fname == "LowRes":
        year_data = xr.Dataset(
            {
                f"{fname}": (["time", "channel", "y", "x"], data_arr),
                "time": datetime_array,
                "channel": channel_vars,
                "latitude": lat_grid,
                "longitude": lon_grid,
            }
        )
    elif fname == "HighRes":
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
        f"{data_base}/{fname}/{year}.zarr",
        mode="w",
        consolidated=True,
        encoding=data_enc,
    )

    print(f"Data for {year} saved to {data_base}/{fname}/{year}.zarr")
