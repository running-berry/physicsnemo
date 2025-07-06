from utils.config import CONFIG
from netCDF4 import Dataset
from datetime import datetime
import os
import numpy as np
import argparse


def load_era5_interp_nc(date_str: str, hr_str: str, variable: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    folder = CONFIG.era5
    if variable == "t2m":
        filepath = dt.strftime(f"{folder}/t2m_%Y%m%d_") + hr_str.zfill(2) + ".nc"
    elif variable == "u10":
        filepath = dt.strftime(f"{folder}/u10_%Y%m%d_") + hr_str.zfill(2) + ".nc"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ds = Dataset(filepath, mode="r")

    return ds


def save_t2m_numpy(
    date_str: str, hr_str: str, variable: str, out_dir: str = "./cache/era5/"
):
    # 1) load dataset
    ds = load_era5_interp_nc(date_str, hr_str, variable)

    # 2) grab the raw arrays
    if variable == "t2m":
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        times = ds.variables["time"][:]  # WRF Times: char array
        data = ds.variables["__xarray_dataarray_variable__"][:]
    elif variable == "u10":
        lat = ds.variables["latitude"][:]  # (num_latitude,) => (721, )
        lon = ds.variables["longitude"][:]  # (num_longitude,) => (1440, )
        times = ds.variables["valid_time"][:]  # unix format => (1,)
        data = ds.variables["u10"][
            :
        ]  # (time, num_vars, num_latitude, num_longitude) => (1, 1, 721, 1440), every grid point represents a value, e.g. data[0, 0, 13, 320] is the value at grid point (13, 320), lat[13] is the latitude of grid point (13, 320), lon[320] is the longitude of grid point (13, 320)
    ds.close()
    # 3) ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # 4) save as .npz (multiple arrays in one file)
    fn = f"{variable}_" + date_str.replace("/", "") + f"_{hr_str}.npz"
    out_path = os.path.join(out_dir, fn)
    np.savez(out_path, **{variable: data}, lat=lat, lon=lon, times=times)
    print(f"Saved arrays to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5 data and save as numpy arrays."
    )
    parser.add_argument(
        "--variable", choices=["t2m", "u10"], required=True, help="Variable to extract:"
    )
    args = parser.parse_args()
    for hr in range(0, 24):
        hr_str = str(hr).zfill(2)
        print("Processing 2019/08/03", hr_str, args.variable)
        save_t2m_numpy("2019/08/03", hr_str, args.variable)


if __name__ == "__main__":
    main()
