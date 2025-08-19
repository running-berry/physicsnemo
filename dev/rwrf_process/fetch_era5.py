import argparse
import os
from datetime import datetime

import numpy as np
from netCDF4 import Dataset
from utils.config import CONFIG

var_map = {
    "t2m": "t2m",
    "u10": "u10",
    "pptn": "tp",
    "lsm": "lsm",
    "orog": "orog",
    # add any others here...
}


def load_era5_interp_nc(date_str: str, hr_str: str, variable: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    folder = CONFIG.era5
    if variable == "pptn":
        filepath = dt.strftime(f"{folder}/tp_%Y%m%d") + hr_str.zfill(2) + ".nc"
    else:
        filepath = dt.strftime(f"{folder}/{variable}_%Y%m%d") + hr_str.zfill(2) + ".nc"
    if not os.path.exists(filepath):
        # raise FileNotFoundError(f"File not found: {filepath}") # use this after all files are downloaded
        print(f"WARNING: File not found: {filepath}, skipping...")  # use this for now
    ds = Dataset(filepath, mode="r")

    return ds


def save_t2m_numpy(
    date_str: str,
    hr_str: str,
    variable: str,
    out_dir: str = "../data/cache/era5/train/",
):
    # 1) load dataset
    ds = load_era5_interp_nc(date_str, hr_str, variable)

    # 2) grab the raw arrays
    for key in ["latitude", "longitude", "valid_time", var_map.get(variable, variable)]:
        if key not in ds.variables:
            raise KeyError(
                f"Key '{key}' not found in dataset variables: {list(ds.variables.keys())}"
            )

    lat = ds.variables["latitude"][:]  # (721, )
    lon = ds.variables["longitude"][:]  # (1440, )
    times = ds.variables["valid_time"][:]  # unix format (1,)
    data = ds.variables[var_map.get(variable, variable)][
        :
    ]  # often shape (1, 721, 1440)

    ds.close()
    # 3) ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # 4) save as .npz (multiple arrays in one file)
    if variable == "pptn":
        variable = "qpepre"
    fn = f"{variable}_" + date_str.replace("/", "") + f"{hr_str}.npz"
    out_path = os.path.join(out_dir, fn)
    np.savez(out_path, **{variable: data}, lat=lat, lon=lon, times=times)
    print(f"Saved arrays to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5 data and save as numpy arrays."
    )
    parser.add_argument(
        "--variable",
        choices=[var for var in var_map.keys()],
        required=True,
        help="Variable to extract:",
    )
    args = parser.parse_args()
    for date_str in CONFIG.date_strs:
        for hr_str in CONFIG.hr_strs:
            print(
                f"Transforming ERA5 {date_str.replace('/', '')}{hr_str}.nc {args.variable} to numpy array..."
            )
            save_t2m_numpy(date_str, hr_str, args.variable)


if __name__ == "__main__":
    main()
