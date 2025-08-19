import argparse
import os
from datetime import datetime
from typing import Optional

import numpy as np
from netCDF4 import Dataset
from utils.config import CONFIG

var_map = {
    "t2m": "T2",
    "u10": "umet10",
    "pptn": "qpepre",
    "lsm": "LANDMASK",
    "orog": "HGT",
    # add any others here...
}


def load_wrf_interp_nc(
    date_str: str,
    hr_str: str,
    variable: Optional[str] = None,
    cropped_qpepre: bool = False,
) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    fmt_dt_str = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
    folder = CONFIG.rwrf
    if variable == "pptn":
        if cropped_qpepre:
            filepath = f"{folder}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp_cropped_qpepre.nc"
        else:
            filepath = f"{folder}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp_qpepre.nc"
    else:
        filepath = f"{folder}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp"
    if not os.path.exists(filepath):
        # raise FileNotFoundError(f"File not found: {filepath}") # use this after all files are downloaded
        print(f"WARNING: File not found: {filepath}, skipping...")  # use this for now
    ds = Dataset(filepath, mode="r")
    return ds


def save_t2m_numpy(
    date_str: str,
    hr_str: str,
    variable: str,
    cropped_qpepre: bool = False,
    out_dir: str = "../data/cache/rwrf/train/",
):
    # 1) load dataset
    ds = load_wrf_interp_nc(date_str, hr_str, variable, cropped_qpepre)

    # 2) grab the raw arrays
    data = ds.variables[var_map.get(variable, variable)][:]
    lat = ds.variables["XLAT"][:]  # often shape (time, y, x)
    lon = ds.variables["XLONG"][:]
    times = ds.variables["Times"][:]  # WRF Times: char array

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
        description="Extract RWRF data and save as numpy arrays."
    )
    parser.add_argument(
        "--variable",
        choices=[var for var in var_map.keys()],
        required=True,
        help="Variable to extract:",
    )
    parser.add_argument(
        "--cropped-qpepre",
        action="store_true",
        help="Use cropped RWRF data by QPEPRE",
    )
    args = parser.parse_args()
    for date_str in CONFIG.date_strs:
        for hr_str in CONFIG.hr_strs:
            print(
                f"Transforming RWRF {date_str.replace('/', '')}{hr_str}.nc {args.variable} to numpy array..."
            )
            save_t2m_numpy(date_str, hr_str, args.variable)


if __name__ == "__main__":
    main()
