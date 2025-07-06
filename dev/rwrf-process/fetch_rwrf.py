from utils.config import CONFIG
from netCDF4 import Dataset
from datetime import datetime
import os
import numpy as np
import argparse

var_map = {
    "t2m": "T2",
    "u10": "umet10",
    # add any others here...
}


def load_wrf_interp_nc(date_str: str, hr_str: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    fmt_dt_str = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
    folder = CONFIG.rwrf
    filepath = f"{folder}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ds = Dataset(filepath, mode="r")
    return ds


def save_t2m_numpy(
    date_str: str, hr_str: str, variable: str, out_dir: str = "./cache/rwrf/"
):
    # 1) load dataset
    ds = load_wrf_interp_nc(date_str, hr_str)

    # 2) grab the raw arrays
    data = ds.variables[var_map.get(variable, variable)][
        :
    ]  # often (time, grid_row, grid_col) => (1, 450, 450), every grid point represents a value, e.g. data[0, 13, 320] is the value at grid point (13, 320)
    lat = ds.variables["XLAT"][
        :
    ]  # (time, grid_row, grid_col) => (1, 450, 450), every grid point represents a latitude, e.g. lat[0, 13, 320] is the latitude of grid point (13, 320)
    lon = ds.variables["XLONG"][
        :
    ]  # (time, grid_row, grid_col) => (1, 450, 450), every grid point represents a longitude, e.g. lon[0, 13, 320] is the longitude of grid point (13, 320)
    times = ds.variables["Times"][:]  # WRF Times: char array, shape (1, 19)

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
        description="Extract RWRF data and save as numpy arrays."
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
