from netCDF4 import Dataset
from datetime import datetime
import os
import numpy as np
import argparse

def load_era5_interp_nc(date_str: str, hr_str: str, variable: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    # folder = "./data/era5/train"
    folder = "../data/era5/train"
    if variable == "t2m":
        filepath = dt.strftime(f"{folder}/t2m_%Y%m%d_") + hr_str.zfill(2) + ".nc"
    elif variable == "u10":
        filepath = dt.strftime(f"{folder}/u10_%Y%m%d_") + hr_str.zfill(2) + ".nc"
    ds = Dataset(filepath, mode='r')

    return ds

def save_t2m_numpy(date_str: str, hr_str: str, variable: str, out_dir: str = "./cache/era5/"):
    # 1) load dataset
    ds = load_era5_interp_nc(date_str, hr_str, variable)
    print("Variables in the dataset:", ds.variables.keys())
    # for var_name in ds.variables:
    #     var = ds.variables[var_name]
    #     print(f"{var_name}: shape {var.shape}")
    
    # 2) grab the raw arrays
    if variable == "t2m":
        lat = ds.variables["lat"][:]    # often shape (time, y, x)
        lon = ds.variables["lon"][:]
        times = ds.variables["time"][:]  # WRF Times: char array
        t2m = ds.variables['__xarray_dataarray_variable__'][:]
    elif variable == "u10":
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]
        times = ds.variables['valid_time'][:]   # unix format
        u10 = ds.variables["u10"][:]    # often shape (time, y, x)

    ds.close()
    # 3) ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # 4) save as .npz (multiple arrays in one file)
    fn = f"{variable}_" + date_str.replace("/", "") + f"_{hr_str}.npz"
    out_path = os.path.join(out_dir, fn)
    if variable == "t2m":
        np.savez(
            out_path,
            t2m=t2m,
            lat=lat,
            lon=lon,
            times=times
        )
    elif variable == "u10":
        np.savez(
            out_path,
            u10=u10,
            lat=lat,
            lon=lon,
            times=times
        )
    print(f"Saved arrays to {out_path}")
    
def main():
    parser = argparse.ArgumentParser(description='Extract ERA5 data and save as numpy arrays.')
    parser.add_argument('--variable', choices=['t2m', 'u10'], required=True,
                        help='Variable to extract:')
    args = parser.parse_args()
    save_t2m_numpy("2020/06/01", "00", args.variable)


if __name__ == "__main__":
    main()