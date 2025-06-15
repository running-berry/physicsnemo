from netCDF4 import Dataset
from datetime import datetime
import os
import numpy as np
import argparse


def load_wrf_interp_nc(date_str: str, hr_str: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    folder = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
    # filepath = f"./data/rwrf/{folder}/wrfout_d01_{folder}_interp"
    filepath = f"../data/rwrf/{folder}/wrfout_d01_{folder}_interp"
    ds = Dataset(filepath, mode='r')
    return ds

def save_t2m_numpy(date_str: str, hr_str: str, variable: str, out_dir: str = "./cache/rwrf/"):
    # 1) load dataset
    ds = load_wrf_interp_nc(date_str, hr_str)
    print("Variables in the dataset:", ds.variables.keys())
    # for var_name in ds.variables:
    #     var = ds.variables[var_name]
    #     print(f"{var_name}: shape {var.shape}")

    # 2) grab the raw arrays
    if variable == "t2m":
        t2m = ds.variables["T2"][:]       # shape (time, y, x)
    elif variable == "u10":
        u10 = ds.variables["umet10"][:]       # shape (time, y, x)

    lat = ds.variables["XLAT"][:]    # often shape (time, y, x)
    lon = ds.variables["XLONG"][:]
    times = ds.variables['Times'][:]    # WRF Times: char array


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
    parser = argparse.ArgumentParser(description='Extract RWRF data and save as numpy arrays.')
    parser.add_argument('--variable', choices=['t2m', 'u10'], required=True,
                        help='Variable to extract:')
    args = parser.parse_args()
    save_t2m_numpy("2019/08/03", "00", args.variable)


if __name__ == "__main__":
    main()