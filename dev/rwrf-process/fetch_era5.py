from netCDF4 import Dataset
from datetime import datetime
import os
import numpy as np

def load_era5_interp_nc(date_str: str, hr_str: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    folder = "./data/era5/train"
    filepath = dt.strftime(f"{folder}/t2m_%Y%m%d_") + hr_str.zfill(2) + ".nc"
    ds = Dataset(filepath, mode='r')

    return ds

def save_t2m_numpy(date_str: str, hr_str: str, out_dir: str = "./cache/era5/"):
    # 1) load dataset
    ds = load_era5_interp_nc(date_str, hr_str)
    print("Variables in the dataset:", ds.variables.keys())
    # 2) grab the raw arrays
    #t2m = ds.variables["variable"][:]       # shape (time, y, x)
    lat = ds.variables["lat"][:]    # often shape (time, y, x)
    lon = ds.variables["lon"][:]
    times = ds.variables['time'][:]    # WRF Times: char array
    t2m = ds.variables['__xarray_dataarray_variable__'][:]    # WRF Times: char array

    ds.close()
    # 3) ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # 4) save as .npz (multiple arrays in one file)
    fn = date_str.replace("/", "") + f"_{hr_str}.npz"
    out_path = os.path.join(out_dir, fn)
    np.savez(
        out_path,
        t2m=t2m,
        lat=lat,
        lon=lon,
        times=times
    )
    print(f"Saved arrays to {out_path}")

if __name__ == "__main__":
    save_t2m_numpy("2020/06/01", "00")