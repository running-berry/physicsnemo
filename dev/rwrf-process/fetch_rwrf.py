from netCDF4 import Dataset
from datetime import datetime
import os
import numpy as np

def load_wrf_interp_nc(date_str: str, hr_str: str) -> Dataset:
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    folder = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
    filepath = f"./data/rwrf/{folder}/wrfout_d01_{folder}_interp"
    ds = Dataset(filepath, mode='r')
    return ds

def save_t2m_numpy(date_str: str, hr_str: str, out_dir: str = "./cache/rwrf/"):
    # 1) load dataset
    ds = load_wrf_interp_nc(date_str, hr_str)

    # 2) grab the raw arrays
    t2m= ds.variables["T2"][:]       # shape (time, y, x)
    lat = ds.variables["XLAT"][:]    # often shape (time, y, x)
    lon = ds.variables["XLONG"][:]
    times = ds.variables['Times'][:]    # WRF Times: char array


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
    save_t2m_numpy("2019/08/03", "00")