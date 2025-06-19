from utils.config import CONFIG
from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
import numpy as np
import regex as re


def get_filename_from_date(date_str: str, hr_str: str) -> str:
    """
    Generate a filename based on the date and hour.
    """
    
    dt_start = datetime.strptime(f"{date_str} {hr_str}", "%Y/%m/%d %H")
    dt_end = dt_start + timedelta(hours=1)

    start_str = dt_start.strftime("%Y%m%d%H%M")
    end_str = dt_end.strftime("%Y%m%d%H%M")

    folder = CONFIG.pptn
    filepath = f"{folder}/qpepre_{start_str}-{end_str}_1_h"
    return filepath


def extract_start_str(txt_path: str) -> str:
    """
    Extracts a start datetime string from the file name.
    """
    match = re.search(r'(\d{12})-\d{12}', txt_path)
    if not match:
        raise ValueError(f"Filename does not match pattern: {txt_path}")
    start_dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    return start_dt.strftime("%Y-%m-%d_%H:%M:%S")


def convert_txt_to_nc(date_str: str, hr_str: str, var_name='qpepre'):
    """
    Convert a text file with lat, lon, and data columns to a NetCDF file.
    """
    
    filename = get_filename_from_date(date_str, hr_str)
    nc_path = filename + ".nc"
    txt_path = filename + ".txt"
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file {txt_path} does not exist.")

    data = np.loadtxt(txt_path)
    start_str = extract_start_str(txt_path)
    
    lon = data[:, 1]
    lat = data[:, 2]
    values = data[:, 3]

    lat_unique = np.unique(lat)
    lon_unique = np.unique(lon)
    n_lat = len(lat_unique)
    n_lon = len(lon_unique)

    val_grid = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)
    lat_grid = np.full((1, n_lat), np.nan, dtype=np.float32)
    lon_grid = np.full((1, n_lon), np.nan, dtype=np.float32)
    lat_to_idx = {lat: i for i, lat in enumerate(lat_unique)}
    lon_to_idx = {lon: j for j, lon in enumerate(lon_unique)}

    for i in range(data.shape[0]):
        _lon, _lat, _val = data[i, 1], data[i, 2], data[i, 3]
        lat_idx = lat_to_idx[_lat]
        lon_idx = lon_to_idx[_lon]
        val_grid[0, lat_idx, lon_idx] = _val

    lat_grid[0, :] = lat_unique
    lon_grid[0, :] = lon_unique

    print(f"Creating NetCDF file: {nc_path}")
    with Dataset(nc_path, 'w', format='NETCDF4') as nc_file:
        nc_file.createDimension('times', 1)
        nc_file.createDimension('date_strlen', len(start_str))
        nc_file.createDimension('lat', n_lat)
        nc_file.createDimension('lon', n_lon)

        times = nc_file.createVariable('times', 'S1', ('times', 'date_strlen'))
        latitudes = nc_file.createVariable('lat', np.float32, ('times', 'lat'))
        longitudes = nc_file.createVariable('lon', np.float32, ('times', 'lon'))
        values = nc_file.createVariable(var_name, np.float32, ('times', 'lat', 'lon'))

        times[:, :] = np.array([list(start_str)], dtype='S1')
        latitudes[:, :] = lat_grid
        longitudes[:, :] = lon_grid
        values[:, :, :] = val_grid

    print(f"Converted {txt_path} to {nc_path}")


def load_pptn_interp_nc(date_str: str, hr_str: str) -> Dataset:
    filepath = get_filename_from_date(date_str, hr_str) + ".nc"
    ds = Dataset(filepath, mode='r')
    return ds


def save_t2m_numpy(date_str: str, hr_str: str, variable = "qpepre", out_dir: str = "./cache/pptn/"):
    convert_txt_to_nc(date_str, hr_str)
    
    # 1) load dataset
    ds = load_pptn_interp_nc(date_str, hr_str)
    print("Variables in the dataset:", ds.variables.keys())
    # for var_name in ds.variables:
    #     var = ds.variables[var_name]
    #     print(f"{var_name}: shape {var.shape}")

    # 2) grab the raw arrays
    data = ds.variables[variable][:]
    lat = ds.variables["lat"][:]    # often shape (time, y, x)
    lon = ds.variables["lon"][:]
    times = ds.variables['times'][:]    # WRF Times: char array

    ds.close()

    # 3) ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # 4) save as .npz (multiple arrays in one file)
    fn = f"{variable}_" + date_str.replace("/", "") + f"_{hr_str}.npz"
    out_path = os.path.join(out_dir, fn)
    np.savez(
        out_path,
        **{variable: data},
        lat=lat,
        lon=lon,
        times=times
    )
    print(f"Saved arrays to {out_path}")
    
def main():
    save_t2m_numpy("2019/08/13", "00")


if __name__ == "__main__":
    main()