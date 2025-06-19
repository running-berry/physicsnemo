from utils.config import CONFIG
import shutil
from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
import numpy as np
import regex as re
from scipy.interpolate import griddata


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
    match = re.search(r"(\d{12})-\d{12}", txt_path)
    if not match:
        raise ValueError(f"Filename does not match pattern: {txt_path}")
    start_dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    return start_dt.strftime("%Y-%m-%d_%H:%M:%S")


def convert_txt_to_nc(date_str: str, hr_str: str, var_name="qpepre"):
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
    ds = Dataset(nc_path, "w", format="NETCDF4")
    with ds as nc_file:
        nc_file.createDimension("times", 1)
        nc_file.createDimension("date_strlen", len(start_str))
        nc_file.createDimension("lat", n_lat)
        nc_file.createDimension("lon", n_lon)

        times = nc_file.createVariable("times", "S1", ("times", "date_strlen"))
        latitudes = nc_file.createVariable("lat", "f4", ("times", "lat"))
        longitudes = nc_file.createVariable("lon", "f4", ("times", "lon"))
        values = nc_file.createVariable(var_name, "f4", ("times", "lat", "lon"))

        times[:, :] = np.array([list(start_str)], dtype="S1")
        latitudes[:, :] = lat_grid
        longitudes[:, :] = lon_grid
        values[:, :, :] = val_grid

    print(f"Converted {txt_path} to {nc_path}")


def combine_rwrf_qpepre(rwrf: Dataset, qpepre: Dataset):
    """
    Combine RWRF and QPEPRE datasets into a single xarray Dataset.
    Interpolates QPEPRE onto the RWRF grid.
    """
    rwrf_lat2d = rwrf["XLAT"][0, :]
    rwrf_lon2d = rwrf["XLONG"][0, :]

    qpepre_lat1d = qpepre["lat"][0, :]
    qpepre_lon1d = qpepre["lon"][0, :]
    qpepre_var2d = qpepre["qpepre"][0, :, :]

    qpepre_lon2d, qpepre_lat2d = np.meshgrid(qpepre_lon1d, qpepre_lat1d)
    points = np.column_stack((qpepre_lat2d.ravel(), qpepre_lon2d.ravel()))
    values = qpepre_var2d.ravel()
    target_points = np.column_stack((rwrf_lat2d.ravel(), rwrf_lon2d.ravel()))

    interpolated = griddata(
        points, values, target_points, method="linear", fill_value=np.nan
    )
    interpolated_2d = interpolated.reshape(rwrf_lat2d.shape)
    var = rwrf.createVariable("qpepre", "f4", rwrf["XLAT"].dimensions)
    var[0, :, :] = interpolated_2d


def load_pptn_interp_nc(date_str: str, hr_str: str) -> Dataset:
    filepath = get_filename_from_date(date_str, hr_str) + ".nc"
    ds = Dataset(filepath, mode="r")
    return ds


def copy_rwrf(date_str: str, hr_str: str) -> str:
    """
    Copy RWRF .nc file from original path to new path.
    """
    dt = datetime.strptime(date_str, "%Y/%m/%d")
    fmt_dt_str = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
    folder = CONFIG.rwrf
    org_path = f"{folder}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp"
    new_path = f"{folder}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp_qpepre.nc"

    shutil.copy(org_path, new_path)
    return new_path


def crop_rwrf_by_qpepre(src_path: str) -> str:
    cropped_rwrf_path = src_path.replace("qpepre.nc", "cropped_qpepre.nc")

    with Dataset(src_path, "r") as src, Dataset(cropped_rwrf_path, "w") as dst:
        qpepre = src.variables["qpepre"][0, :, :]
        mask = ~np.isnan(qpepre)
        if not np.any(mask):
            os.remove(cropped_rwrf_path)
            raise ValueError("No non-NaN values found in qpepre.")

        coords = np.argwhere(mask)
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)

        # Copy dimensions (keep all the same except for spatial)
        for name, dim in src.dimensions.items():
            ny = max_row - min_row + 1
            nx = max_col - min_col + 1
            if name == "south_north":
                dst.createDimension(name, ny)
            elif name == "west_east":
                dst.createDimension(name, nx)
            else:
                dst.createDimension(name, len(dim))

        # Copy global attributes
        for attr in src.ncattrs():
            dst.setncattr(attr, src.getncattr(attr))

        for name, var in src.variables.items():
            if var.shape[-2:] == (
                src.dimensions["south_north"].size,
                src.dimensions["west_east"].size,
            ):
                newvar = dst.createVariable(name, var.datatype, var.dimensions)
                for attr in var.ncattrs():
                    newvar.setncattr(attr, var.getncattr(attr))

                newvar[...] = var[..., min_row : max_row + 1, min_col : max_col + 1]
            else:
                newvar = dst.createVariable(name, var.datatype, var.dimensions)
                newvar[:] = var[:]
                for attr in var.ncattrs():
                    newvar.setncattr(attr, var.getncattr(attr))

    print(f"Cropped RWRF file saved to: {cropped_rwrf_path}")
    return cropped_rwrf_path


def store_rwrf_qpepre_dataset(date_str: str, hr_str: str):
    convert_txt_to_nc(date_str, hr_str)

    ds = load_pptn_interp_nc(date_str, hr_str)
    new_rwrf_path = copy_rwrf(date_str, hr_str)
    rwrf_ds = Dataset(new_rwrf_path, mode="a")
    combine_rwrf_qpepre(rwrf_ds, ds)

    print("Variables in the pptn dataset:", ds.variables.keys())
    # for var_name in ds.variables:
    #     var = ds.variables[var_name]
    #     print(f"{var_name}: shape {var.shape}")
    ds.close()

    print("Variables in the rwrf_qpepre dataset:", rwrf_ds.variables.keys())
    # for var_name in rwrf_ds.variables:
    #     var = rwrf_ds.variables[var_name]
    #     print(
    #         f"{var_name}: shape {var.shape} dtype {var.dtype} attributes {var.ncattrs()} dimensions {var.dimensions}"
    #     )

    # for name, dim in rwrf_ds.dimensions.items():
    #     print(f"Dimension {name}: size {len(dim)}")

    # for attr in rwrf_ds.ncattrs():
    #     print(f"Global attribute {attr}: {rwrf_ds.getncattr(attr)}")

    rwrf_ds.close()
    cropped_rwrf_path = crop_rwrf_by_qpepre(new_rwrf_path)


def main():
    store_rwrf_qpepre_dataset("2019/08/03", "00")


if __name__ == "__main__":
    main()
