import logging
import os
import pathlib
from datetime import datetime, timedelta

import numpy as np
import regex as re
from netCDF4 import Dataset
from scipy.interpolate import griddata
from utils import CONFIG

logger = logging.getLogger(__name__)


class RWRFQPEPREProcessor:
    """
    Processes and interpolates/combines RWRF data with QPEPRE data.

    Parameters
    ----------
    output_dir : str
        The directory where the output files will be saved.
    qpepre_src : str
        The directory where the QPEPRE input files are located.
    rwrf_src : str
        The directory where the RWRF input files are located.
    """

    def __init__(
        self,
        output_dir: str,
        qpepre_src: str = CONFIG.qpepre,
        rwrf_src: str = CONFIG.rwrf,
    ):
        self.qpepre_src = qpepre_src
        self.rwrf_src = rwrf_src
        self.output_dir = output_dir
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)  # redundant?

    def __call__(self):
        """Interpolates all QPEPRE files in the source directory into corresponding RWRF datasets."""
        src_path = pathlib.Path(self.qpepre_src)
        qpepre_files = list(src_path.rglob("*.txt"))
        for file in qpepre_files:
            basename = file.stem
            date = basename.split("_")[1][:-2]
            date_str = f"{date[:4]}/{date[4:6]}/{date[6:8]}"
            hr_str = date[-2:]
            if self._check_exists(date_str, hr_str):
                logger.info(
                    f"RWRF QPEPRE dataset already exists for {date_str} {hr_str}. Skipping."
                )
                continue
            self._process(date_str, hr_str)

    def process_files_from_date_list(
        self, date_strs: list[str], hr_strs: list[str]
    ) -> None:
        """
        Executes the processing workflow for a given list of dates and hours.

        Parameters
        ----------
        date_strs : list[str]
            A list of date strings in 'YYYY/MM/DD' format.
        hr_strs : list[str]
            A list of hour strings in 'HH' format (e.g., '00', '12', '23').
        """
        for date_str in date_strs:
            for hr_str in hr_strs:
                if self._check_exists(date_str, hr_str):
                    logger.info(
                        f"Converted file already exists for QPEPRE {date_str} {hr_str}, skipping conversion."
                    )
                    continue

                logger.info(
                    f"Queueing RWRF-QPEPRE processing for {date_str} {hr_str}..."
                )
                self._process(date_str, hr_str)

    def _get_qpepre_filename_from_date(self, date_str: str, hr_str: str) -> str:
        """Gets QPEPRE filename based on the date and hour in QPEPRE directory.

        Parameters
        ----------
        date_str : str
            The date string in 'YYYY/MM/DD' format.
        hr_str : str
            The hour string in 'HH' format (e.g., '00', '12', '23').

        Returns
        -------
        str
            The path to the QPEPRE file.
        """
        dt_start = datetime.strptime(f"{date_str} {hr_str}", "%Y/%m/%d %H")
        dt_end = dt_start + timedelta(hours=1)

        start_str = dt_start.strftime("%Y%m%d%H%M")
        end_str = dt_end.strftime("%Y%m%d%H%M")

        return f"{self.qpepre_src}/qpepre_{start_str}-{end_str}_1_h"

    def _get_rwrf_paths(self, date_str: str, hr_str: str) -> tuple[str, str]:
        """Gets the original RWRF file path and the path for the new processed file.

        Parameters
        ----------
        date_str : str
            The date string in 'YYYY/MM/DD' format.
        hr_str : str
            The hour string in 'HH' format (e.g., '00', '12', '23').

        Returns
        -------
        tuple[str, str]
            The original RWRF file path and the path for the new processed file.
        """
        dt = datetime.strptime(date_str, "%Y/%m/%d")
        fmt_dt_str = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
        org_path = f"{self.rwrf_src}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp"
        new_path = (
            f"{self.output_dir}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp_qpepre.nc"
        )
        pathlib.Path(f"{self.output_dir}/{fmt_dt_str}").mkdir(
            parents=True, exist_ok=True
        )
        return org_path, new_path

    def _check_exists(self, date_str: str, hr_str: str) -> bool:
        """Checks if the final cropped and combined file already exists.

        Parameters
        ----------
        date_str : str
            The date string in 'YYYY/MM/DD' format.
        hr_str : str
            The hour string in 'HH' format (e.g., '00', '12', '23').

        Returns
        -------
        bool
            True if the processed file exists, False otherwise.
        """
        _, new_path = self._get_rwrf_paths(date_str, hr_str)
        cropped_path = new_path.replace("qpepre.nc", "cropped_qpepre.nc")
        return os.path.exists(cropped_path)

    def _convert_txt_to_nc(self, date_str: str, hr_str: str, var_name="qpepre") -> str:
        """Converts a QPEPRE text file to a temporary NetCDF file.

        Parameters
        ----------
        date_str : str
            The date string in 'YYYY/MM/DD' format.
        hr_str : str
            The hour string in 'HH' format (e.g., '00', '12', '23').
        var_name : str
            The variable name to use in the NetCDF file.

        Returns
        -------
        str
            The path to the temporary NetCDF file.
        """
        filename = self._get_qpepre_filename_from_date(date_str, hr_str)
        txt_path = filename + ".txt"
        nc_path = self.output_dir + "/" + filename.split("/")[-1] + ".nc"

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Source QPEPRE text file not found: {txt_path}")

        data = np.loadtxt(txt_path)
        match = re.search(r"(\d{12})-\d{12}", txt_path)
        if not match:
            raise ValueError(f"Filename does not match pattern: {txt_path}")
        start_dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
        start_str = start_dt.strftime("%Y-%m-%d_%H:%M:%S")

        lon, lat, values = data[:, 1], data[:, 2], data[:, 3]
        lat_unique, lon_unique = np.unique(lat), np.unique(lon)
        n_lat, n_lon = len(lat_unique), len(lon_unique)

        val_grid = np.full((1, n_lat, n_lon), np.nan, dtype=np.float32)
        lat_grid = np.full((1, n_lat), np.nan, dtype=np.float32)
        lon_grid = np.full((1, n_lon), np.nan, dtype=np.float32)
        lat_to_idx = {lat: i for i, lat in enumerate(lat_unique)}
        lon_to_idx = {lon: i for i, lon in enumerate(lon_unique)}

        for i in range(data.shape[0]):
            _lon, _lat, _val = data[i, 1], data[i, 2], data[i, 3]
            lat_idx = lat_to_idx[_lat]
            lon_idx = lon_to_idx[_lon]
            val_grid[0, lat_idx, lon_idx] = _val

        lat_grid[0, :] = lat_unique
        lon_grid[0, :] = lon_unique

        with Dataset(nc_path, "w", format="NETCDF4") as ds:
            ds.createDimension("times", 1)
            ds.createDimension("date_strlen", len(start_str))
            ds.createDimension("lat", n_lat)
            ds.createDimension("lon", n_lon)

            times_var = ds.createVariable("times", "S1", ("times", "date_strlen"))
            lat_var = ds.createVariable("lat", "f4", ("times", "lat"))
            lon_var = ds.createVariable("lon", "f4", ("times", "lon"))
            val_var = ds.createVariable(var_name, "f4", ("times", "lat", "lon"))

            times_var[:] = np.array([list(start_str)], dtype="S1")
            lat_var[:] = lat_grid
            lon_var[:] = lon_grid
            val_var[0, :, :] = val_grid

        return nc_path

    @staticmethod
    def combine_rwrf_qpepre(
        rwrf_ds: Dataset, qpepre_ds: Dataset, output_path: str
    ) -> None:
        """Create a new NetCDF file by copying all data and metadata from the RWRF dataset, then interpolating the QPEPRE data onto the RWRF grid and adding it as a new variable.

        Parameters
        ----------
        rwrf_ds : Dataset
            The source RWRF NetCDF dataset.
        qpepre_ds : Dataset
            The source QPEPRE NetCDF dataset to be interpolated and merged.
        output_path : str
            The file path where the combined NetCDF output will be saved.
        """
        with Dataset(output_path, "w", format=rwrf_ds.file_format) as dst:
            for name, dim in rwrf_ds.dimensions.items():
                dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

            for attr in rwrf_ds.ncattrs():
                dst.setncattr(attr, rwrf_ds.getncattr(attr))

            for name, var in rwrf_ds.variables.items():
                out_var = dst.createVariable(name, var.datatype, var.dimensions)
                for attr in var.ncattrs():
                    out_var.setncattr(attr, var.getncattr(attr))
                out_var[:] = var[:]

            rwrf_lat2d, rwrf_lon2d = rwrf_ds["XLAT"][0, :], rwrf_ds["XLONG"][0, :]
            qpepre_lat1d, qpepre_lon1d = qpepre_ds["lat"][:], qpepre_ds["lon"][:]
            qpepre_var2d = qpepre_ds["qpepre"][0, :, :]
            qpepre_lon2d, qpepre_lat2d = np.meshgrid(qpepre_lon1d, qpepre_lat1d)

            points = np.column_stack((qpepre_lat2d.ravel(), qpepre_lon2d.ravel()))
            values = qpepre_var2d.ravel()
            target_points = np.column_stack((rwrf_lat2d.ravel(), rwrf_lon2d.ravel()))

            interp_val = griddata(
                points, values, target_points, method="linear", fill_value=0.0
            )
            interp_2d = interp_val.reshape(rwrf_lat2d.shape)

            qpepre_var = dst.createVariable("qpepre", "f4", rwrf_ds["XLAT"].dimensions)
            qpepre_var.description = (
                "Precipitation from QPEPRE, interpolated to RWRF grid"
            )
            qpepre_var.units = "mm/hr"  # Needs verfication
            qpepre_var[0, :, :] = interp_2d

    def _crop_rwrf_by_qpepre(self, src_path: str) -> str:
        """Crops a combined RWRF-QPEPRE file to the valid (non-NaN) data area of QPEPRE.

        Parameters
        ----------
        src_path : str
            The file path of the source combined RWRF-QPEPRE NetCDF file.

        Returns
        -------
        str
            The file path of the cropped NetCDF file.
        """
        cropped_path = src_path.replace("qpepre.nc", "cropped_qpepre.nc")

        with Dataset(src_path, "r") as src, Dataset(cropped_path, "w") as dst:
            qpepre = src.variables["qpepre"][0, :, :]
            mask = ~np.isnan(qpepre)
            if not np.any(mask):
                os.remove(cropped_path)
                raise ValueError("No valid (non-NaN) data found in qpepre variable.")

            coords = np.argwhere(mask)
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)

            dst.createDimension("south_north", max_row - min_row + 1)
            dst.createDimension("west_east", max_col - min_col + 1)
            for name, dim in src.dimensions.items():
                if name not in ["south_north", "west_east"]:
                    dst.createDimension(name, len(dim))

            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))

            for name, var in src.variables.items():
                out_var = dst.createVariable(name, var.datatype, var.dimensions)
                for attr in var.ncattrs():
                    out_var.setncattr(attr, var.getncattr(attr))

                if var.ndim >= 2 and var.dimensions[-2:] == (
                    "south_north",
                    "west_east",
                ):
                    out_var[:] = var[..., min_row : max_row + 1, min_col : max_col + 1]
                else:
                    out_var[:] = var[:]

        return cropped_path

    def _process(self, date_str: str, hr_str: str):
        """Processes a single date and hour by converting the corresponding QPEPRE text file to NetCDF, combining it with the matching RWRF NetCDF file, and cropping the result to the valid QPEPRE data region.

        Parameters
        ----------
        date_str : str
            The date string in 'YYYY/MM/DD' format.
        hr_str : str
            The hour string in 'HH' format (e.g., '00', '12', '23').
        """
        temp_nc_path = None
        try:
            temp_nc_path = self._convert_txt_to_nc(date_str, hr_str)

            org_rwrf_path, new_rwrf_path = self._get_rwrf_paths(date_str, hr_str)
            with Dataset(temp_nc_path) as qpepre_ds, Dataset(org_rwrf_path) as rwrf_ds:
                self.combine_rwrf_qpepre(rwrf_ds, qpepre_ds, new_rwrf_path)

            cropped_path = self._crop_rwrf_by_qpepre(new_rwrf_path)
            logger.info(f"Successfully created: {cropped_path}")

        except Exception as e:
            logger.info(f"ERROR processing QPEPRE {date_str} {hr_str}- {e}")
        finally:
            if temp_nc_path and os.path.exists(temp_nc_path):
                logger.debug(f"Removed temporary file: {temp_nc_path}")
                os.remove(temp_nc_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
    )
    processor = RWRFQPEPREProcessor(
        qpepre_src=CONFIG.qpepre,
        rwrf_src=CONFIG.rwrf,
        output_dir=CONFIG.rwrf,
    )
    processor.process_files_from_date_list(CONFIG.date_strs, CONFIG.hr_strs)
