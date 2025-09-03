import logging
import os
import pathlib
from datetime import datetime, timedelta
import time
import csv
import json
import csv

import numpy as np
import regex as re
from netCDF4 import Dataset
from scipy.interpolate import griddata

logger = logging.getLogger("rwrf")

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
        qpepre_src: str,
        rwrf_src: str,
        perf_dir: str | None = None, 
    ):
        self.qpepre_src = pathlib.Path(qpepre_src)
        self.rwrf_src   = pathlib.Path(rwrf_src)      # <-- make Path
        self.output_dir = pathlib.Path(output_dir)    # <-- make Path
        self.output_dir.mkdir(parents=True, exist_ok=True) # redundant?
        
        # ensure "rwrf" subdirectory inside output_dir
        self.output_rwrf_dir = self.output_dir / "rwrf"
        self.output_rwrf_dir.mkdir(parents=True, exist_ok=True)

        # for performance measurement
        self.time_records: dict[str, float] = {
            "txt_to_nc": 0.0,
            "combine": 0.0,
            "crop": 0.0,
            "find_rwrf_paths": 0.0,
            "total": 0.0,
        }

        # build index of RWRF files once
        self._rwrf_index: dict[str, pathlib.Path] = {}
        self._build_rwrf_index()


    def __call__(self):
        """
        Interpolates all QPEPRE files in the source directory into corresponding RWRF datasets.
        Handles filenames like: qpepre_YYYYMMDDHHMM-YYYYMMDDHHMM_1_h.txt
        """
        src_path = pathlib.Path(self.qpepre_src)
        qpepre_files = list(src_path.rglob("*.txt"))

        # regex: capture the first 12-digit timestamp after 'qpepre_'

        pat = re.compile(r"qpepre_(\d{12})-\d{12}_1_h(?:\.txt)?$")

        for file in qpepre_files:
            m = pat.search(file.name)
            if not m:
                logger.debug(f"Skip (unrecognized name): {file}")
                continue

            start_str = m.group(1)      # YYYYMMDDHHMM
            y, mth, d, hh = start_str[0:4], start_str[4:6], start_str[6:8], start_str[8:10]
            date_str = f"{y}/{mth}/{d}"
            hr_str = hh

            if self._check_exists(date_str, hr_str):
                logger.debug(f"RWRF QPEPRE dataset already exists for {date_str} {hr_str}. Skipping.")
                continue

            self._process(date_str, hr_str)

    def _build_rwrf_index(self) -> None:
        """
        Scan the rwrf_src folder once and index wrfout files by their parent folder name
        (e.g., 'YYYY-MM-DD_HH').
        """
        t0 = time.perf_counter()
        logger.info(f"Building RWRF index under {self.rwrf_src}...")
        for p in self.rwrf_src.rglob("wrfout_d01_*_interp"):
            key = p.parent.name  # e.g. '2020-07-01_12'
            self._rwrf_index[key] = p
        logger.info(f"Indexed {len(self._rwrf_index)} RWRF files.")
        t1 = time.perf_counter()
        self.time_records["find_rwrf_paths"] += (t1 - t0)

    def _resolve_qpepre_dir_for_year(self, year: int) -> pathlib.Path | None:
        """
        Return the subdirectory under qpepre_src that holds files for `year`.
        Checks: <root>/<year>/  then  <root>/forai_1hrobs_<year>/
        Falls back to None if neither exists.
        """
        # Common cases based on your layout
        logger.debug(f"Resolving QPEPRE directory for year: {year}")
        cand1 = self.qpepre_src / f"{year}"
        cand2 = self.qpepre_src / f"forai_1hrobs_{year}"
        logger.debug("resolve done")
        if cand1.is_dir():
            return cand1
        if cand2.is_dir():
            return cand2
        return None

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
        logger.info(f"Processing files for dates: {date_strs} and hours: {hr_strs}")
        for date_str in date_strs:
            for hr_str in hr_strs:
                if self._check_exists(date_str, hr_str):
                    logger.debug(
                        f"Converted file already exists for QPEPRE {date_str} {hr_str}, skipping conversion."
                    )
                    continue

                logger.debug(
                    f"Queueing RWRF-QPEPRE processing for {date_str} {hr_str}..."
                )
                self._process(date_str, hr_str)

    def _get_qpepre_filename_from_date(self, date_str: str, hr_str: str) -> str:
        """
        Return the *stem* path (without '.txt') to the QPEPRE file for the given date/hour.
        This handles different per-year folder names and falls back to a recursive search.
        """
        logger.debug(f"Getting QPEPRE filename for date: {date_str}, hour: {hr_str}")
        dt_end = datetime.strptime(f"{date_str} {hr_str}", "%Y/%m/%d %H")
        dt_start = dt_end - timedelta(hours=1)

        start_str = dt_start.strftime("%Y%m%d%H%M")
        end_str = dt_end.strftime("%Y%m%d%H%M")
        filename = f"qpepre_{start_str}-{end_str}_1_h"
        year = dt_start.year

        # 1) Try expected subfolder(s) for the year
        year_dir = self._resolve_qpepre_dir_for_year(year)
        if year_dir:
            candidate = year_dir / f"{filename}.txt"
            if candidate.exists():
                # Return path *without* extension, to match existing call sites
                return str(candidate.with_suffix(""))

        # 2) Fallback: recursive search under the whole root
        matches = list(self.qpepre_src.rglob(f"{filename}.txt"))
        if matches:
            return str(matches[0].with_suffix(""))

        # 3) Give a helpful error
        raise FileNotFoundError(
            f"QPEPRE file not found for {date_str} {hr_str}. "
            f"Tried: {self.qpepre_src}/{{{year}, forai_1hrobs_{year}}}/{filename}.txt"
        )

    # def _get_rwrf_paths(self, date_str: str, hr_str: str) -> tuple[str, str]:
    #     """Gets the original RWRF file path and the path for the new processed file.

    #     Parameters
    #     ----------
    #     date_str : str
    #         The date string in 'YYYY/MM/DD' format.
    #     hr_str : str
    #         The hour string in 'HH' format (e.g., '00', '12', '23').

    #     Returns
    #     -------
    #     tuple[str, str]
    #         The original RWRF file path and the path for the new processed file.
    #     """
    #     dt = datetime.strptime(date_str, "%Y/%m/%d")
    #     fmt_dt_str = dt.strftime(f"%Y-%m-%d_{int(hr_str):02d}")
        
    #     base = pathlib.Path(self.rwrf_src)
    #     matches = list(base.rglob(fmt_dt_str))
    #     if not matches:
    #         logger.warning(f"No matching RWRF files found for {date_str} {hr_str}")
    #         org_path = f"{self.rwrf_src}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp" # dummy path
    #     else:
    #         target_folder = matches[0]
    #         org_path = str(target_folder / f"wrfout_d01_{fmt_dt_str}_interp")
            
    #     new_path = (
    #         f"{self.output_dir}/{fmt_dt_str}/wrfout_d01_{fmt_dt_str}_interp_qpepre.nc"
    #     )
    #     return org_path, new_path
    
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

        org_path = self._rwrf_index.get(fmt_dt_str)
        if org_path is None:
            logger.warning(f"No matching RWRF files found for {date_str} {hr_str}")
            # fallback dummy path to maintain consistency
            org_path = self.rwrf_src / fmt_dt_str / f"wrfout_d01_{fmt_dt_str}_interp"

        new_path = self.output_dir / fmt_dt_str / f"wrfout_d01_{fmt_dt_str}_interp_qpepre.nc"
        return str(org_path), str(new_path)
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
        nc_path = self.output_dir / (pathlib.Path(filename).name + ".nc")

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
            t0 = time.perf_counter()
            logger.info(f"converting txt to nc for {date_str} {hr_str}...")
            temp_nc_path = self._convert_txt_to_nc(date_str, hr_str)
            t1 = time.perf_counter()
            self.time_records["txt_to_nc"] += (t1 - t0)
            logger.info(f"conversion for {date_str} {hr_str} done: {temp_nc_path}")

            t9 = time.perf_counter()
            org_rwrf_path, new_rwrf_path = self._get_rwrf_paths(date_str, hr_str)
            logger.info(f"RWRF path: {org_rwrf_path}, new path: {new_rwrf_path}")
            pathlib.Path(new_rwrf_path).parent.mkdir(parents=True, exist_ok=True)


            t2 = time.perf_counter()
            logger.info(f"combining rwrf and qpepre for {date_str} {hr_str}...")
            with Dataset(temp_nc_path) as qpepre_ds, Dataset(org_rwrf_path) as rwrf_ds:
                self.combine_rwrf_qpepre(rwrf_ds, qpepre_ds, new_rwrf_path)
            t3 = time.perf_counter()
            self.time_records["combine"] += (t3 - t2)
            logger.info(f"combining done rwrf and qpepre for {date_str} {hr_str}") 
            
            t4 = time.perf_counter()
            logger.info(f"cropping rwrf by qpepre for {date_str} {hr_str}...")
            cropped_path = self._crop_rwrf_by_qpepre(new_rwrf_path)
            logger.info(f"Successfully created: {cropped_path}")
            t5 = time.perf_counter()
            self.time_records["crop"] += (t5 - t4)
            logger.info(f"cropping rwrf done by qpepre for {date_str} {hr_str}...")

        except Exception as e:
            logger.debug(f"ERROR processing QPEPRE {date_str} {hr_str}- {e}")
        finally:
            if temp_nc_path and os.path.exists(temp_nc_path):
                logger.debug(f"Removed temporary file: {temp_nc_path}")
                os.remove(temp_nc_path)

    def save_time_records(self, file_path: str, fmt: str = "csv") -> None:
        """
        Save accumulated time records to a specified file.

        Parameters
        ----------
        file_path : str
            Path to the file where timing results will be written.
        fmt : str, optional
            Format of output: "csv" or "json". Default is "csv".
        """
        out_path = pathlib.Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Timing results:")
        if fmt == "csv":
            with out_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "seconds"])
                for step, seconds in self.time_records.items():
                    w.writerow([step, f"{seconds:.6f}"])
        elif fmt == "json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(self.time_records, f, indent=2)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'.")

        logger.info(f"Saved timing results to {out_path}")
        
