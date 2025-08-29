import logging
import pathlib
import re
import shutil

import netCDF4 as nc
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ERA5:
    """Local ERA5 datasource that reads NetCDF files synchronously and saves them
    to NPZ format. The output folder acts as a cache.

    Parameters
    ----------
    nc_folder : str
        The root directory where source NetCDF files are stored.
        Expected structure: `{nc_folder}/{variable}_{YYYY}{MM}{DD}{HH}.nc`
    npz_folder : str
        The directory where output NPZ files will be saved. This acts as a cache.
    verbose : bool, optional
        If True, shows a progress bar during conversion. By default True.
    overwrite: bool, optional
        If True, overwrites existing NPZ files. By default False.
    """

    def __init__(
        self,
        nc_folder: str,
        error_folder: str,
        npz_folder: str,
        overwrite: bool = False,
    ):
        self.nc_folder = pathlib.Path(nc_folder)
        self.error_folder = pathlib.Path(error_folder)
        self.npz_folder = pathlib.Path(npz_folder)
        self._overwrite = overwrite

        if not self.nc_folder.is_dir():
            raise FileNotFoundError(f"Input NetCDF folder not found: {nc_folder}")

        self.npz_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"ERA5 source directory: {self.nc_folder}")

    def __call__(self) -> None:
        """Function to get data"""
        self.process_files()

    def process_files(self) -> None:
        """Synchronously converts all NetCDF files in the folder to NPZ format."""
        nc_files = list(self.nc_folder.rglob("*.nc"))
        logger.info("Converting ERA5 NetCDF to NPZ")
        for nc_file in nc_files:
            self.convert_to_npz(nc_file)

    def convert_to_npz(self, nc_path: pathlib.Path) -> None:
        """Loads a single NetCDF file, extracts data, and saves it as an NPZ file.

        Parameters
        ----------
        nc_path : pathlib.Path
            The path to the NetCDF file to convert.
        """
        if not nc_path.exists():
            logger.warning(f"NetCDF file not found, skipping: {nc_path}")
            return

        try:
            with nc.Dataset(nc_path) as ds:
                # Try to extract variable name from filename: {variable}_{YYYY}{MM}{DD}{HH}.nc
                basename = nc_path.stem
                parts = basename.split("_")
                if len(parts) < 2:
                    logger.error(f"Filename format not recognized: {basename}")
                    return
                variable = parts[0]
                
                # parse variable if kind of pressure level variable
                is_pres_lvl_var = False
                pressure_level_pattern = r"^(u|v|z|t|q)(\d+)$"
                if (
                    variable != "u10"
                    and variable != "v10"
                    and re.match(pressure_level_pattern, variable)
                ):
                    variable = parts[0][0] # e.g. u1000 -> u, v250 -> v
                    is_pres_lvl_var = True
                date_str = parts[1]

                for key in ["latitude", "longitude", "valid_time", variable]:
                    if key not in ds.variables:
                        raise KeyError(
                            f"Key '{key}' not found in dataset variables: {list(ds.variables.keys())}"
                        )

                lat = ds.variables["latitude"][:]
                lon = ds.variables["longitude"][:]
                times = ds.variables["valid_time"][:]
                data = ds.variables[variable][:] if not is_pres_lvl_var else ds.variables[variable][:, 0, :, :]
                # revert variable name if needed for saving .npz
                if variable == "tp":
                    variable = "qpepre"
                else:
                    variable = parts[0]

        except OSError as e:
            logger.error(f"Could not open NetCDF file {nc_path}: {e}")
            # move to error folder
            shutil.move(nc_path, self.error_folder / nc_path.name)
            return
        except Exception as e:
            logger.error(f"Error processing file {nc_path}: {e}")
            # move to error folder
            shutil.move(nc_path, self.error_folder / nc_path.name)
            return

        fn = f"{variable}_{date_str}.npz"
        out_path = self.npz_folder / fn

        if not self._overwrite and out_path.exists():
            logger.debug(f"File exists, skipping: {out_path}")  #comment out to speed up
            return

        try:
            np.savez(out_path, **{variable: data}, lat=lat, lon=lon, times=times)
            logger.debug(f"Successfully saved {out_path}")
        except Exception as e:
            logger.error(f"Failed to save NPZ file {out_path}: {e}")
            return

    def info(self, nc_file: str | None = None) -> None:
        """Prints info about the data source."""
        if nc_file == None:
            nc_files = list(self.nc_folder.rglob("*.nc"))
            if not nc_files:
                logger.warning("No NetCDF files found to read info from.")
                return
            nc_file = nc_files[0] # reads the first nc file
        
        try:
            with xr.open_dataset(
                nc_file, decode_coords=True, mask_and_scale=False
            ) as ds:
                logger.debug("ERA5 dataset global attributes:")
                for k, v in ds.attrs.items():
                    logger.debug(f"{k}: {v}")

                logger.debug("ERA5 dataset dimensions:")
                for k, v in ds.sizes.items():
                    logger.debug(f"{k}: {v}")

                logger.debug("ERA5 dataset variables:")
                for k, v in ds.variables.items():
                    logger.debug(f"{k}: {v}")

        except Exception as e:
            logger.error(f"Could not read info from NetCDF file: {e}")
            raise e

