import logging
import pathlib

import netCDF4 as nc
import numpy as np
import xarray as xr
from lexicon import RWRFLexicon

logger = logging.getLogger(__name__)

PRES_IDX = {
    1000: 0,
    925: 3,
    850: 6,
    700: 11,
    600: 13,
    500: 15,
    400: 17,
    300: 19,
    250: 20,
    200: 22,
    150: 24,
    100: 26,
    50: 28,
}

VARIABLES = [
    "u10",
    "v10",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
    "qpepre",
]


class RWRF:
    """Local RWRF datasource that reads NetCDF files asynchronously and saves them
    to NPZ format. The output folder acts as a cache.

    Parameters
    ----------
    nc_folder : str
        The root directory where source NetCDF files are stored.
        Expected structure: `{nc_folder}/{YYYY}-{MM}-{DD}_{HH}/wrfout_d01_{YYYY}-{MM}-{DD}_{HH}_interp_cropped_qpepre.nc`
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
        npz_folder: str,
        verbose: bool = True,
        overwrite: bool = False,
    ):
        self.nc_folder = pathlib.Path(nc_folder)
        self.npz_folder = pathlib.Path(npz_folder)
        self._verbose = verbose
        self._overwrite = overwrite
        self.lexicon = RWRFLexicon

        if not self.nc_folder.is_dir():
            raise FileNotFoundError(f"Input NetCDF folder not found: {nc_folder}")

        self.npz_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"RWRF source directory: {self.nc_folder}")

    def __call__(self) -> None:
        """Function to get data"""
        self.process_files()

    async def process_files(self) -> None:
        """Asynchronously converts all NetCDF files in the folder to NPZ format."""
        nc_files = list(self.nc_folder.rglob("*_cropped_qpepre.nc"))
        logger.info("Converting RWRF NetCDF to NPZ")
        for nc_file in nc_files:
            self.convert_to_npz(nc_file)

    def convert_to_npz(self, nc_path: pathlib.Path) -> None:
        """Loads a single NetCDF file and creates concurrent tasks to save each
        variable to a separate NPZ file.

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
                # Extract date string from filename: wrfout_d01_{YYYY}-{MM}-{DD}_{HH}_interp_cropped_qpepre.nc
                basename = nc_path.stem
                if not basename.startswith("wrfout_d01_") or not basename.endswith(
                    "_interp_cropped_qpepre"
                ):
                    logger.error(f"Filename format not recognized: {basename}")
                    return
                date_str = basename[11:-22].replace("_", "").replace("-", "")
                lat = ds.variables["XLAT"][:]
                lon = ds.variables["XLONG"][:]
                times = ds.variables["Times"][:]

                for var in VARIABLES:
                    rwrf_name, modifier = self.lexicon[var]
                    if ds.variables[rwrf_name].ndim == 3:
                        data = ds.variables[rwrf_name][:]
                    else:
                        level = int(var[1:])
                        data = ds.variables[rwrf_name][:, PRES_IDX[level], :, :]
                    data = modifier(data)
                    self._save_variable_npz(var, date_str, lat, lon, times, data)

        except Exception as e:
            logger.error(f"Error processing file {nc_path}: {e}")
            raise e

    def _save_variable_npz(
        self,
        variable_id: str,
        date_str: str,
        lat: np.ndarray,
        lon: np.ndarray,
        times: np.ndarray,
        data: np.ndarray,
    ) -> None:
        """Saves data of a single variable to an NPZ file.

        Parameters
        ----------
        ds : nc.Dataset
            The NetCDF dataset containing the variable to save.
        variable_id : str
            The ID of the variable to save.
        date_str : str
            The date string extracted from the filename.
        lat : np.ndarray
            The latitude array.
        lon : np.ndarray
            The longitude array.
        times : np.ndarray
            The time array.
        data: np.ndarray
            The data array to save.
        """
        try:
            fn = f"{variable_id}_{date_str}.npz"
            out_path = self.npz_folder / fn
            if not self._overwrite and out_path.exists():
                logger.debug(f"File exists, skipping: {out_path}")
                return

            np.savez(out_path, **{variable_id: data}, lat=lat, lon=lon, times=times)
            logger.info(f"Successfully saved {out_path}")
        except Exception as e:
            logger.error(f"Failed to save NPZ file {out_path}: {e}")
            raise e

    @property
    def info(self) -> None:
        """Prints info about the data source."""
        try:
            nc_files = list(self.nc_folder.rglob("*.nc"))
            if not nc_files:
                logger.warning("No NetCDF files found to read info from.")
                return

            first_nc = nc_files[0]
            with xr.open_dataset(
                first_nc, decode_coords=True, mask_and_scale=False
            ) as ds:
                logger.debug("RWRF dataset global attributes:")
                for k, v in ds.attrs.items():
                    logger.debug(f"{k}: {v}")

                logger.debug("RWRF dataset dimensions:")
                for k, v in ds.sizes.items():
                    logger.debug(f"{k}: {v}")

                logger.debug("RWRF dataset variables:")
                for k, v in ds.variables.items():
                    if k == "pres_levels":
                        logger.debug(f"{k}: {v}\n{v.values} (pressure levels in hPa)")
                    else:
                        logger.debug(f"{k}: {v}")

        except Exception as e:
            logger.error(f"Could not read info from NetCDF file: {e}")
            raise e
