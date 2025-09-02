import logging
import pathlib
import re
import shutil
import os

import netCDF4 as nc
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Iterable

logger = logging.getLogger(__name__)


def _convert_one_nc(
    nc_path_str: str,
    npz_folder_str: str,
    error_folder_str: str,
    overwrite: bool,
) -> tuple[str, bool, str | None]:
    """
    Worker function (must be at module level for 'spawn').
    Returns: (filename, success, error_message)
    """
    nc_path = pathlib.Path(nc_path_str)
    npz_folder = pathlib.Path(npz_folder_str)
    error_folder = pathlib.Path(error_folder_str)

    try:
        if not nc_path.exists():
            return (nc_path.name, False, "NetCDF file not found")

        # Extract variable/date from filename early so we can determine output path
        basename = nc_path.stem
        parts = basename.split("_")
        if len(parts) < 2:
            # Move unrecognized format files to error
            try:
                shutil.move(nc_path, error_folder / nc_path.name)
            except Exception:
                pass
            return (nc_path.name, False, f"Filename format not recognized: {basename}")

        variable = parts[0]
        date_str = parts[1]

        # Output filename (may be re-mapped if tp -> qpepre later)
        # We'll compute the final variable name after reading. For now, compute candidate paths.
        # But we must still open to confirm remapping, so can't skip solely on this pre-name.
        # We still do a quick existence check with the naive name to reduce submissions;
        # the child will do a final check again with the true name.
        naive_out = npz_folder / f"{variable}_{date_str}.npz"
        if not overwrite and naive_out.exists():
            # Might be a false positive for 'tp' remap, but child will recheck after reading.
            pass

        # Read dataset
        try:
            with nc.Dataset(nc_path) as ds:
                # Parse variable if pressure-level like u1000, v250, z700, t850, q500 (but not 10m winds)
                pressure_level_pattern = r"^(u|v|z|t|q)(\d+)$"
                is_pres_lvl_var = (
                    variable not in ("u10", "v10") and re.match(pressure_level_pattern, variable) is not None
                )
                var_key = parts[0][0] if is_pres_lvl_var else parts[0]

                # Validate keys exist
                for key in ["latitude", "longitude", "valid_time", var_key]:
                    if key not in ds.variables:
                        raise KeyError(
                            f"Key '{key}' not in dataset variables: {list(ds.variables.keys())}"
                        )

                lat = ds.variables["latitude"][:]
                lon = ds.variables["longitude"][:]
                times = ds.variables["valid_time"][:]
                data = ds.variables[var_key][:] if not is_pres_lvl_var else ds.variables[var_key][:, 0, :, :]

                # Revert variable name for saving
                save_var = "qpepre" if var_key == "tp" else parts[0]
        except OSError as e:
            # Move unreadable files to error
            try:
                shutil.move(nc_path, error_folder / nc_path.name)
            except Exception:
                pass
            return (nc_path.name, False, f"Could not open NetCDF: {e}")
        except Exception as e:
            try:
                shutil.move(nc_path, error_folder / nc_path.name)
            except Exception:
                pass
            return (nc_path.name, False, f"Processing error: {e}")

        out_path = npz_folder / f"{save_var}_{date_str}.npz"

        # Final existence check to avoid races
        if (not overwrite) and out_path.exists():
            return (out_path.name, True, None)

        try:
            np.savez(out_path, **{save_var: data}, lat=lat, lon=lon, times=times)
        except Exception as e:
            return (out_path.name, False, f"Failed to save NPZ: {e}")

        return (out_path.name, True, None)

    except Exception as e:
        # Catch-all to ensure the parent gets an error string
        return (nc_path.name, False, f"Unexpected worker error: {e}")


class ERA5_MP:
    """Local ERA5 datasource that reads NetCDF files and saves them as NPZ (cache)."""

    def __init__(
        self,
        nc_folder: str,
        error_folder: str,
        npz_folder: str,
        overwrite: bool = False,
        max_workers: int | None = None,
        chunk_size: int = 64,
    ):
        self.nc_folder = pathlib.Path(nc_folder)
        self.error_folder = pathlib.Path(error_folder)
        self.npz_folder = pathlib.Path(npz_folder)
        self._overwrite = overwrite
        self._max_workers = max_workers or max(os.cpu_count() or 1, 1)
        self._chunk_size = max(1, int(chunk_size))

        if not self.nc_folder.is_dir():
            raise FileNotFoundError(f"Input NetCDF folder not found: {nc_folder}")

        self.npz_folder.mkdir(parents=True, exist_ok=True)
        self.error_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"ERA5 source directory: {self.nc_folder}")
        logger.info(f"ERA5 NPZ cache directory: {self.npz_folder}")
        logger.info(f"ERA5 error directory: {self.error_folder}")
        logger.info(f"overwrite={self._overwrite} max_workers={self._max_workers} chunk_size={self._chunk_size}")

    def __call__(self) -> None:
        self.process_files()

    def _iter_nc_files(self) -> Iterable[pathlib.Path]:
        return self.nc_folder.rglob("*.nc")

    def process_files(self) -> None:
        """Parallel converts all NetCDF files in the folder to NPZ format."""
        nc_files = list(self._iter_nc_files())
        if not nc_files:
            logger.warning("No NetCDF files found.")
            return

        logger.info("Converting ERA5 NetCDF to NPZ (parallel)")

        # Optional pre-filter to avoid submitting obviously-done items:
        to_submit: list[str] = []
        for p in nc_files:
            parts = p.stem.split("_")
            if len(parts) < 2:
                # let worker move it to error
                to_submit.append(str(p))
                continue
            candidate = self.npz_folder / f"{parts[0]}_{parts[1]}.npz"
            if self._overwrite or (not candidate.exists()):
                to_submit.append(str(p))
            else:
                # Might still be tp->qpepre mismatch; submit anyway in batches occasionally.
                # Heuristic: skip most, but allow every Nth to be verified.
                # For simplicity and determinism, we’ll still submit everything — it’s safer.
                to_submit.append(str(p))

        worker = partial(
            _convert_one_nc,
            npz_folder_str=str(self.npz_folder),
            error_folder_str=str(self.error_folder),
            overwrite=self._overwrite,
        )

        submitted = 0
        succeeded = 0
        failed = 0

        # Submit in chunks to keep memory/FD usage sane on huge directories
        for i in range(0, len(to_submit), self._chunk_size):
            batch = to_submit[i : i + self._chunk_size]
            with ProcessPoolExecutor(max_workers=self._max_workers) as ex:
                futures = [ex.submit(worker, path) for path in batch]
                for fut in as_completed(futures):
                    submitted += 1
                    name, ok, err = fut.result()
                    if ok:
                        succeeded += 1
                        # Using debug to keep logs light
                        logger.debug(f"[OK] {name}")
                    else:
                        failed += 1
                        logger.error(f"[FAIL] {name}: {err}")

            logger.info(f"Progress: {submitted}/{len(to_submit)} processed…")

        logger.info(f"Done. Total={subm itted}  Succeeded={succeeded}  Failed={failed}")

    def convert_to_npz(self, nc_path: pathlib.Path) -> None:
        """
        Left intact for synchronous, single-file use (e.g., for ad-hoc calls/tests).
        Parallel path uses the module-level worker.
        """
        # You can keep your original implementation here if you still use it elsewhere.
        # For brevity, you can also call the worker directly:
        name, ok, err = _convert_one_nc(
            str(nc_path),
            str(self.npz_folder),
            str(self.error_folder),
            self._overwrite,
        )
        if not ok:
            logger.error(f"[FAIL] {name}: {err}")
        else:
            logger.debug(f"[OK] {name}")

    def info(self, nc_file: str | None = None) -> None:
        if nc_file is None:
            nc_files = list(self.nc_folder.rglob("*.nc"))
            if not nc_files:
                logger.warning("No NetCDF files found to read info from.")
                return
            nc_file = str(nc_files[0])  # reads the first nc file

        try:
            with xr.open_dataset(nc_file, decode_coords=True, mask_and_scale=False) as ds:
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