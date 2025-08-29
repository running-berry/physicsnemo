import logging
import os
import pathlib
from datetime import datetime, timedelta
import shutil
import yaml

from datasource import RWRF, RWRFQPEPREProcessor
from utils import CONFIG

logger = logging.getLogger(__name__)

def create_datestrs(date_list):
    """
    date_list: ["YYYY/MM/DD", "YYYY/MM/DD"]
    returns:   ["YYYY/MM/DD", "YYYY/MM/DD", ...] inclusive
    """
    fmt = "%Y/%m/%d"
    start = datetime.strptime(date_list[0], fmt).date()
    end   = datetime.strptime(date_list[1], fmt).date()

    if start > end:
        start, end = end, start

    return [(start + timedelta(days=i)).strftime(fmt)
            for i in range((end - start).days + 1)]
class RWRFLite:
    def __init__(
        self,
        tmp_src: str,
        npz_folder: str,
        config_src: str,
        qpepre_src: str = CONFIG.qpepre,
        rwrf_src: str = CONFIG.rwrf,
        verbose: bool = True,
        overwrite: bool = False,
    ):
        self.rwrf_qpepre_processor = RWRFQPEPREProcessor(
            qpepre_src=CONFIG.qpepre, rwrf_src=CONFIG.rwrf, output_dir=tmp_src
        )
        self.rwrf = RWRF(
            nc_folder=tmp_src,
            npz_folder="../data/cache/rwrf",
            verbose=True,
        )
        with open(config_src, "r") as f:
            self.cfg_mod = yaml.safe_load(f)

    def __call__(self, *args, **kwds):
        """Function to get data"""
        train_date_strs = create_datestrs(self.cfg_mod["train_dates"])
        valid_date_strs = create_datestrs(self.cfg_mod["valid_dates"])
        date_strs = train_date_strs + valid_date_strs 
        self.process_files_from_date_list(date_strs, CONFIG.hr_strs) 
    
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
                if self.rwrf._check_exists(date_str, hr_str):
                    logger.info(
                        f"Converted npz files already exists for RWRF  {date_str} {hr_str}, skipping conversion."
                    )
                    continue

                logger.info(
                    f"Queueing RWRF-QPEPRE processing for {date_str} {hr_str}..."
                )
                self.rwrf_qpepre_processor._process(date_str, hr_str)
                logger.info("Converting RWRF NetCDF to NPZ")
                nc_path = self.get_cropped_path(date_str, hr_str)
                self.rwrf.convert_to_npz(nc_path)
                self.remove_temp_files(nc_path, remove_parent=True, force_parent=True)
            
    def get_cropped_path(self, date_str: str, hr_str: str) -> pathlib.Path:
        """Constructs the path to the NetCDF file for a given date and hour.
        Parameters
        ----------
        date_str : str
            The date string in 'YYYY/MM/DD' format.
        hr_str : str
            The hour string in 'HH' format (e.g., '00', '12', '23').

        Returns
        -------
        str
            The path to the NetCDF file for the specified date and hour.
        """
        _, new_path = self.rwrf_qpepre_processor._get_rwrf_paths(date_str, hr_str)
        cropped_path = new_path.replace("qpepre.nc", "cropped_qpepre.nc")

        return pathlib.Path(cropped_path)
    
    def remove_temp_files(
        self,
        nc_path: pathlib.Path,
        remove_parent: bool = True,
        force_parent: bool = False,
        ) -> None:
        """Removes temporary files created during processing.
        Parameters
        ----------
        nc_path : pathlib.Path
            The path to the NetCDF file that was processed.
        Raises
        ------
        FileNotFoundError
            If the specified NetCDF file does not exist.              
        """
        if not nc_path.exists():
            logger.warning(f"Temporary file not found, skipping removal: {nc_path}")
        else:
            try:
                nc_path.unlink()  # os.remove works too, but Path is cleaner
                logger.info("Removed temporary file: %s", nc_path)
            except Exception as e:
                logger.error("Error removing temporary file %s: %s", nc_path, e)
                raise

        if remove_parent:
            parent = nc_path.parent
            try:
                if force_parent:
                    shutil.rmtree(parent)
                    logger.info("Removed parent directory recursively: %s", parent)
                else:
                    parent.rmdir()  # only succeeds if the directory is empty
                    logger.info("Removed empty parent directory: %s", parent)
            except FileNotFoundError:
                logger.warning("Parent directory not found, skipping: %s", parent)
            except OSError as e:
                # Commonly: Directory not empty when force_parent=False
                logger.debug("Did not remove parent directory %s: %s", parent, e)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
        
    )
    rwrf_lite = RWRFLite(
        qpepre_src=CONFIG.qpepre,
        rwrf_src=CONFIG.rwrf,
        config_src="../../examples/weather/stormcast/config/dataset/small.yaml",
        tmp_src="../data/tmp",
        npz_folder="../data/cache/rwrf",
        verbose=True,
    )
    rwrf_lite()