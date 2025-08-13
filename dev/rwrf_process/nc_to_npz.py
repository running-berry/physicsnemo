import logging
import shutil

from datasource import ERA5, RWRF, RWRFQPEPREProcessor
from utils import CONFIG

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

era5 = ERA5(
    nc_folder=CONFIG.era5,
    npz_folder="../data/cache/era5/train",
    verbose=True,
)
era5()

# make sure qpepre is interpolated to rwrf
tmp_folder = "../data/tmp"
rwrf_qpepre_processor = RWRFQPEPREProcessor(
    qpepre_src=CONFIG.qpepre, rwrf_src=CONFIG.rwrf, output_dir=tmp_folder
)
rwrf_qpepre_processor()

rwrf = RWRF(
    nc_folder=tmp_folder,
    npz_folder="../data/cache/rwrf/train",
    verbose=True,
)
rwrf()
shutil.rmtree(tmp_folder, ignore_errors=True)  # for temporary files
