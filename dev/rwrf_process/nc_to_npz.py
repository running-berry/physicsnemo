import logging
import shutil

from datasource import ERA5, RWRF, RWRFQPEPREProcessor, RWRFLite
from utils import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

era5 = ERA5(
    nc_folder=CONFIG.era5,
    error_folder=CONFIG.era5_deprecated,
    npz_folder="../data/cache/era5/",
    verbose=True,
)
era5()

# make sure qpepre is interpolated to rwrf
tmp_folder = "../data/tmp"

# create new RWRFLite instance for RWRFQPEPRE and RWRF processing

rwrf_lite = RWRFLite(
    qpepre_src=CONFIG.qpepre,
    rwrf_src=CONFIG.rwrf,
    config_src="../../examples/weather/stormcast/config/dataset/small.yaml",
    tmp_src=tmp_folder,
    npz_folder="../data/cache/rwrf/",
    verbose=True,
)

rwrf_lite() #this will process rwrf nc files, convert them to npz format and delete the tmp files one hour at a time

shutil.rmtree(tmp_folder)  # for temporary files
