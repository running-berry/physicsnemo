import logging
import shutil

from datasource import ERA5, RWRF, RWRFLite, RWRFQPEPREProcessor
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

# create new RWRFLite instance for RWRFQPEPRE and RWRF processing
rwrf_lite = RWRFLite(
    qpepre_src=CONFIG.qpepre,
    rwrf_src=CONFIG.rwrf,
    tmp_src=tmp_folder,
    npz_folder="../data/cache/rwrf/train",
    verbose=True,
)

rwrf_lite()  # this will process rwrf nc files, convert them to npz format and delete the tmp files one hour at a time

shutil.rmtree(tmp_folder)  # for temporary files
