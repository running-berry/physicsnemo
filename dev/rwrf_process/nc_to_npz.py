import logging
import shutil

from datasource import ERA5, RWRF, RWRFQPEPREProcessor, RWRFLite
from utils import CONFIG

logging.basicConfig(
    level=logging.DEBUG, # change to DEBUG for detailed logs
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

era5 = ERA5(
    nc_folder=CONFIG.era5,
    error_folder=CONFIG.era5_deprecated,
    npz_folder="../data/cache/era5/",
    # overwrite=True, # set to True to overwrite existing npz files
)
era5()
# these two have stds errors
# era5.info(nc_file="/mnt/ncdr/era5/t925/2019/t925_2019010100.nc")
# era5.info(nc_file="/mnt/ncdr/era5/t1000/2019/t1000_2019010100.nc")

# make sure qpepre is interpolated to rwrf
tmp_folder = "../data/tmp"

# create new RWRFLite instance for RWRFQPEPRE and RWRF processing

rwrf_lite = RWRFLite(
    qpepre_src=CONFIG.qpepre,
    rwrf_src=CONFIG.rwrf,
    config_src="../../examples/weather/stormcast/config/dataset/small.yaml",
    tmp_src=tmp_folder,
    npz_folder="../data/cache/rwrf/",
    # overwrite=True, # set to True to overwrite existing npz files
)

rwrf_lite() #this will process rwrf nc files, convert them to npz format and delete the tmp files one hour at a time

shutil.rmtree(tmp_folder)  # for temporary files
