import logging

from datasource import ERA5, RWRF

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

era5 = ERA5(
    nc_folder="./data/era5/train",
    npz_folder="./data/cache/era5/train",
    verbose=True,
)

era5()

rwrf = RWRF(
    nc_folder="./data/rwrf",
    npz_folder="./data/cache/rwrf/train",
    verbose=True,
)
rwrf()
