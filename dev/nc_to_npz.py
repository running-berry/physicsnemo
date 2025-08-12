import logging

from dev.datasource import ERA5, RWRF

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

era5 = ERA5(
    nc_folder="./dev/data/era5/train",
    npz_folder="./dev/data/cache/era5/train",
    verbose=True,
)

era5()

rwrf = RWRF(
    nc_folder="./dev/data/rwrf",
    npz_folder="./dev/data/cache/rwrf/train",
    verbose=True,
)
rwrf()
