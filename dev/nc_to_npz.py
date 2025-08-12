import logging
import pathlib

from datasource import ERA5, RWRF
from rwrf_process.combine_rwrf_qpepre import (
    check_rwrf_qpepre_exists,
    store_rwrf_qpepre_dataset,
)

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

# make sure qpepre is appended to rwrf
pptn_folder = pathlib.Path("./data/pptn")
qpepre_files = list(pptn_folder.rglob("*.txt"))
for file in qpepre_files:
    basename = file.stem
    date = basename.split("_")[1][:-2]
    date_str = f"{date[:4]}/{date[4:6]}/{date[6:8]}"
    hr_str = date[-2:]
    if check_rwrf_qpepre_exists(date_str, hr_str):
        logger.info(
            f"RWRF QPEPRE dataset already exists for {date_str} {hr_str}. Skipping."
        )
        continue
    store_rwrf_qpepre_dataset(date_str, hr_str)

rwrf = RWRF(
    nc_folder="./data/rwrf",
    npz_folder="./data/cache/rwrf/train",
    verbose=True,
)
rwrf()
