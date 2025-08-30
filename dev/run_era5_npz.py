import sys
sys.path.insert(0, "/mnt/home/dczy-cmla-8ec31f/projects/N1_new/physicsnemo/dev/rwrf_process/datasource")

from era5 import ERA5

nc_folder = "/mnt/home/dczy-cmla-8ec31f/NCDR_StormCast/era5/v1000/2018"
npz_folder = "/mnt/home/dczy-cmla-8ec31f/NCDR_StormCast/era5_npz/v1000/2018"

converter = ERA5(nc_folder, npz_folder, overwrite=True)
converter()
