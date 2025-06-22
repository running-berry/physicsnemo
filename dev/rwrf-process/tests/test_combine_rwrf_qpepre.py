import os
import shutil
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata

from utils.config import CONFIG
import combine_rwrf_qpepre as mod
import fetch_rwrf as fetr

# --- parameters ---
date_str = "2019/08/03"
hr_str = "00"

rwrf_nc_orig = os.path.join("..", "data", "rwrf", "2019-08-03_00", "wrfout_d01_2019-08-03_00_interp")
rwrf_nc_comb = os.path.join("..", "cache_test", "rwrf", "wrfout_d01_2019-08-03_00_interp_qpepre.nc")
qpepre_txt = os.path.join("..", "data", "pptn", "qpepre_201908030000-201908030100_1_h.txt")
qpepre_nc  = qpepre_txt.replace('.txt', '.nc')

os.makedirs(os.path.dirname(rwrf_nc_comb), exist_ok=True)

# 1) Convert QPEPRE txt → nc if needed
txt_base = qpepre_txt.replace('.txt','')
if not os.path.exists(qpepre_nc):
    print(f"Converting QPEPRE text → NetCDF: {qpepre_txt} → {qpepre_nc}")
    mod.convert_txt_to_nc(date_str, hr_str)
else:
    print(f"QPEPRE NetCDF exists, skipping: {qpepre_nc}")

# 2) Copy original RWRF to combined path
shutil.copy(rwrf_nc_orig, rwrf_nc_comb)
print(f"Copied RWRF: {rwrf_nc_orig} → {rwrf_nc_comb}")

# 3) Apply combine_rwrf_qpepre to add ‘qpepre’ variable time to open datasets
with Dataset(rwrf_nc_comb, 'r+') as rwrf_ds, Dataset(qpepre_nc, 'r') as qpe_ds:
    print("Combining datasets…")
    mod.combine_rwrf_qpepre(rwrf_ds, qpe_ds)
    qpe_interp = rwrf_ds.variables['qpepre'][0, :, :]

# 4) Validate that the new variable exists and has non-NaN values
nan_count = np.isnan(qpe_interp).sum()
total = qpe_interp.size
print(f"qpepre var added: shape={qpe_interp.shape}, NaNs={nan_count}/{total}")

# 5) Extract raw RWRF precip from the original file
with Dataset(rwrf_nc_orig, 'r') as orig_ds:
    rwrf_raw = orig_ds.variables['RAINNC'][0, :, :].data

with Dataset(rwrf_nc_comb, 'r') as comb_ds:
    rwrf_comb = comb_ds.variables['RAINNC'][0, :, :].data

# 6) Compare first and last rows for RWRF, QPEPRE, and combined
for idx in [0, -1]:
    print(f"\nRow {idx} comparison:")
    print(f"  RWRF raw       : {rwrf_raw[idx, :]}")
    print(f"  QPEPRE interp  : {qpe_interp[idx, :]}")
    print(f"  RWRF combined  : {rwrf_comb[idx, :]}")
    # difference between combined and raw
    diff = rwrf_comb[idx, :] - rwrf_raw[idx, :]
    print(f"  Diff (comb - raw): {diff}")

# 6) Spot-check with direct griddata interpolation
#    load the original txt points: lon, lat, precip
pts = np.loadtxt(qpepre_txt)  # assumes columns: lon, lat, precip
lons_pts, lats_pts, vals_pts = pts[:,0], pts[:,1], pts[:,2]

# get the WRF grid
with Dataset(rwrf_nc_orig, 'r') as orig_ds:
    lons_wrf = orig_ds.variables['XLONG'][0, :, :]
    lats_wrf = orig_ds.variables['XLAT'][0, :, :]

# perform direct interpolation
qpe_direct = griddata(
    (lons_pts, lats_pts),
    vals_pts,
    (lons_wrf, lats_wrf),
    method='linear'
)

# compare to in-file result
for idx in [0, -1]:
    direct_row = qpe_direct[idx, :]
    interp_row = qpe_interp[idx, :]
    print(f"\nDirect griddata vs. in-file interp (row {idx}):")
    print(f"  direct  : {direct_row}")
    print(f"  in-file : {interp_row}")
    print(f"  Δ        : {interp_row - direct_row}")


print("Test complete.")
