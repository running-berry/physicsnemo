import os
import shutil
import numpy as np
from netCDF4 import Dataset

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

# Ensure cache folder exists
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
    array_mod = rwrf_ds.variables['qpepre'][0, :, :]

# 4) Validate that the new variable exists and has non-NaN values
nan_count = np.isnan(array_mod).sum()
total = array_mod.size
print(f"qpepre var added: shape={array_mod.shape}, NaNs={nan_count}/{total}")

# 5) Optional: Compare interpolation against direct griddata call for spot-check
# (load raw qpepre points & RWRF lat/lon to validate)

print("Test complete.")
