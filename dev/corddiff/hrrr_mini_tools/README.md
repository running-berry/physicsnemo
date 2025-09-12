
## Overview

1. **Slicing & Stats**  
   `daily_split_and_stats.py`  
   - Reads the combined `hrrr_mini_train.nc` (with `/`, `/input`, `/output`, `/invariant` groups).  
   - Splits it into one file per day, preserving all four groups.  
   - Computes per-variable means & standard deviations and writes `stats_YYYYMMDD.json`. 
   ```bash
   Execution: python3 hrrr_mini_daily_split_and_stats.py hrrr_mini_nc_file_path output_dir
2. **Inspection**  
   `nc_group_inspector.py`  
   - Recursively lists all NetCDF-4 groups, dimensions, variables (and their dims) inside any `.nc` file.  
    ```bash
    Execution: python3 nc_group_inspector.py hrrr_mini_nc_file_path
3. **Add dummy precipitation to HRRR Mini**
   `hrrr_mini_add_precip.py`
   - Copy a NetCDF file and inject a synthetic 'precipitation' DataArray into the existing 'output' group using the netCDF4 low-level API.
    ```bash
    Execution: python3 hrrr_mini_add_precip.py hrrr_mini_nc_file_path 
4. **Extract and save metadata, quick-look plots**
   `plot_n_metadata.py`
   - Reads each group’s dimensions and variable names and writes them out to metadata.json in the specified output directory.
   - For every data variable in the prediction group, it slices the array at ensemble=0, time=0, plots it with Matplotlib, and saves each figure as prediction_<variable>.png in the same directory.
   - You may also visualize the results using view_hrrr_mini_daily_prediction.ipynb
    ```bash
    Execution: python3 plot_n_metadata.py output_dir_path
---

## Requirements

- Python 3.8+  
- [xarray](https://xarray.pydata.org)  
- [netCDF4-python](https://unidata.github.io/netcdf4-python/)  
- [pandas](https://pandas.pydata.org)  
- [numpy](https://numpy.org)  
- [tqdm](https://github.com/tqdm/tqdm)  

Install via pip:

```bash
pip install xarray netCDF4 pandas numpy tqdm