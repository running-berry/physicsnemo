import os
import xarray as xr
import numpy as np
import util_extract as u1

data_var = "t2m"
num_channel = len(data_var)
domain_size = (32, 32)
test_datetime_start = "2019/08/03"
test_datetime_last = "2019/08/03"
test_years = [2019]
cache_path = "./cache"
data_path = "./data"

def create_dummy_arr(
    dt,
    data_path: str,
    data_var: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
):
    """
    Create a dummy data array (filled with NaNs) matching the shape
    of the real data for timestamp dt, and also return the lon/lat grids
    and time coords.
    """
    # build the filename for this dt
    yy, mm, dd, hh = np.datetime_as_string(dt, unit='h')\
                       .replace('T','-').split('-')
    fn = f"{yy}{mm}{dd}_{hh}.npz"
    dt_path = os.path.join(data_path, fn)

    # grab one real sample to infer shapes
    real_arr, lon_grid, lat_grid, times = u1.extract_region(
        dt_path,
        data_var,
        lon_min, lon_max,
        lat_min, lat_max,
    )

    # now make a dummy array of the same shape, filled with NaN
    dummy_arr = np.full_like(real_arr, fill_value=np.nan)

    return dummy_arr, lon_grid, lat_grid, times
"""
def collect_data(date_dt64, cache_dir: str = cache_path):
    missing = []
    # date_str: "YYYY/MM/DD"
    year, month, day = map(int, np.datetime_as_string(date_dt64, unit='D').split('-'))
    base = f"{year}{month}{day}"
    date_str = f"{year:04d}/{month:02d}/{day:02d}"
    
    hh = f"{h:02d}"
    fname = f"{base}_{hh}.npz"
    path = os.path.join(f"{cache_dir}/rwrf", fname)
    if not os.path.exists(path):
        # missing → fetch & save
        try:
            #u2r.save_t2m_numpy(date_str, hh)
            pass
        except:
            missing.append(hh)
    else:
        # already exists → load
        data = np.load(path)
        if h == 0:
            data_arr = data[data_var]
            lat_grid = data['lat']
            lon_grid = data['lon']
            times = data['times']
        else:
            data_arr = np.concatenate((data_arr, data[data_var]), axis=0)
    
    if missing:
        print(f"Missing hours fetched and saved: {', '.join(missing)}")
    else:
        print("All 24 hourly files are already present.")
"""
    
for fname in ["HighRes", "LowRes"]:
    folder_path=f"{fname}/stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print(f"'{folder_path}' is existed")
    

    # determine data path base
    if fname == "HighRes":
        data_path = "./cache/rwrf/"
        import fetch_rwrf as u2
    elif fname == "LowRes":
        data_path = "./cache/era5/"
        import fetch_era5 as u2
    
    
    #lon_min, lon_max = 121.5386, 121.5404
    #lat_min, lat_max = 25.01685, 25.01775   
    lon_min, lon_max = 121.00, 121.75
    lat_min, lat_max = 25.00, 25.75
    #$$Next
    for year in test_years:
        base_date = np.datetime64(test_datetime_start.replace('/', '-') + 'T00:00:00')
        end_date = np.datetime64(test_datetime_last.replace('/', '-')) + np.timedelta64(23, 'h')
        total_hours = int((end_date - base_date) / np.timedelta64(1, 'h')) + 1
        offsets = np.arange(total_hours, dtype=np.int64)
        datetime_array = base_date + offsets * np.timedelta64(1, 'h')
        #create the dummy data format by the first data
        dummy_data, dummy_lon_grid, dummy_lat_grid, dummy_times = create_dummy_arr(
            datetime_array[0],
            data_path, data_var,
            lon_min, lon_max,
            lat_min, lat_max,
        )
        missing_data = []
        data_arr = None
        
        for dt in datetime_array:
            yy, mm, dd, hh = np.datetime_as_string(dt, unit='h').replace('T', '-').split('-')
            dt_path = os.path.join(data_path, f"{yy}{mm}{dd}_{hh}.npz")
            print(f"Processing {dt_path}")

            try:
                dt_data, lon_grid, lat_grid, times= u1.extract_region(
                    dt_path, data_var,
                    lon_min, lon_max,
                    lat_min, lat_max,
                ) 

            except FileNotFoundError:
                #create dummy data
                dt_data = dummy_data.copy()
                missing_data.append(f"{yy}-{mm}-{dd} {hh}:00")
            
            #concatenate data
            if data_arr is None:
                data_arr = dt_data.copy()
                print(data_arr)
            else:
                # concatenate along axis=0 (time or channel, whichever you're stacking)
                data_arr = np.concatenate((data_arr, dt_data), axis=0)
            print(data_arr)
                
        # compute mean and std over time, latitude & longitude → leaves (n_chan,)
        # data_arr shape is (n_time, n_chan, ny, nx)
        print(data_arr)
        means = np.nanmean(data_arr, axis= ..., dtype=np.float64).astype(np.float32)
        stds  = np.nanstd(data_arr, axis= ..., dtype=np.float64).astype(np.float32)

        # save them
        np.save(f"{folder_path}/means.npy", means)
        np.save(f"{folder_path}/stds.npy",  stds)
        """
        data_arr = collect_data(datetime_array)

        data_arr, lon_grid, lat_grid, times= u1.extract_region(
            data_path, data_var,
            lon_min, lon_max,
            lat_min, lat_max,
        )
        
        data_shape = (total_hours, num_channel)+domain_size
        year_data = xr.Dataset({
            f'{fname}': (['time', 'channel', 'y', 'x'], data_arr),
            'time': datetime_array,
            'channel': data_var,
            'latitude': (["y", "x"], lat_grid),
            'longitude': (["y", "x"], lon_grid)
        })
        """
        """
        data_enc = {f'{fname}':{'dtype':'float32', 'compressor':None}}
        year_data.to_zarr(f'{fname}/{year}.zarr', mode='w', consolidated=True, encoding=data_enc)
        """