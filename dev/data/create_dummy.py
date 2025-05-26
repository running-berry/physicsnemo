import os
import xarray as xr
import numpy as np

num_channel = 5
domain_size = (32, 32)
test_years = [2025]

for fname in ["DummyHighRes", "DummyLowRes"]:
    folder_path=f"{fname}/stats"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print(f"'{folder_path}' is existed")
    np.save(f"{folder_path}/means.npy", np.random.rand(num_channel,).astype(np.float32))
    np.save(f"{folder_path}/stds.npy", np.random.rand(num_channel,).astype(np.float32))

    # an example
    lon_min, lon_max = 272.28, 272.32
    lat_min, lat_max = 42.06, 42.02   
    lon = np.linspace(lon_min, lon_max, domain_size[0])
    lat = np.linspace(lat_min, lat_max, domain_size[1])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for year in test_years:
        base_date = np.datetime64(f'{year}-01-01T00:00:00')
        offsets = np.arange(365*24).astype(np.int64)
        datetime_array = base_date + offsets * np.timedelta64(1, 'h')

        chunk_sizes = {
            'time': 1,
            'channel': num_channel,
            'latitude': len(lat_grid),
            'longitude': len(lon_grid),
        }

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        data_shape = (365*24, num_channel)+domain_size
        year_data = xr.Dataset({
            f'{fname}': (['time', 'channel', 'y', 'x'], np.random.rand(*data_shape).astype(np.float32)),
            'time': datetime_array,
            'channel': ['a', 'b', 'c', 'd', 'e'],
            'latitude': (["y", "x"], lat_grid),
            'longitude': (["y", "x"], lon_grid)
        })

        data_enc = {f'{fname}':{'dtype':'float32', 'compressor':None}}
        year_data.to_zarr(f'{fname}/{year}.zarr', mode='w', consolidated=True, encoding=data_enc)
