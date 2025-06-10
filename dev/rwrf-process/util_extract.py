import numpy as np

def load_npz(path: str) -> dict:
    return np.load(path, allow_pickle=True)

def find_index_bounds(arr: np.ndarray, vmin: float, vmax: float) -> tuple[int, int]:
    i_min = np.argmin(np.abs(arr - vmin))
    i_max = np.argmin(np.abs(arr - vmax))
    return tuple(sorted([i_min, i_max]))

def slice_region(lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float
    ) -> tuple[np.ndarray, np.ndarray]:

    i0, i1 = find_index_bounds(lat, lat_min, lat_max)
    j0, j1 = find_index_bounds(lon, lon_min, lon_max)

    lat_sel = lat[i0:i1+1]
    lon_sel = lon[j0:j1+1]

    return np.meshgrid(lon_sel, lat_sel)

def extract_region(path: str,
  var: str,
  lon_min: float, lon_max: float,
  lat_min: float, lat_max: float
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data = load_npz(path)
    arr = data[var]
    lat = data['lat']
    lon = data['lon']
    times = data['times']

    lon_grid, lat_grid = slice_region(lat, lon, lat_min, lat_max, lon_min, lon_max)

    # now slice the data itself: assume arr dims are (..., y, x)
    # find the y/x slices we used for lat/lon
    i0, i1 = find_index_bounds(lat, lat_min, lat_max)
    j0, j1 = find_index_bounds(lon, lon_min, lon_max)

    # support for arr[..., y, x]  or arr[time, y, x] or arr[time, level, y, x]
    slice_y = slice(i0, i1+1)
    slice_x = slice(j0, j1+1)
    data_sub = arr[..., slice_y, slice_x]           # (t,lev, y, x), or (t, y, x)

    return data_sub, lon_grid, lat_grid, times

if __name__ == "__main__":
    # example usage
    PATH = "./cache/era5/20200601_00.npz"
    VAR = "t2m"
    LON_MIN, LON_MAX = 121.00, 121.75
    LAT_MIN, LAT_MAX = 25.00, 25.75

    data_sub, lon_grid, lat_grid, times= extract_region(
      PATH, VAR,
      lon_min=LON_MIN, lon_max=LON_MAX,
      lat_min=LAT_MIN, lat_max=LAT_MAX,
    )

    print("lat:\n", lat_grid)
    print("lon:\n", lon_grid)
    print(f"{VAR} slice shape:", data_sub.shape)

    data = load_npz(PATH)
    t2m = data[VAR]
    lat = data['lat']
    lon = data['lon']
    times = data['times']
    print("Original data shapes:")
    print(t2m.shape, lat.shape, lon.shape, times.shape)

    t2m = data_sub  # assuming t2m is the variable of interest
    lat = lat_grid
    lon = lon_grid
    print("Extracted data shapes:")
    print(t2m.shape, lat.shape, lon.shape, times.shape)