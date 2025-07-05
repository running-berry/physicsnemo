import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import RegularGridInterpolator


def load_npz(path: str) -> dict:
    return np.load(path, allow_pickle=True)


def find_index_bounds(arr: np.ndarray, vmin: float, vmax: float) -> tuple[int, int]:
    i_min = np.argmin(np.abs(arr - vmin))
    i_max = np.argmin(np.abs(arr - vmax))
    return tuple(sorted([i_min, i_max]))


def slice_region(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray]:

    i0, i1 = find_index_bounds(lat, lat_min, lat_max)
    j0, j1 = find_index_bounds(lon, lon_min, lon_max)

    lat_sel = lat[i0 : i1 + 1]
    lon_sel = lon[j0 : j1 + 1]

    return np.meshgrid(lon_sel, lat_sel)


def get_equidistant_indices(length: int, target: int) -> np.ndarray:
    """
    Return `target` indices evenly spaced over [0, length-1].
    """
    if target >= length:
        # no downsampling needed; just return all indices
        return np.arange(length)
    # np.linspace includes both endpoints, round to nearest int
    idx = np.round(np.linspace(0, length - 1, target)).astype(int)
    return np.unique(idx)  # ensure uniqueness if rounding collapsed points


def interp_to_domain(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    data: Optional[np.ndarray],
    domain_size: Tuple[int, int],
    method: str = "linear",
) -> Tuple:
    """
    Upsample (or downsample) a rectilinear grid + optional data array
    to exactly `domain_size = (Ny, Nx)` via interpolation.

    Args:
      lon_grid, lat_grid : 2D arrays of shape (Ny_full, Nx_full)
      data               : optional ndarray with shape (..., Ny_full, Nx_full)
      domain_size        : desired output shape (Ny, Nx)
      method             : interpolation method ("linear", "nearest", "cubic")

    Returns:
      If data is None:
         (lon_new, lat_new)
      else:
         (data_new, lon_new, lat_new)
      where lon_new, lat_new are both shape (Ny, Nx) and
      data_new has shape (..., Ny, Nx).
    """
    Ny, Nx = domain_size
    Ny_full, Nx_full = lat_grid.shape

    # 1D coord vectors (assumes your grid is rectilinear)
    lat1d = lat_grid[:, 0]
    lon1d = lon_grid[0, :]

    # New target coords (evenly spaced in the original lat/lon domain)
    lat_new_1d = np.linspace(lat1d.min(), lat1d.max(), Ny)
    lon_new_1d = np.linspace(lon1d.min(), lon1d.max(), Nx)
    lon_new, lat_new = np.meshgrid(lon_new_1d, lat_new_1d)

    # Prepare points to sample: shape (Ny*Nx, 2)
    pts_new = np.column_stack((lat_new.ravel(), lon_new.ravel()))

    # If no data array given, just return the new grids
    if data is None:
        return lon_new, lat_new

    # Allocate output
    out_shape = data.shape[:-2] + (Ny, Nx)
    data_new = np.empty(out_shape, dtype=data.dtype)

    # Helper to build & use interpolator for a 2D slice
    def interp_slice(arr2d):
        interp = RegularGridInterpolator(
            (lat1d, lon1d), arr2d, method=method, bounds_error=False, fill_value=None
        )
        return interp(pts_new).reshape(Ny, Nx)

    # If data is exactly 2D, interpolate directly
    if data.ndim == 2:
        data_new = interp_slice(data)
    else:
        # loop over all leading dims
        for idx in np.ndindex(*data.shape[:-2]):
            data_new[idx] = interp_slice(data[idx])

    return data_new, lon_new, lat_new


def extract_region(
    path: str,
    var: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    domain_size: Tuple[int, int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data = load_npz(path)
    arr = data[var]  # (t,y,x) or (t,lev,y,x)
    lat = data["lat"]  # maybe (1,Ny,Nx)
    lon = data["lon"]  # maybe (1,Ny,Nx)
    times = data["times"]

    # get true 1D coords
    if lat.ndim == 3:
        lat_1d = lat[0, :, 0]
        lon_1d = lon[0, 0, :]
    else:
        lat_1d = lat
        lon_1d = lon

    # find i/j bounds on those 1D arrays
    i0, i1 = find_index_bounds(lat_1d, lat_min, lat_max)
    j0, j1 = find_index_bounds(lon_1d, lon_min, lon_max)

    # clamp and slice exactly as before...
    slice_y = slice(max(0, i0), min(arr.shape[-2], i1 + 1))
    slice_x = slice(max(0, j0), min(arr.shape[-1], j1 + 1))

    idx = [slice(None)] * arr.ndim
    idx[-2], idx[-1] = slice_y, slice_x
    data_sub = arr[tuple(idx)]

    # build the full meshgrid of lat/lon
    lon_grid_full, lat_grid_full = slice_region(
        lat_1d, lon_1d, lat_min, lat_max, lon_min, lon_max
    )
    Ny_full, Nx_full = lat_grid_full.shape

    # if no down-sampling requested, return full slice
    if domain_size is None:
        return data_sub, lon_grid_full, lat_grid_full, times

    # otherwise, down-sample to (Ny, Nx)
    Ny, Nx = domain_size
    y_idx = get_equidistant_indices(Ny_full, Ny)
    x_idx = get_equidistant_indices(Nx_full, Nx)

    # subset the grids
    lat_grid = lat_grid_full[y_idx][:, x_idx]
    lon_grid = lon_grid_full[y_idx][:, x_idx]

    # subset the data
    # data_sub has shape (..., Ny_full, Nx_full)
    # so we take:
    #   axis = -2 for y, and axis = -1 for x
    data_sub_ds = data_sub[..., y_idx, :][..., :, x_idx]

    return data_sub_ds, lon_grid, lat_grid, times


def main():
    # example usage
    PATH = "./cache/era5/20200601_00.npz"
    VAR = "t2m"
    LON_MIN, LON_MAX = 121.00, 121.75
    LAT_MIN, LAT_MAX = 25.00, 25.75

    data_sub, lon_grid, lat_grid, times = extract_region(
        PATH,
        VAR,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
    )

    print("lat:\n", lat_grid)
    print("lon:\n", lon_grid)
    print(f"{VAR} slice shape:", data_sub.shape)

    data = load_npz(PATH)
    t2m = data[VAR]
    lat = data["lat"]
    lon = data["lon"]
    times = data["times"]
    print("Original data shapes:")
    print(t2m.shape, lat.shape, lon.shape, times.shape)

    t2m = data_sub  # assuming t2m is the variable of interest
    lat = lat_grid
    lon = lon_grid
    print("Extracted data shapes:")
    print(t2m.shape, lat.shape, lon.shape, times.shape)


if __name__ == "__main__":
    main()
