# util_extract.py
import numpy as np
from typing import Tuple, Optional
try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def load_npz(path: str, mmap: bool = True) -> dict:
    """Load NPZ, optionally memmap arrays to reduce peak RAM."""
    # np.load(..., mmap_mode='r') works for .npz (members become memmaps)
    return np.load(path, allow_pickle=True, mmap_mode="r" if mmap else None)


def _ensure_1d_coords(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Accept (Ny,Nx) grids or (1,Ny,Nx) or 1D; return 1D (Ny,), (Nx,).
    """
    if lat.ndim == 3:
        lat_1d = lat[0, :, 0]
    elif lat.ndim == 2:
        lat_1d = lat[:, 0]
    else:
        lat_1d = lat

    if lon.ndim == 3:
        lon_1d = lon[0, 0, :]
    elif lon.ndim == 2:
        lon_1d = lon[0, :]
    else:
        lon_1d = lon

    return np.asarray(lat_1d), np.asarray(lon_1d)


def _search_bounds_1d(arr: np.ndarray, vmin: float, vmax: float) -> tuple[int, int]:
    """
    Fast bounds on monotonic 1D arrays. Works if arr asc or desc.
    Returns inclusive indices [i0, i1].
    """
    if arr[0] <= arr[-1]:  # ascending
        i0 = int(np.searchsorted(arr, vmin, side="left"))
        i1 = int(np.searchsorted(arr, vmax, side="right") - 1)
    else:  # descending
        arr_rev = arr[::-1]
        j0 = int(np.searchsorted(arr_rev, vmax, side="left"))
        j1 = int(np.searchsorted(arr_rev, vmin, side="right") - 1)
        # map back
        n = arr.size
        i0 = n - 1 - j1
        i1 = n - 1 - j0

    i0 = max(0, min(i0, arr.size - 1))
    i1 = max(0, min(i1, arr.size - 1))
    if i0 > i1:
        i0, i1 = i1, i0
    return i0, i1


def get_equidistant_indices(length: int, target: int) -> np.ndarray:
    """
    Return `target` indices evenly spaced over [0, length-1].
    """
    if target >= length:
        return np.arange(length, dtype=np.int64)
    idx = np.round(np.linspace(0, length - 1, target)).astype(np.int64)
    # Guarantee strictly non-decreasing unique sequence
    return np.unique(idx)


def interp_to_domain(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    data: Optional[np.ndarray],
    domain_size: Tuple[int, int],
    method: str = "linear",
    out_dtype: Optional[np.dtype] = np.float32,
):
    """
    Resample a rectilinear grid (and optional data) to (Ny, Nx).
    Uses SciPy's RegularGridInterpolator only when needed.
    """
    Ny, Nx = domain_size
    Ny_full, Nx_full = lat_grid.shape

    lat1d = lat_grid[:, 0]
    lon1d = lon_grid[0, :]

    # build target coords
    lat_new_1d = np.linspace(lat1d.min(), lat1d.max(), Ny)
    lon_new_1d = np.linspace(lon1d.min(), lon1d.max(), Nx)
    lon_new, lat_new = np.meshgrid(lon_new_1d, lat_new_1d)

    if data is None:
        return lon_new.astype(out_dtype), lat_new.astype(out_dtype)

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available for interpolation; install scipy or use downsample mode.")

    pts_new = np.column_stack((lat_new.ravel(), lon_new.ravel()))
    out_shape = data.shape[:-2] + (Ny, Nx)
    data_new = np.empty(out_shape, dtype=out_dtype or data.dtype)

    def _interp_slice(arr2d):
        interp = RegularGridInterpolator(
            (lat1d, lon1d),
            arr2d,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        return interp(pts_new).reshape(Ny, Nx)

    if data.ndim == 2:
        data_new[...] = _interp_slice(np.asarray(data))
    else:
        # iterate only over leading dims; NumPy ndindex is fast in C
        for idx in np.ndindex(*data.shape[:-2]):
            data_new[idx] = _interp_slice(np.asarray(data[idx]))

    return data_new, lon_new.astype(out_dtype), lat_new.astype(out_dtype)


def extract_region(
    path: str,
    var: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    domain_size: Optional[Tuple[int, int]] = None,
    resample: str = "downsample",  # "downsample" (fast) or "interp"
    cast_float32: bool = True,
):
    """
    Extract a lat/lon window from an NPZ and optionally resize to (Ny, Nx).

    Returns:
      data_sub, lon_grid, lat_grid, times
        - data_sub: (..., Ny, Nx)
        - lon_grid, lat_grid: (Ny, Nx)
    """
    npz = load_npz(path)
    arr  = npz[var]   # (..., y, x)
    lat  = npz["lat"]
    lon  = npz["lon"]
    times = npz["times"]

    # 1D coordinates
    lat_1d, lon_1d = _ensure_1d_coords(np.asarray(lat), np.asarray(lon))

    # FAST bounds
    i0, i1 = _search_bounds_1d(lat_1d, lat_min, lat_max)
    j0, j1 = _search_bounds_1d(lon_1d, lon_min, lon_max)

    # slice data
    idx = [slice(None)] * arr.ndim
    idx[-2] = slice(i0, i1 + 1)
    idx[-1] = slice(j0, j1 + 1)
    data_sub = arr[tuple(idx)]

    # lazily build grids only once
    lat_slice = lat_1d[i0 : i1 + 1]
    lon_slice = lon_1d[j0 : j1 + 1]

    # If no resize requested, return the exact window
    if domain_size is None:
        lon_grid, lat_grid = np.meshgrid(lon_slice, lat_slice)
        if cast_float32 and data_sub.dtype != np.float32:
            data_sub = data_sub.astype(np.float32, copy=False)
        return data_sub, lon_grid.astype(np.float32), lat_grid.astype(np.float32), times

    H, W = domain_size

    if resample == "downsample":
        # Fast path: uniform downsampling by index selection (no heavy math)
        yi = get_equidistant_indices(lat_slice.size, H)
        xi = get_equidistant_indices(lon_slice.size, W)

        # grids
        lat_grid = lat_slice[yi][:, None].repeat(xi.size, axis=1)
        lon_grid = lon_slice[None, xi].repeat(yi.size, axis=0)

        # data (..., y, x) → (..., H, W)
        data_ds = data_sub[..., yi, :][..., :, xi]
        if cast_float32 and data_ds.dtype != np.float32:
            data_ds = data_ds.astype(np.float32, copy=False)
        return data_ds, lon_grid.astype(np.float32), lat_grid.astype(np.float32), times

    elif resample == "interp":
        # Exact resize to (H, W) via interpolation
        lon_grid_full, lat_grid_full = np.meshgrid(lon_slice, lat_slice)
        out = interp_to_domain(
            lon_grid_full,
            lat_grid_full,
            np.asarray(data_sub),
            (H, W),
            method="linear",
            out_dtype=np.float32 if cast_float32 else None,
        )
        return out[0], out[1], out[2], times

    else:
        raise ValueError("resample must be 'downsample' or 'interp'.")


# -------------------- Example -------------------- #
if __name__ == "__main__":
    PATH = "./cache/era5/20200601_00.npz"
    VAR = "t2m"
    LON_MIN, LON_MAX = 121.00, 121.75
    LAT_MIN, LAT_MAX = 25.00, 25.75

    data_sub, lon_grid, lat_grid, times = extract_region(
        PATH, VAR, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, domain_size=None
    )
    print(VAR, "slice:", data_sub.shape, lon_grid.shape, lat_grid.shape)

    # Resize quickly by downsampling (fast)
    data_ds, lon_ds, lat_ds, _ = extract_region(
        PATH, VAR, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, domain_size=(128, 128), resample="downsample"
    )
    print("Downsampled:", data_ds.shape)

    # Or, resize by interpolation (slower, smoother)
    try:
        data_ip, lon_ip, lat_ip, _ = extract_region(
            PATH, VAR, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, domain_size=(128, 128), resample="interp"
        )
        print("Interpolated:", data_ip.shape)
    except RuntimeError as e:
        print("Interpolation unavailable:", e)