import numpy as np
import xarray as xr


def create_metadata(
    variable: str | list[str] | np.ndarray[str],
    conditioning_variable: str | list[str] | np.ndarray[str],
    invariant: str | list[str] | np.ndarray[str],
    y: int = 32,
    x: int = 32,
    variable_file_path: str | None = None,
    conditioning_variable_file_path: str | None = None,
    invariant_file_path: str | None = None,
) -> xr.Dataset:
    """Creates a StormCast metadata xarray dataset with a structure matching the StormCast model's metadata.

    Parameters
    ----------
    variable : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to variables to return. Must be in the RWRF lexicon.
    conditioning_variable : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to conditioning variables to return. Must be in the RWRF lexicon.
    invariant : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to invariant to return. Must be in the RWRF lexicon.
    y : int
        The number of latitude grid points.
    x : int
        The number of longitude grid points.
    variable_file_path : str | None
        Path to the file containing the variable stds/means. If None, the variable stds/means will be randomly generated.
    conditioning_variable_file_path : str | None
        Path to the file containing the conditioning variable stds/means. If None, the conditioning variable stds/means will be randomly generated.
    invariant_file_path : str | None
        Path to the file containing the invariant data. If None, the invariant data will be randomly generated.

    Returns
    -------
    xarray.Dataset
        A Stormcast metadata xarray dataset

    """

    variable, conditioning_variable, invariant = prep_metadata_inputs(
        variable, conditioning_variable, invariant
    )

    dims = {
        "conditioning_variable": len(conditioning_variable),
        "invariant": len(invariant),
        "y": y,
        "x": x,
        "variable": len(variable),
    }

    lat, lon = load_lat_lon_data(
        file_path=invariant_file_path
    )  # load from invariants zarr

    coords = {
        "conditioning_variable": conditioning_variable,
        "invariant": invariant,
        "lat": (("y", "x"), lat),
        "lon": (("y", "x"), lon),
        "variable": variable,
        "x": np.arange(dims["x"]),
        "y": np.arange(dims["y"]),
    }

    variable_means, variable_stds = load_means_stds(file_path=variable_file_path)

    conditioning_means, conditioning_stds = load_means_stds(
        file_path=conditioning_variable_file_path
    )

    invariant_data = load_invariant_data(file_path=invariant_file_path)

    data_vars = {
        "conditioning_means": (
            ("conditioning_variable",),
            conditioning_means,
        ),
        "conditioning_stds": (
            ("conditioning_variable",),
            conditioning_stds,
        ),
        "invariants": (("invariant", "y", "x"), invariant_data),
        "means": (("variable",), variable_means),
        "stds": (("variable",), variable_stds),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds


def prep_metadata_inputs(
    variable: str | list[str] | np.ndarray[str],
    conditioning_variable: str | list[str] | np.ndarray[str],
    invariant: str | list[str] | np.ndarray[str],
) -> tuple[list[str], list[str], list[str]]:
    """Simple method to pre-process metadata inputs into a common form

    Parameters
    ----------
    variable : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to variables
    conditioning_variable : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to conditioning variables
    invariant : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to invariant variables

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Variable, conditioning variable, and invariant lists
    """
    if isinstance(variable, str):
        variable = [variable]

    if isinstance(conditioning_variable, str):
        conditioning_variable = [conditioning_variable]

    if isinstance(invariant, str):
        invariant = [invariant]

    return variable, conditioning_variable, invariant


def load_means_stds(
    dims: tuple[int, ...] | None = None, file_path: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simple method to load means and standard deviations from .npy files or generate random numpy arrays.

    Parameters
    ----------
    dims : tuple[int, ...] | None
        Dimensions of the array to be generated. If None, file_path must be provided to load data.
    file_path : str | None
        Path to the directory containing the means and stds .npy files. If None, random data will be generated, dims must be provided.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the means and standard deviations as numpy arrays.
    """
    if dims is None and file_path is None:
        raise ValueError("Either dims or file_path must be provided.")

    if file_path is not None:
        means = np.load(f"{file_path}/means.npy")
        stds = np.load(f"{file_path}/stds.npy")
    else:
        means = np.random.rand(*dims).astype(np.float32)
        stds = np.random.rand(*dims).astype(np.float32) + 0.1
    return means, stds


def load_invariant_data(
    dims: tuple[int, ...] | None = None, file_path: str | None = None
) -> np.ndarray:
    """Simple method to load invariant data from zarr files or generate random numpy arrays.

    Parameters
    ----------
    dims : tuple[int, ...] | None
        Dimensions of the array to be generated. If None, file_path must be provided to load data.
    file_path : str | None
        Path to the directory containing the invariant data files. If None, random data will be generated, dims must be provided.

    Returns
    -------
    np.ndarray
        Numpy array containing the invariant data.
    """
    if dims is None and file_path is None:
        raise ValueError("Either dims or file_path must be provided.")

    if file_path is not None:
        ds = xr.open_zarr(file_path)
        if "HighRes_invariants" not in ds:
            raise KeyError(
                f"'HighRes_invariants' not found in Zarr store at {file_path}"
            )
        return ds["HighRes_invariants"].values
    else:
        return np.random.rand(*dims).astype(np.float32)


def load_lat_lon_data(
    dims: tuple[int, ...] | None = None, file_path: str | None = None
) -> np.ndarray:
    """Simple method to load latitude and longitude data from zarr files or generate random numpy arrays.

    Parameters
    ----------
    dims : tuple[int, ...] | None
        Dimensions of the array to be generated. If None, file_path must be provided to load data.
    file_path : str | None
        Path to the directory containing the latitude/longitude data files. If None, random data will be generated, dims must be provided.

    Returns
    -------
    np.ndarray
        Numpy array containing the latitude and longitude data.
    """
    if dims is None and file_path is None:
        raise ValueError("Either dims or file_path must be provided.")

    if file_path is not None:
        ds = xr.open_zarr(file_path)
        if "latitude" not in ds or "longitude" not in ds:
            raise KeyError(
                f"'latitude' or 'longitude' not found in Zarr store at {file_path}"
            )
        return ds["latitude"].values, ds["longitude"].values
    else:
        return np.random.rand(*dims).astype(np.float32) * 90, np.random.rand(
            *dims
        ).astype(np.float32) * 180
