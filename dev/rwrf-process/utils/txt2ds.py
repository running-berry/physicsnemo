import numpy as np
import os
import xarray as xr

def load_qpepre(txt_path):
    """
    Load a whitespace-delimited QPEPRE text file into an xarray.Dataset
    with a 1D “point” dimension named ‘point’.
    """
    # define fields
    field_defs = [
        ("idx",    np.int32),
        ("lat",    np.float64),
        ("lon",    np.float64),
        ("qpepre", np.float64),
    ]
    # read into structured array
    data = np.genfromtxt(
        txt_path,
        dtype=field_defs,
        delimiter=None,
        autostrip=True,
        comments=None,
    )
    # build Dataset
    ds = xr.Dataset(
        {"qpepre": ("point", data["qpepre"])},
        coords={
            "idx": ("point", data["idx"]),
            "lat": ("point", data["lat"]),
            "lon": ("point", data["lon"]),
        }
    )
    return ds

def qpepre_stats(ds):
    """
    Load the QPEPRE file and compute mean & std of the ‘qpepre’ variable.
    Returns (mean, std) as Python scalars.
    """
    mean_qpe = ds["qpepre"].mean(dim="point").item()
    std_qpe  = ds["qpepre"].std(dim="point").item()
    return mean_qpe, std_qpe
