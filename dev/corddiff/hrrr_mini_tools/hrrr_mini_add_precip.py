#!/usr/bin/env python3
"""
add_dummy_precip.py

Copy a NetCDF file and inject a synthetic 'precipitation' DataArray
into the existing 'output' group using the netCDF4 low-level API.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import netCDF4 as nc

def main():
    p = argparse.ArgumentParser(
        description="Copy a .nc and append dummy precipitation into 'output' group."
    )
    p.add_argument("infile", help="Path to input NetCDF (must have group 'output').")
    p.add_argument(
        "-o", "--out",
        help="Output file (defaults to infile + '_dummy_precipitation.nc')."
    )
    p.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed for precipitation."
    )
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        p.error(f"Input file not found: {infile}")

    # Determine output path
    if args.out:
        outfile = Path(args.out)
    else:
        outfile = infile.with_name(infile.stem + '_dummy_precipitation.nc')

    # 1) Copy entire file
    shutil.copy2(infile, outfile)
    print(f"Copied {infile} → {outfile}")

    # 2) Open with netCDF4 in r+ mode
    ds = nc.Dataset(outfile, mode='r+')

    # 3) Navigate to the 'output' group
    if 'output' not in ds.groups:
        ds.close()
        raise RuntimeError("Group 'output' not found in the copied file")
    outgrp = ds.groups['output']

    # 4) Determine dims existing in output group
    #    We expect dims ('sample', 'y_hr' or 'y', 'x_hr' or 'x')
    dims = outgrp.dimensions
    sample_dim = 'sample'
    if sample_dim not in dims:
        ds.close()
        raise RuntimeError("'sample' dimension missing in 'output' group")
    # pick either high-res or low-res
    ydim = 'y_hr' if 'y_hr' in dims else 'y'
    xdim = 'x_hr' if 'x_hr' in dims else 'x'
    ny, nx = len(dims[ydim]), len(dims[xdim])
    ns = len(dims[sample_dim])

    # 5) Create the precipitation variable
    if 'precipitation' in outgrp.variables:
        print("Warning: 'precipitation' already exists, it will be overwritten")
        var = outgrp.variables['precipitation']
    else:
        var = outgrp.createVariable(
            'precipitation',
            'f4',
            (sample_dim, ydim, xdim),
            zlib=True
        )

    # 6) Generate and write data
    np.random.seed(args.seed)
    data = (np.random.rand(ns, ny, nx) * 50).astype('float32')
    var[:] = data
    var.units = 'mm'             # optional metadata
    var.long_name = 'Dummy precip'

    # 7) Close
    ds.close()
    print(f"✅ Appended 'precipitation' ({ns}×{ny}×{nx}) to group 'output' in {outfile}")

if __name__ == '__main__':
    main()