#!/usr/bin/env python3
import argparse
import os
import json

import xarray as xr
import matplotlib.pyplot as plt


def main(nc_path: str, output_dir: str):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open the NetCDF groups
    ds_input = xr.open_dataset(nc_path, group='input', engine='netcdf4')
    ds_truth = xr.open_dataset(nc_path, group='truth', engine='netcdf4')
    ds_pred  = xr.open_dataset(nc_path, group='prediction', engine='netcdf4')

    # Gather dimensions and variable metadata
    metadata = {
        'input': {
            'dims': dict(ds_input.dims),
            'variables': list(ds_input.data_vars),
        },
        'truth': {
            'dims': dict(ds_truth.dims),
            'variables': list(ds_truth.data_vars),
        },
        'prediction': {
            'dims': dict(ds_pred.dims),
            'variables': list(ds_pred.data_vars),
        },
    }

    # Dump metadata to JSON
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as jf:
        json.dump(metadata, jf, indent=2)
    print(f"Wrote metadata to {meta_path}")

    # Plot each prediction variable at ensemble=0, time=0
    for var in ds_pred.data_vars:
        da = ds_pred[var].isel(ensemble=0, time=0)
        fig, ax = plt.subplots()
        da.plot(ax=ax)
        ax.set_title(f'Prediction: {var} (ensemble=0, time=0)')
        img_path = os.path.join(output_dir, f'prediction_{var}.png')
        fig.savefig(img_path)
        plt.close(fig)
        print(f"Saved plot for {var} to {img_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dump metadata and plot prediction variables from a NetCDF file.'
    )
    parser.add_argument(
        'nc_file', metavar='NC_FILE', help='Path to the NetCDF file'
    )
    parser.add_argument(
        'output_dir', metavar='OUT_DIR', help='Directory to save JSON and plots'
    )
    args = parser.parse_args()
    main(args.nc_file, args.output_dir)