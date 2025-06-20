import sys
import argparse

import xarray as xr
import numpy as np

def diff_global_attrs(ds1, ds2):
    """Return list of (attr, val1, val2) for any global-attribute mismatches."""
    diffs = []
    keys = set(ds1.attrs) | set(ds2.attrs)
    for k in sorted(keys):
        v1 = ds1.attrs.get(k, None)
        v2 = ds2.attrs.get(k, None)
        if v1 != v2:
            diffs.append((k, v1, v2))
    return diffs

def diff_dimensions(ds1, ds2):
    """Return list of (dim, len1, len2) for any size mismatches."""
    diffs = []
    keys = set(ds1.sizes) | set(ds2.sizes)
    for k in sorted(keys):
        l1 = ds1.sizes.get(k, None)
        l2 = ds2.sizes.get(k, None)
        if l1 != l2:
            diffs.append((k, l1, l2))
    return diffs

def diff_var_attrs(da1, da2):
    """Return list of (attr, val1, val2) for mismatches in a DataArray's attrs."""
    diffs = []
    keys = set(da1.attrs) | set(da2.attrs)
    for k in sorted(keys):
        a1 = da1.attrs.get(k, None)
        a2 = da2.attrs.get(k, None)
        if a1 != a2:
            diffs.append((k, a1, a2))
    return diffs

def diff_data_values(da1, da2, rtol=1e-7, atol=0.0, max_report=5):
    """
    For numeric arrays, find indices where values differ.
    Returns (dtype, total_mismatches, sample_indices).
    """
    arr1 = da1.values
    arr2 = da2.values

    # floating types: use isclose
    if np.issubdtype(arr1.dtype, np.floating):
        mask = ~np.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True)
    else:
        mask = arr1 != arr2

    # flatten mask to find mismatches
    idxs = np.argwhere(mask)
    total = idxs.shape[0]
    sample = [tuple(idx) for idx in idxs[:max_report]]
    return arr1.dtype, total, sample

def compare_netcdf_detailed(f1, f2, rtol=1e-7, atol=0.0):
    ds1 = xr.open_dataset(f1)
    ds2 = xr.open_dataset(f2)
    any_diff = False

    # 1. Global attrs
    gdiff = diff_global_attrs(ds1, ds2)
    if gdiff:
        any_diff = True
        print("Global attribute differences:")
        for k, v1, v2 in gdiff:
            print(f"  • {k}: {v1!r} != {v2!r}")
    else:
        print("Global attributes: identical.")

    # 2. Dimensions
    ddiff = diff_dimensions(ds1, ds2)
    if ddiff:
        any_diff = True
        print("Dimension size differences:")
        for dim, l1, l2 in ddiff:
            print(f"  • {dim}: {l1} != {l2}")
    else:
        print("Dimensions: identical.")

    # 3. Variables
    vars_union = sorted(set(ds1.data_vars) | set(ds2.data_vars))
    for var in vars_union:
        if var not in ds1 or var not in ds2:
            any_diff = True
            print(f"Variable missing: {var}")
        else:
            print(f"Comparing variable: {var}")
            print(f"  - {var} found in both files")

        da1, da2 = ds1[var], ds2[var]

        # 3a. Attrs
        av = diff_var_attrs(da1, da2)
        if av:
            any_diff = True
            print(f"Attribute mismatches in variable {var!r}:")
            for k, a1, a2 in av:
                print(f"  • {k}: {a1!r} != {a2!r}")
        else:
            print(f"Attributes for {var!r}: identical.")

        # 3b. Shape check
        if da1.shape != da2.shape:
            any_diff = True
            print(f"Shape mismatch for {var!r}: {da1.shape} != {da2.shape}")
        else:
            print(f"Shape for {var!r}: identical ({da1.shape})")

        # 3c. Data values
        dtype, count, samples = diff_data_values(da1, da2, rtol, atol)
        if count:
            any_diff = True
            print(f"Value mismatches in {var!r} (dtype={dtype}): {count} cells differ")
            print("  sample indices:", samples)
        else:
            print(f"Values for {var!r}: identical (dtype={dtype})")

    ds1.close()
    ds2.close()

    if not any_diff:
        print("Files are identical.")
        return True

    return False

def main():
    p = argparse.ArgumentParser(
        description="Compare two NetCDF files for identical contents."
    )
    p.add_argument("file1", help="First NetCDF file")
    p.add_argument("file2", help="Second NetCDF file")
    p.add_argument("--rtol", type=float, default=1e-7,
                   help="Relative tolerance for floating comparisons")
    p.add_argument("--atol", type=float, default=0.0,
                   help="Absolute tolerance for floating comparisons")
    args = p.parse_args()

    identical = compare_netcdf_detailed(args.file1, args.file2, args.rtol, args.atol)
    if identical:
        print("Result: Files are identical.")
        sys.exit(0)
    else:
        print("Result: Files differ.")
        sys.exit(1)

if __name__ == "__main__":
    main()
