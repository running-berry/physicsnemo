import xarray as xr
import numpy as np

base_zarr_data = "./"
base_zarr_dummy = "./zarr"

highres_data = xr.open_zarr(f"{base_zarr_data}/DummyHighRes/2019.zarr", consolidated=True)
lowres_data = xr.open_zarr(f"{base_zarr_data}/DummyLowRes/2019.zarr", consolidated=True)
highres_dummy = xr.open_zarr(f"{base_zarr_dummy}/DummyHighRes/2019.zarr", consolidated=True)
lowres_dummy = xr.open_zarr(f"{base_zarr_dummy}/DummyLowRes/2019.zarr", consolidated=True)

def show_coord_examples(ds1, ds2):
  print("Coordinates:")
  for coord in ds1.coords:
    print(f"\n-- {coord} --")
    v1 = ds1[coord].values

    if coord not in ds2.coords:
      print("  Missing in dummy dataset")
      preview1 = v1.flatten()[:10]
      print(f"  ds1[{coord}] (dtype: {v1.dtype}, shape: {v1.shape}): {preview1}")
      continue

    v2 = ds2[coord].values
    preview1 = v1.flatten()[:10]
    preview2 = v2.flatten()[:10]
    print(f"  ds1[{coord}] (dtype: {v1.dtype}, shape: {v1.shape}): {preview1}")
    print(f"  ds2[{coord}] (dtype: {v2.dtype}, shape: {v2.shape}): {preview2}")

    if np.issubdtype(v1.dtype, np.number) and np.issubdtype(v2.dtype, np.number):
      diff = np.abs(v1 - v2).max()
      print(f"  Numeric comparison: max difference = {diff}")
    else:
      equal = np.array_equal(v1.astype(str), v2.astype(str))
      status = "equal" if equal else "not equal"
      print(f"  Non-numeric comparison: string arrays are {status}")

def compare_datasets(name, ds1, ds2):
    print(f"\n=== Comparing {name} ===")
    
    # Compare dimensions
    print("Dimensions:")
    for dim in ds1.sizes:
      shape1 = ds1.sizes[dim]
      shape2 = ds2.sizes.get(dim, None)
      print(f"  {dim}: data={shape1}, dummy={shape2}")
    
    # Compare coordinates
    print("\nCoordinates:")
    for coord in ds1.coords:
      if coord not in ds2.coords:
        print(f"  {coord}: missing in dummy")
        continue

      try:
        v1 = ds1[coord].values
        v2 = ds2[coord].values

        if np.issubdtype(v1.dtype, np.number) and np.issubdtype(v2.dtype, np.number):
          diff = np.abs(v1 - v2).max()
          print(f"  {coord}: max diff = {diff}")
        else:
          # fallback to string comparison
          equal = np.array_equal(v1.astype(str), v2.astype(str))
          status = "equal" if equal else "not equal"
          print(f"  {coord}: non-numeric, string arrays are {status}")
      except Exception as e:
        print(f"  {coord}: comparison failed – {e}")   
    #$$ 
    # Compare variables
    print("\nVariables:")
    for var in ds1.data_vars:
      if var in ds2:
        v1 = ds1[var].values
        v2 = ds2[var].values

        if v1.shape != v2.shape:
          print(f"  {var}: shape mismatch {v1.shape} vs {v2.shape}")
        else:
          mean_diff = np.abs(v1 - v2).mean()
          print(f"  {var}: mean diff = {mean_diff:.4f}")
      else:
        print(f"  {var}: missing in dummy")


print(f"ds1: dummy, ds2: data")
compare_datasets("HighRes", highres_dummy, highres_data)
compare_datasets("LowRes", lowres_dummy, lowres_data)
show_coord_examples(highres_dummy, highres_data)
show_coord_examples(lowres_dummy, lowres_data)