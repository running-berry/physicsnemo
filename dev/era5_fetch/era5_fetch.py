import os
import yaml
from datetime import datetime, timedelta
from earth2studio.data import CDS

# 1. Load all parameters from era.yaml
with open("era5.yaml", "r") as f:
    cfg = yaml.safe_load(f)

dataset_root = cfg["dataset_root"]
dataset_name = cfg["dataset_name"]
split        = cfg["split"]

# Parse ISO‐formatted timestamps into datetime objects
start_time = datetime.fromisoformat(cfg["start_time"])
end_time   = datetime.fromisoformat(cfg["end_time"])

variables = cfg["variables"]

# Create directory structure
output_dir = os.path.join(dataset_root, dataset_name, split)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the CDS object
cds = CDS(cache=True, verbose=True)

# Define time range (example: 24 hours on June 1st, 2020)
time_range = [
    start_time + timedelta(hours=i) for i in range(
            (end_time - start_time).seconds // 3600 + 1)
    ]

# Loop and save each time step as its own .nc file
for var in variables:
    for time in time_range:
        print(f"Fetching data for {time}...")
        data = cds(time=time, variable=var) 
        
        # Build filename from timestamp
        time_str = time.strftime("%Y%m%d_%H")
        filename = f"{var}_{time_str}.grib"
        filepath = os.path.join(output_dir, filename)
        
        # Save to NetCDF file
        data.to_netcdf(filepath)
        print(f"Saved to {filepath}")
