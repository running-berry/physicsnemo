import os
import yaml
import numpy as np
from earth2studio.data import CDS
from datetime import datetime, timedelta
import time as time_module
import shutil
import xarray as xr

def era5_download(cfg):
    dataset_root = cfg["dataset_root"]
    dataset_name = cfg["dataset_name"]
    subdir = cfg["subdir"]
    
    start_time = datetime.fromisoformat(cfg["start_time"])
    end_time = datetime.fromisoformat(cfg["end_time"])
    
    variables = cfg["variables"]
    save_stats = cfg.get("save_stats", False)
    stats_dir = cfg.get("stats_dir", "stats")
    
    cds = CDS(cache=True, verbose=True)

    # Include end_time 
    total_hours = int((end_time - start_time).total_seconds() // 3600)
    time_range = [start_time + timedelta(hours=i) for i in range(total_hours + 1)]
    
    """
    If you want to exclude end_time use this:
    
    time_range = [start_time + timedelta(hours=i) for i in range(total_hours)]
    """

    output_dir = os.path.join(dataset_root, dataset_name, subdir)
    os.makedirs(output_dir, exist_ok=True)

    all_data = [] 
    
    for var in variables:
        for time in time_range:           
            # Retry mechanism to improve download stability:
            # Sometimes downloading from CDS fails due to network issues or corrupted cache.
            # This loop will retry the download up to 3 times, waiting 10 seconds between attempts.
            # If a cache corruption is suspected, it clears the local cache to force a clean retry.
            max_retries = 3
            data = None
            for attempt in range(max_retries):
                try:
                    data = cds(time=time, variable=var)
                    break
                except (EOFError, Exception) as e:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in 10s")
                        time_module.sleep(10)
                        # clear cache
                        cache_dir = os.path.expanduser("~/.cache/earth2studio")
                        if os.path.exists(cache_dir):
                            try:
                                shutil.rmtree(cache_dir)
                            except:
                                pass
                    else:
                        break
            
            if data is None:
                continue
                
            time_str = time.strftime("%Y%m%d_%H")
            filename = f"{var}_{time_str}.nc"
            filepath = os.path.join(output_dir, filename)
            
            try:
                data.to_netcdf(filepath)
                if save_stats:
                    all_data.append(data)
            except Exception as e:
                print(f"Error saving {filepath}: {e}")
                continue



    if save_stats and all_data:
        combined = xr.concat(all_data, dim="time")

        mean = combined.mean().values
        std = combined.std().values

        stats_output_dir = os.path.join(dataset_root, dataset_name, stats_dir)
        os.makedirs(stats_output_dir, exist_ok=True)

        np.save(os.path.join(stats_output_dir, "means.npy"), mean)
        np.save(os.path.join(stats_output_dir, "stds.npy"), std)

