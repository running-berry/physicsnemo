import numpy as np

def load_data(path, source, variable):
    """
    Load NPZ data for ERA5 or RWRF.
    """
    if source not in ['era5', 'rwrf']:
        raise ValueError(f"Unknown source: {source}")
    if variable not in ['t2m', 'u10']:
        raise ValueError(f"Unknown variable: {variable}")
    
    data = np.load(path, allow_pickle=True)
    var_data = data[variable]
    lat = data['lat']
    lon = data['lon']
    times = data['times']
    return var_data, lat, lon, times


def decode_time(raw, source, variable):
    """
    Decode timestamp from array of bytes.
    """
    if source == 'era5':
        if variable == 't2m':
            # raw may be numpy scalar or bytes
            try:
                time_str = str(raw)
            except Exception:
                time_str = raw.tobytes().decode('utf-8').strip()
        elif variable == 'u10':
            from datetime import datetime, timezone

            dt = datetime.fromtimestamp(raw, tz=timezone.utc)
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    else:  # rwrf
        time_str = ''.join(s.decode('utf-8') for s in raw).strip()
    return time_str


def extract_slice(var_data, lat, lon, source, variable):
    """
    Extract first time slice and corresponding lat/lon grid.
    """
    if source == 'era5':
        if variable == 't2m':
            data_slice = var_data[0, 0, :, :] - 273.15  # K to C
        elif variable == 'u10':
            data_slice = var_data[0, :, :]
        lat_grid = lat
        lon_grid = lon
    else:  # rwrf
        data_slice = var_data[0, :, :]
        if variable == 't2m':
            data_slice = data_slice - 273.15  # K to C
        lat_grid = lat[0, :, :]
        lon_grid = lon[0, :, :]
    return data_slice, lat_grid, lon_grid

