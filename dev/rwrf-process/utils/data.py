import numpy as np

def load_data(path, source, variable):
    """
    Load NPZ data for ERA5 or RWRF.
    """
    data = np.load(path, allow_pickle=True)
    if source == 'era5':
        if variable == 't2m':
            t2m = data['t2m']
        elif variable == 'u10':
            u10 = data['u10']
        lat = data['lat']
        lon = data['lon']
        times = data['times']
    elif source == 'rwrf':
        if variable == 't2m':
            t2m = data['t2m']
        elif variable == 'u10':
            u10 = data['u10']
        lat = data['lat']
        lon = data['lon']
        times = data['times']
    else:
        raise ValueError(f"Unknown source: {source}")

    if variable == 't2m':
        return t2m, lat, lon, times
    elif variable == 'u10':
        return u10, lat, lon, times


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
            # convert K to C
            t2m_0 = var_data[0, 0, :, :]
            t2m_c = t2m_0 - 273.15
        elif variable == 'u10':
            u10_0 = var_data[0, :, :]
        lat_grid = lat
        lon_grid = lon
    else:  # rwrf
        if variable == 't2m':
            # convert K to C
            t2m_0 = var_data[0, :, :]
            t2m_c = t2m_0 - 273.15
        elif variable == 'u10':
            u10_0 = var_data[0, :, :]
        lat_grid = lat[0, :, :]
        lon_grid = lon[0, :, :]
        
    if variable == 't2m':
        return t2m_c, lat_grid, lon_grid
    elif variable == 'u10':
        return u10_0, lat_grid, lon_grid

