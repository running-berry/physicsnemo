import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_data(path, source):
    """
    Load NPZ data for ERA5 or RWRF.
    """
    data = np.load(path, allow_pickle=True)
    if source == 'era5':
        t2m = data['t2m']
        lat = data['lat']
        lon = data['lon']
        times = data['times']
    elif source == 'rwrf':
        t2m = data['t2m']
        lat = data['lat']
        lon = data['lon']
        times = data['times']
    else:
        raise ValueError(f"Unknown source: {source}")
    return t2m, lat, lon, times


def decode_time(raw, source):
    """
    Decode timestamp from array of bytes.
    """
    if source == 'era5':
        # raw may be numpy scalar or bytes
        try:
            time_str = str(raw)
        except Exception:
            time_str = raw.tobytes().decode('utf-8').strip()
    else:  # rwrf
        time_str = ''.join(s.decode('utf-8') for s in raw).strip()
    return time_str


def extract_slice(t2m, lat, lon, source):
    """
    Extract first time slice and corresponding lat/lon grid.
    """
    # convert K to C
    if source == 'era5':
        t2m_0 = t2m[0, 0, :, :]
        t2m_c = t2m_0 - 273.15
        lat_grid = lat
        lon_grid = lon
    else:  # rwrf
        t2m_0 = t2m[0, :, :]
        t2m_c = t2m_0 - 273.15
        lat_grid = lat[0, :, :]
        lon_grid = lon[0, :, :]
    return t2m_c, lat_grid, lon_grid


def plot_t2m(lon, lat, t2m_c, time_str, levels=20, figsize=(8,6)):
    """
    Create a filled contour plot of 2m temperature.
    """
    plt.figure(figsize=figsize)
    cs = plt.contourf(lon, lat, t2m_c, levels=levels, cmap='coolwarm')
    plt.colorbar(cs, label='T2 (°C)')
    plt.title(f"2 m Temp at {time_str}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot 2m temperature from NPZ file.')
    parser.add_argument('path', help='Path to NPZ file')
    parser.add_argument('--source', choices=['era5', 'rwrf'], required=True,
                        help='Data source type')
    args = parser.parse_args()

    t2m, lat, lon, times = load_data(args.path, args.source)
    raw = times[0]
    time_str = decode_time(raw, args.source)
    t2m_c, lat_grid, lon_grid = extract_slice(t2m, lat, lon, args.source)
    plot_t2m(lon_grid, lat_grid, t2m_c, time_str)


if __name__ == '__main__':
    main()
    # # Example usage:
    # python plot_var.py ./cache/era5/20200601_00.npz --source era5
