from utils.data import load_data, decode_time, extract_slice
import matplotlib.pyplot as plt
import argparse


def plot_var(lon, lat, field, time_str, source, variable, levels=20, figsize=(8, 6)):
    """
    Create a filled contour plot for a field (t2m or u10).
    """
    plt.figure(figsize=figsize)
    if variable == 't2m':
        cmap = 'coolwarm'
        label = 'T2 (°C)'
        title = f"2 m Temp at {time_str}"
        img_path = f't2m_{source}_{time_str}.png'
    elif variable == 'u10':
        cmap = plt.cm.jet
        label = 'u10 (m/s)'
        title = f"10m u-component of wind at {time_str}"
        img_path = f'u10_{source}_{time_str}.png'
    
    cs = plt.contourf(lon, lat, field, levels=levels, cmap=cmap)
    plt.colorbar(cs, label=label)
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot 2m temperature from NPZ file.')
    parser.add_argument('path', help='Path to NPZ file')
    parser.add_argument('--source', choices=['era5', 'rwrf'], required=True,
                        help='Data source type')
    parser.add_argument('--variable', choices=['t2m', 'u10'], required=True,
                        help='Variable to plot')
    args = parser.parse_args()

    data, lat, lon, times = load_data(args.path, args.source, args.variable)
    raw = times[0]
    time_str = decode_time(raw, args.source, args.variable)
    extracted_data, lat_grid, lon_grid = extract_slice(data, lat, lon, args.source, args.variable)
    
    plot_var(lon_grid, lat_grid, extracted_data, time_str, args.source, args.variable)


if __name__ == '__main__':
    main()
    # # Example usage:
    # python plot_var.py ./cache/era5/t2m_20200601_00.npz --source era5 --variable t2m
