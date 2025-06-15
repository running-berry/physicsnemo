from utils.data import load_data, decode_time, extract_slice
import matplotlib.pyplot as plt
import argparse


def plot_t2m(lon, lat, t2m_c, time_str, source, levels=20, figsize=(8,6)):
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
    plt.savefig(f't2m_{source}_{time_str}.png')
    plt.show()
    
def plot_u10(lon, lat, u10_0, time_str, source, levels=20, figsize=(8,6)):
    """
    Create a filled contour plot of 10m u-component of wind.
    """
    plt.figure(figsize=figsize)
    cs = plt.contourf(lon, lat, u10_0, levels=levels, cmap=plt.cm.jet)
    plt.colorbar(cs, label='u10 (m/s)')
    plt.title(f"10m u-component of wind at {time_str}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f'u10_{source}_{time_str}.png')
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
    
    if args.variable == 't2m':
        plot_t2m(lon_grid, lat_grid, extracted_data, time_str, args.source)
    elif args.variable == 'u10':
        plot_u10(lon_grid, lat_grid, extracted_data, time_str, args.source)


if __name__ == '__main__':
    main()
    # # Example usage:
    # python plot_var.py ./cache/era5/t2m_20200601_00.npz --source era5 --variable t2m
