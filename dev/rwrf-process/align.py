from utils.data import load_data, decode_time, extract_slice
import numpy as np
import matplotlib.pyplot as plt
import argparse


def find_common_boundaries(era5_lat, era5_lon, rwrf_lat, rwrf_lon):
    """
    Find common lat/lon boundaries for both datasets.
    """

    lat_min = max(np.min(era5_lat), np.min(rwrf_lat))
    lat_max = min(np.max(era5_lat), np.max(rwrf_lat))
    lon_min = max(np.min(era5_lon), np.min(rwrf_lon))
    lon_max = min(np.max(era5_lon), np.max(rwrf_lon))

    return lat_min, lat_max, lon_min, lon_max


def plot_field(
    lon,
    lat,
    data,
    time_str,
    variable,
    source,
    boundaries,
    ax,
    levels=20
):
    """
    Create a countour plot for u10 or t2m within specified boundaries.
    """
    lat_min, lat_max, lon_min, lon_max = boundaries

    if source == "era5":
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        data_filtered = data[lat_mask, :][:, lon_mask]
        lat_filtered = lat[lat_mask]
        lon_filtered = lon[lon_mask]
    else:
        data_filtered = data
        lat_filtered = lat
        lon_filtered = lon

    cmap = "coolwarm" if variable == "t2m" else plt.cm.jet
    label = "T2 (°C)" if variable == "t2m" else "u10 (m/s)"
    if variable == "t2m":
        title = f"{source.upper()} 2m Temp at {time_str}"
    elif variable == "u10":
        title = f"{source.upper()} 10m u-component of wind at {time_str}"

    cs = ax.contourf(lon_filtered, lat_filtered, data_filtered, levels=levels, cmap=cmap)
    plt.colorbar(cs, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

def main():
    parser = argparse.ArgumentParser(
        description="Plot and compare 2m temperature from ERA5 and RWRF."
    )
    parser.add_argument("--era5_path", help="Path to ERA5 NPZ file", required=True)
    parser.add_argument("--rwrf_path", help="Path to RWRF NPZ file", required=True)
    parser.add_argument('--variable', choices=['t2m', 'u10'], required=True,help='Variable to plot')
    args = parser.parse_args()

    # Load both datasets
    era5_u10, era5_lat, era5_lon, era5_times = load_data(args.era5_path, "era5", args.variable)
    rwrf_u10, rwrf_lat, rwrf_lon, rwrf_times = load_data(args.rwrf_path, "rwrf", args.variable)

    # Get time strings
    era5_time_str = decode_time(era5_times[0], "era5", args.variable)
    rwrf_time_str = decode_time(rwrf_times[0], "rwrf", args.variable)

    # Extract data slices
    era5_extracted_data, era5_lat_grid, era5_lon_grid = extract_slice(
        era5_u10, era5_lat, era5_lon, "era5", args.variable
    )
    rwrf_extracted_data, rwrf_lat_grid, rwrf_lon_grid = extract_slice(
        rwrf_u10, rwrf_lat, rwrf_lon, "rwrf", args.variable
    )

    # Find common boundaries
    boundaries = find_common_boundaries(
        era5_lat_grid, era5_lon_grid, rwrf_lat_grid, rwrf_lon_grid
    )


    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_field(
        era5_lon_grid, era5_lat_grid, era5_extracted_data, era5_time_str,
        variable=args.variable, source="era5", boundaries=boundaries, ax=ax1
    )
    plot_field(
        rwrf_lon_grid, rwrf_lat_grid, rwrf_extracted_data, rwrf_time_str,
        variable=args.variable, source="rwrf", boundaries=boundaries, ax=ax2
    )
    plt.tight_layout()
    plt.savefig(f"align_{args.variable}.png")
    plt.show()


if __name__ == "__main__":
    main()
    # # Example usage:
    # python align.py --era5_path ./cache/era5/t2m_20200601_00.npz --rwrf_path ./cache/rwrf/t2m_20190803_00.npz --variable t2m
