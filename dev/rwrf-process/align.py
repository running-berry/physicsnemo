from utils.data import load_data, decode_time, extract_slice
import numpy as np
import matplotlib.pyplot as plt
import argparse


def find_common_boundaries(era5_lat, era5_lon, rwrf_lat, rwrf_lon):
    """
    Find common lat/lon boundaries for both datasets.
    """
    # For ERA5
    era5_lat_min, era5_lat_max = np.min(era5_lat), np.max(era5_lat)
    era5_lon_min, era5_lon_max = np.min(era5_lon), np.max(era5_lon)

    # For RWRF
    rwrf_lat_min, rwrf_lat_max = np.min(rwrf_lat), np.max(rwrf_lat)
    rwrf_lon_min, rwrf_lon_max = np.min(rwrf_lon), np.max(rwrf_lon)

    lat_min = max(era5_lat_min, rwrf_lat_min)
    lat_max = min(era5_lat_max, rwrf_lat_max)
    lon_min = max(era5_lon_min, rwrf_lon_min)
    lon_max = min(era5_lon_max, rwrf_lon_max)

    return lat_min, lat_max, lon_min, lon_max


def plot_u10(
    lon,
    lat,
    u10,
    time_str,
    levels=20,
    source="era5",
    boundaries=None,
    subplot=None,
):
    """
    Create a contour plot of 10m u-component of wind with boundaries.
    """
    ax = subplot

    lat_min, lat_max, lon_min, lon_max = boundaries

    if source == "era5":
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        u10_filtered = u10[lat_mask, :][:, lon_mask]
        lat_filtered = lat[lat_mask]
        lon_filtered = lon[lon_mask]
    elif source == "rwrf":
        u10_filtered = u10
        lat_filtered = lat
        lon_filtered = lon

    cs = ax.contourf(
        lon_filtered, lat_filtered, u10_filtered, levels=levels, cmap=plt.cm.jet
    )
    plt.colorbar(cs, ax=ax, label="u10 (m/s)")
    ax.set_title(f"{source.upper()} 10m u-component of wind at  {time_str}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_t2m(
    lon,
    lat,
    t2m_c,
    time_str,
    levels=20,
    source="era5",
    boundaries=None,
    subplot=None,
):
    """
    Create a contour plot of 2m temperature with boundaries.
    """
    ax = subplot

    lat_min, lat_max, lon_min, lon_max = boundaries

    if source == "era5":
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        t2m_filtered = t2m_c[lat_mask, :][:, lon_mask]
        lat_filtered = lat[lat_mask]
        lon_filtered = lon[lon_mask]
    elif source == "rwrf":
        t2m_filtered = t2m_c
        lat_filtered = lat
        lon_filtered = lon

    cs = ax.contourf(
        lon_filtered, lat_filtered, t2m_filtered, levels=levels, cmap="coolwarm"
    )
    plt.colorbar(cs, ax=ax, label="T2 (°C)")
    ax.set_title(f"{source.upper()} 2m Temp at {time_str}")
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
    lat_min, lat_max, lon_min, lon_max = find_common_boundaries(
        era5_lat_grid, era5_lon_grid, rwrf_lat_grid, rwrf_lon_grid
    )
    common_boundaries = (lat_min, lat_max, lon_min, lon_max)

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    if args.variable == 't2m':
        plot_t2m(
            era5_lon_grid,
            era5_lat_grid,
            era5_extracted_data,
            era5_time_str,
            source="era5",
            boundaries=common_boundaries,
            subplot=ax1,
        )
        plot_t2m(
            rwrf_lon_grid,
            rwrf_lat_grid,
            rwrf_extracted_data,
            rwrf_time_str,
            source="rwrf",
            boundaries=common_boundaries,
            subplot=ax2,
        )
        plt.tight_layout()
        plt.savefig("comparison_t2m.png")
        plt.show()
    elif args.variable == 'u10':
        plot_u10(
            era5_lon_grid,
            era5_lat_grid,
            era5_extracted_data,
            era5_time_str,
            source="era5",
            boundaries=common_boundaries,
            subplot=ax1,
        )
        plot_u10(
            rwrf_lon_grid,
            rwrf_lat_grid,
            rwrf_extracted_data,
            rwrf_time_str,
            source="rwrf",
            boundaries=common_boundaries,
            subplot=ax2,
        )
        plt.tight_layout()
        plt.savefig("comparison_u10.png")
        plt.show()


if __name__ == "__main__":
    main()
    # # Example usage:
    # python align.py --era5_path ./cache/era5/t2m_20200601_00.npz --rwrf_path ./cache/rwrf/t2m_20190803_00.npz --variable t2m
