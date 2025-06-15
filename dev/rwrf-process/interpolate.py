from utils.data import load_data, decode_time, extract_slice
import matplotlib.pyplot as plt
import argparse
import xarray as xr

def plot_variable(
    lon,
    lat,
    data,
    time_str,
    levels=20,
    source="era5",
    variable="t2m",
    subplot=None,
):
    ax = subplot
    if variable == 't2m':
        cmap = "coolwarm"
        label = "T2 (°C)"
        title = f"{source.upper()} 2m Temp at {time_str}"
    elif variable == 'u10':
        cmap = plt.cm.jet
        label = "u10 (m/s)"
        title = f"{source.upper()} 10m u-component of wind at {time_str}"
    cs = ax.contourf(lon, lat, data, levels=levels, cmap=cmap)
    plt.colorbar(cs, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

def main():
    parser = argparse.ArgumentParser(
        description="Plot and compare 2m temperature or u10 from ERA5 and RWRF."
    )
    parser.add_argument("--era5_path", help="Path to ERA5 NPZ file", required=True)
    parser.add_argument("--rwrf_path", help="Path to RWRF NPZ file", required=True)
    parser.add_argument('--variable', choices=['t2m', 'u10'], required=True, help='Variable to plot')
    args = parser.parse_args()

    era5_data, era5_lat, era5_lon, era5_times = load_data(args.era5_path, "era5", args.variable)
    rwrf_data, rwrf_lat, rwrf_lon, rwrf_times = load_data(args.rwrf_path, "rwrf", args.variable)

    era5_time_str = decode_time(era5_times[0], "era5", args.variable)
    rwrf_time_str = decode_time(rwrf_times[0], "rwrf", args.variable)

    era5_extracted_data, era5_lat_grid, era5_lon_grid = extract_slice(
        era5_data, era5_lat, era5_lon, "era5", args.variable
    )
    rwrf_extracted_data, rwrf_lat_grid, rwrf_lon_grid = extract_slice(
        rwrf_data, rwrf_lat, rwrf_lon, "rwrf", args.variable
    )

    da_era5 = xr.DataArray(
        era5_extracted_data,
        dims=('lat', 'lon'),
        coords={'lat': era5_lat_grid, 'lon': era5_lon_grid}
    )

    da_era5_on_rwrf = da_era5.interp(
        lat=(('y', 'x'), rwrf_lat_grid),
        lon=(('y', 'x'), rwrf_lon_grid),
        method='linear'
    )
    interpolated_era5 = da_era5_on_rwrf.values
    # TODO: Save interpolated data to a file if needed

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_variable(
        rwrf_lon_grid, rwrf_lat_grid, interpolated_era5, era5_time_str,
        source="era5", variable=args.variable, subplot=ax1
    )
    plot_variable(
        rwrf_lon_grid, rwrf_lat_grid, rwrf_extracted_data, rwrf_time_str,
        source="rwrf", variable=args.variable, subplot=ax2
    )
    plt.tight_layout()
    plt.savefig(f"interp_{args.variable}.png")
    plt.show()

if __name__ == "__main__":
    main()
