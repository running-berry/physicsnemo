import os
import pytest
import xarray as xr
import numpy as np
from netCDF4 import Dataset
import combine_rwrf_qpepre as mod

# parameters
DATE = "2019/08/03"
HOUR = "00"


class TestCombineRwrfQpepre:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.mock_rwrf_path = tmp_path / "fake_rwrf.nc"
        self.mock_qpepre_path = tmp_path / "fake_qpepre.nc"
        self.mock_combined_path = tmp_path / "combined.nc"
        # simple linear function for the ground truth to test interpolation
        self.known_func = lambda lat, lon: 2 * lat + 0.5 * lon

        # Mock RWRF (5x5) curvilinear grid with T2M and lat/lon variables
        with Dataset(self.mock_rwrf_path, "w", format="NETCDF4") as rwrf_ds:
            rwrf_ds.createDimension("Time", 1)
            rwrf_ds.createDimension("south_north", 5)
            rwrf_ds.createDimension("west_east", 5)
            lats = np.linspace(22, 23, 5)
            lons = np.linspace(120, 121, 5)
            lon2d, lat2d = np.meshgrid(lons, lats)
            x_idx, y_idx = np.meshgrid(np.arange(5), np.arange(5))
            lat2d += np.sin(x_idx * 0.2) * 0.02
            lon2d += np.cos(y_idx * 0.2) * 0.02
            xlat = rwrf_ds.createVariable(
                "XLAT", "f4", ("Time", "south_north", "west_east")
            )
            xlat[0, :, :] = lat2d
            xlong = rwrf_ds.createVariable(
                "XLONG", "f4", ("Time", "south_north", "west_east")
            )
            xlong[0, :, :] = lon2d
            t2 = rwrf_ds.createVariable(
                "T2", "f4", ("Time", "south_north", "west_east")
            )
            t2[:] = np.ones((1, 5, 5))

        # Mock QPEPRE (2x2) regular grid
        with Dataset(self.mock_qpepre_path, "w", format="NETCDF4") as qpepre_ds:
            qpepre_ds.createDimension("times", 1)
            qpepre_ds.createDimension("lat", 2)
            qpepre_ds.createDimension("lon", 2)
            lat1d_var = qpepre_ds.createVariable("lat", "f4", ("times", "lat"))
            lat1d_var[0, :] = [21.9, 23.1]
            lon1d_var = qpepre_ds.createVariable("lon", "f4", ("times", "lon"))
            lon1d_var[0, :] = [119.9, 121.1]
            qpepre_var = qpepre_ds.createVariable(
                "qpepre", "f4", ("times", "lat", "lon")
            )
            q_lon2d, q_lat2d = np.meshgrid(lon1d_var[0, :], lat1d_var[0, :])
            qpepre_var[0, :, :] = self.known_func(q_lat2d, q_lon2d)

        # Real Data Paths
        self.real_data_paths = {
            "orig": os.path.join(
                "..", "data", "rwrf", "2019-08-03_00", "wrfout_d01_2019-08-03_00_interp"
            ),
            "comb": os.path.join(
                "..",
                "data",
                "rwrf",
                "2019-08-03_00",
                "wrfout_d01_2019-08-03_00_interp_cropped_qpepre.nc",
            ),
        }
        yield

    @pytest.fixture
    def true_values(self):
        with Dataset(self.mock_rwrf_path) as rwrf_ds:
            rwrf_lat = rwrf_ds.variables["XLAT"][:]
            rwrf_lon = rwrf_ds.variables["XLONG"][:]
            return self.known_func(rwrf_lat, rwrf_lon)

    @pytest.fixture
    def griddata_interp_result(self):
        with (
            Dataset(self.mock_rwrf_path) as rwrf_ds,
            Dataset(self.mock_qpepre_path) as qpepre_ds,
        ):
            mod.combine_rwrf_qpepre(rwrf_ds, qpepre_ds, str(self.mock_combined_path))
        with Dataset(self.mock_combined_path) as result_ds:
            return result_ds.variables["qpepre"][:]

    @pytest.fixture
    def xarray_interp_result(self):
        with Dataset(self.mock_rwrf_path) as rwrf_ds:
            rwrf_lat = rwrf_ds.variables["XLAT"][:]
            rwrf_lon = rwrf_ds.variables["XLONG"][:]
        qpepre_da = xr.open_dataset(self.mock_qpepre_path)["qpepre"].squeeze()
        target_lat_da = xr.DataArray(
            rwrf_lat.squeeze(), dims=("south_north", "west_east")
        )
        target_lon_da = xr.DataArray(
            rwrf_lon.squeeze(), dims=("south_north", "west_east")
        )
        xarray_result_da = qpepre_da.interp(
            lat=target_lat_da, lon=target_lon_da, method="linear"
        )
        return xarray_result_da.to_numpy()[np.newaxis, :, :]

    @pytest.fixture
    def regular_grid_result(self):
        from scipy.interpolate import RegularGridInterpolator

        with Dataset(self.mock_qpepre_path) as qpepre_ds:
            source_lats = qpepre_ds.variables["lat"][0, :]
            source_lons = qpepre_ds.variables["lon"][0, :]
            source_values = qpepre_ds.variables["qpepre"][0, :, :]
        interp = RegularGridInterpolator((source_lats, source_lons), source_values)
        target_lats = np.linspace(22, 23, 5)
        target_lons = np.linspace(120, 121, 5)
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
        target_points = np.column_stack(
            (target_lat_grid.ravel(), target_lon_grid.ravel())
        )
        return interp(target_points).reshape(1, 5, 5)

    def test_latlon_match(self):
        comb_path = self.real_data_paths["comb"]
        if not os.path.exists(comb_path):
            pytest.skip(f"Real data file not found: {comb_path}")
        ds = xr.open_dataset(comb_path, decode_times=False)
        assert ds["qpepre"]["XLAT"].equals(ds["RAINNC"]["XLAT"])
        assert ds["qpepre"]["XLONG"].equals(ds["RAINNC"]["XLONG"])

    def test_interpolation_results(
        self,
        griddata_interp_result,
        xarray_interp_result,
        regular_grid_result,
        true_values,
    ):
        # Check if the interpolation results match the known values
        # griddata and xarray should match the true values, while regular grid should not
        assert np.allclose(
            griddata_interp_result, true_values, atol=1e-5
        ), "Griddata interpolation result does not match known values"
        assert np.allclose(
            xarray_interp_result, true_values, atol=1e-5
        ), "Xarray interpolation result does not match known values"
        assert not np.allclose(
            regular_grid_result, true_values, atol=1e-5
        ), "Regular grid interpolation result should not match known values"

        # Check if the interpolation results from different methods are consistent
        # Griddata and xarray should be similar, but regular grid should differ
        assert np.allclose(
            griddata_interp_result, xarray_interp_result, atol=1e-5
        ), "Griddata and xarray results differ"
        assert not np.allclose(
            griddata_interp_result, regular_grid_result, atol=1e-5
        ), "Griddata and regular grid interpolation results should differ"
        assert not np.allclose(
            xarray_interp_result, regular_grid_result, atol=1e-5
        ), "Xarray and regular grid interpolation results should differ"

    def test_original_has_no_qpepre(self):
        orig_path = self.real_data_paths["orig"]
        if not os.path.exists(orig_path):
            pytest.skip(f"Real data file not found: {orig_path}")
        ds_orig = xr.open_dataset(orig_path, decode_times=False)
        assert "qpepre" not in ds_orig.data_vars
