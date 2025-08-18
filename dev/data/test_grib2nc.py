import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import calendar
import numpy as np

# Import the function under test
from grib2nc import grib_to_netcdf_pygrib


class DummyGrib:
    def __init__(self):
        self.values = [[1, 2], [3, 4]]
    
    def latlons(self):
        lats = np.array([[10, 10], [20, 20]])
        lons = np.array([[30, 40], [30, 40]])
        return lats, lons

    def get(self, key, default=None):
        if key == 'units':
            return 'K'
        if key == 'name':
            return 'Temperature'
        if key == 'gridType':
            return 'regular_ll'
        return default

    def __str__(self):
        return "DummyGrib message"


@pytest.fixture
def mock_pygrib_open():
    class DummyGribFile:
        def __init__(self, msgs):
            self._msgs = msgs
            self.closed = False
        # iterable
        def __iter__(self):
            return iter(self._msgs)
        # close-able
        def close(self):
            self.closed = True
        # context-manager compatible
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False  # don’t suppress exceptions

    with patch("grib2nc.pygrib.open", autospec=True) as mock_open:
        mock_open.return_value = DummyGribFile([DummyGrib()])
        yield mock_open


@pytest.fixture
def mock_netcdf_dataset():
    with patch("grib2nc.Dataset") as mock_ds:
        mock_nc = MagicMock()
        mock_ds.return_value.__enter__.return_value = mock_nc
        yield mock_ds


def test_grib_to_netcdf_success(tmp_path, mock_pygrib_open, mock_netcdf_dataset):
    grib_file = tmp_path / "t2m_2025010100.grib"
    grib_file.write_text("dummy content")  # create fake file

    output = grib_to_netcdf_pygrib(str(grib_file))

    assert output.endswith(".nc")
    mock_pygrib_open.assert_called_once_with(str(grib_file))
    mock_netcdf_dataset.assert_called_once()


def test_grib_to_netcdf_bad_filename(tmp_path, mock_pygrib_open, mock_netcdf_dataset):
    grib_file = tmp_path / "invalidname.grib"
    grib_file.write_text("dummy content")

    output = grib_to_netcdf_pygrib(str(grib_file))

    assert output.endswith(".nc")
    mock_pygrib_open.assert_called_once()
    mock_netcdf_dataset.assert_called_once()


def test_grib_to_netcdf_exception(tmp_path):
    grib_file = tmp_path / "t2m_2025010100.grib"
    grib_file.write_text("dummy content")

    # Force pygrib.open to raise an exception
    with patch("grib2nc.pygrib.open", side_effect=RuntimeError("cannot open")), \
         patch("grib2nc.Dataset") as mock_nc:
        output = grib_to_netcdf_pygrib(str(grib_file))

    # Should return None on failure
    assert output is None
