import os
import pytest
import xarray as xr

from utils.config import CONFIG
import combine_rwrf_qpepre as mod
import fetch_rwrf as fetr
from utils.txt2ds import load_qpepre, qpepre_stats

# parameters
DATE = "2019/08/03"
HOUR = "00"

@pytest.fixture
def paths(tmp_path, monkeypatch):
    """
    Set CONFIG paths to temporary dirs, ensure data is in place, 
    run conversion & combine once for all tests.
    """
    rwrf_nc_orig = os.path.join("..", "data", "rwrf", "2019-08-03_00", "wrfout_d01_2019-08-03_00_interp")
    rwrf_nc_comb = os.path.join("..", "data", "rwrf", "2019-08-03_00", "wrfout_d01_2019-08-03_00_interp_qpepre.nc")
    qpepre_txt = os.path.join("../", "data", "pptn", "qpepre_201908030000-201908030100_1_h.txt")

    return {
        "orig": rwrf_nc_orig,
        "comb": rwrf_nc_comb,
        "txt": qpepre_txt,
    }

def test_latlon_match(paths):
    ds = xr.open_dataset(paths["comb"], decode_times=False)
    # XLAT
    assert ds["qpepre"]["XLAT"].equals(ds["RAINNC"]["XLAT"])
    # XLONG
    assert ds["qpepre"]["XLONG"].equals(ds["RAINNC"]["XLONG"])
    pass

def test_qpe_stats(paths):
    # from TXT
    txt_ds = load_qpepre(paths["txt"])
    mean_txt, std_txt = qpepre_stats(txt_ds)

    # from combined NC
    ds = xr.open_dataset(paths["comb"], decode_times=False)
    mean_nc = float(ds["qpepre"].mean().item())
    std_nc  = float(ds["qpepre"].std().item())

    assert mean_nc == pytest.approx(mean_txt, rel=1e-3)
    assert std_nc  == pytest.approx(std_txt,  rel=1e-3)

def test_original_has_no_qpepre(paths):
    ds_orig = xr.open_dataset(paths["orig"], decode_times=False)
    assert "qpepre" not in ds_orig.data_vars