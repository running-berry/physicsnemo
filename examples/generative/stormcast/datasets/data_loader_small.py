# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import torch
import numpy as np
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from datetime import datetime, timedelta
import dask
import xarray as xr
import regex as re

from .dataset import StormCastDataset

logger = PythonLogger("dataset")


class Dataset(StormCastDataset):
    """
    Paired dataset object serving time-synchronized pairs of LowRes and HighRes samples
    TODO
    """

    def __init__(self, params, train):

        dist = DistributedManager()
        self.logger0 = RankZeroLoggingWrapper(logger, dist)

        dask.config.set(
            scheduler="synchronous"
        )  # for threadsafe multiworker dataloaders
        self.params = params
        self.location = self.params.location
        self.train = train
        self.path_suffix = "train" if train else "valid"
        self.dt = params.dt
        self.normalize = True
        self._get_files_stats()

        self.kept_LowRes_channels = (
            self.LowRes_channels
            if params.kept_LowRes_channels == "all"
            else params.kept_LowRes_channels
        )
        self.kept_HighRes_channels = (
            self.HighRes_channels
            if params.kept_HighRes_channels == "all"
            else params.kept_HighRes_channels
        )
        kept_LowRes_idx = [self.LowRes_channels.index(c) for c in self.kept_LowRes_channels]
        kept_HighRes_idx = [self.HighRes_channels.index(c) for c in self.kept_HighRes_channels]

        self.means_HighRes = np.load(
            os.path.join(
                self.location, "HighRes", "stats", "means.npy"
            )
        )[kept_HighRes_idx, None, None]
        self.stds_HighRes = np.load(
            os.path.join(
                self.location, "HighRes", "stats", "stds.npy"
            )
        )[kept_HighRes_idx, None, None]
        self.means_LowRes = np.load(
            os.path.join(self.location, "LowRes", "stats", "means.npy")
        )[kept_LowRes_idx, None, None]
        self.stds_LowRes = np.load(
            os.path.join(self.location, "LowRes", "stats", "stds.npy")
        )[kept_LowRes_idx, None, None]
        self.invariants = params.invariants

    def background_channels(self):
        """Metadata for the background channels. A list of channel names, one for each channel"""
        return self.kept_LowRes_channels

    def state_channels(self):
        """Metadata for the state channels. A list of channel names, one for each channel"""
        return self.kept_HighRes_channels

    def image_shape(self):
        """Get the (height, width) of the data (same for input and output)."""
        return tuple(self.params.HighRes_img_size)

    def get_invariants(self):
        """Return invariants used for training, or None if no invariants are used."""
        return None
    
    def _extract_date_key(self, path):
        """Extract the start and end dates from the zarr filename."""

        # Assumes filename format: exp_<channelvars>_<startdate>-<enddate>.zarr or exp_<channelvars>_<domain>_<startdate>-<enddate>.zarr
        fname = os.path.basename(path).replace(".zarr", "")
        pattern = r"^exp_[a-zA-Z0-9\-]+(_[a-zA-Z0-9x]+)?_\d{8}-\d{8}$"  # Matches 'exp_<channelvars>[_<domain>]_<startdate>-<enddate>'
        assert re.match(
            pattern, fname
        ), f"Filename '{fname}' does not match expected format 'exp_<channelvars>[_<domain>]_<startdate>-<enddate>'"

        parts = fname.split("_")
        # The last part are the dates
        if len(parts) >= 3:
            return parts[-1].split("-")
        return ("", "")

    def _get_files_stats(self):
        """
        Scan directories and extract metadata for LowRes and HighRes
        """

        # LowRes parsing
        self.LowRes_paths = glob.glob(
            os.path.join(self.location, "LowRes", "**", "*.zarr"), recursive=True
        )

        self.LowRes_paths = sorted(
            self.LowRes_paths, key=lambda x: self._extract_date_key(x)
        )

        self.logger0.info(f"list of all LowRes paths: {self.LowRes_paths}")

        if self.train:
            # keep only zarr files specified in the params.exp_train_zarrs list
            self.LowRes_paths = [
                x
                for x in self.LowRes_paths
                if os.path.basename(x).replace(".zarr", "")
                in self.params.exp_train_zarrs
            ]
            self.LowRes_zarrs = [
                os.path.basename(x).replace(".zarr", "") for x in self.LowRes_paths
            ]
            self.years = [
                int(self._extract_date_key(x)[0][:4]) for x in self.LowRes_paths
            ]
        else:
            # keep only zarr files specified in the params.exp_valid_zarrs list
            self.LowRes_paths = [
                x
                for x in self.LowRes_paths
                if os.path.basename(x).replace(".zarr", "")
                in self.params.exp_valid_zarrs
            ]
            self.LowRes_zarrs = [
                os.path.basename(x).replace(".zarr", "") for x in self.LowRes_paths
            ]
            self.years = [
                int(self._extract_date_key(x)[0][:4]) for x in self.LowRes_paths
            ]

        self.logger0.info(f"list of all LowRes paths after filtering: {self.LowRes_paths}")
        self.n_zarrs = len(self.LowRes_paths)
        self.years = sorted(set(self.years))
        self.logger0.info(f"list of all LowRes years: {self.years}")

        with xr.open_zarr(self.LowRes_paths[0], consolidated=True) as ds:
            self.LowRes_channels = list(ds.channel.values)
            self.LowRes_lat = ds.latitude
            self.LowRes_lon = ds.longitude

        self.n_samples_total = self.compute_total_samples()
        self.ds_LowRes = [
            xr.open_zarr(self.LowRes_paths[i], consolidated=True)
            for i in range(self.n_zarrs)
        ]

        # HighRes parsing
        self.HighRes_paths = glob.glob(
            os.path.join(self.location, "HighRes", "**", "*.zarr"),
            recursive=True,
        )
        self.logger0.info(f"list of all HighRes paths: {self.HighRes_paths}")
        self.HighRes_paths = sorted(
            self.HighRes_paths, key=lambda x: self._extract_date_key(x)
        )
        if self.train:
            # keep only zarr files specified in the params.exp_train_zarrs list
            self.HighRes_paths = [
                x
                for x in self.HighRes_paths
                if os.path.basename(x).replace(".zarr", "")
                in self.params.exp_train_zarrs
            ]
            self.HighRes_zarrs = [
                os.path.basename(x).replace(".zarr", "")
                for x in self.HighRes_paths
            ]
            self.HighRes_years = [
                int(self._extract_date_key(x)[0][:4]) for x in self.HighRes_paths
            ]
        else:
            # keep only zarr files specified in the params.exp_valid_zarrs list
            self.HighRes_paths = [
                x
                for x in self.HighRes_paths
                if os.path.basename(x).replace(".zarr", "")
                in self.params.exp_valid_zarrs
            ]
            self.HighRes_zarrs = [
                os.path.basename(x).replace(".zarr", "")
                for x in self.HighRes_paths
            ]
            self.HighRes_years = [
                int(self._extract_date_key(x)[0][:4]) for x in self.HighRes_paths
            ]

        self.logger0.info(f"list of all HighRes paths after filtering: {self.HighRes_paths}")
        self.HighRes_years = sorted(set(self.HighRes_years))
        self.logger0.info(f"list of all HighRes years: {self.HighRes_years}")
        
        assert (
            self.LowRes_zarrs == self.HighRes_zarrs
        ), "Number of zarrs for LowRes in %s and HighRes in %s must match" % (
            os.path.join(self.location, "LowRes/*.zarr"),
            os.path.join(self.location, "HighRes/*.zarr"),
        )
        assert (
            self.HighRes_years == self.years
        ), "Number of years for LowRes in %s and HighRes in %s must match" % (
            os.path.join(self.location, "LowRes/*.zarr"),
            os.path.join(self.location, "HighRes/*.zarr"),
        )
        with xr.open_zarr(self.HighRes_paths[0], consolidated=True) as ds:
            self.HighRes_channels = list(ds.channel.values)
            self.HighRes_lat = ds.latitude
            self.HighRes_lon = ds.longitude
            
        self.ds_HighRes = [
            xr.open_zarr(self.HighRes_paths[i], consolidated=True, mask_and_scale=False)
            for i in range(self.n_zarrs)
        ]



    def __len__(self):
        return self.n_samples_total

    def to_datetime(self, date):

        timestamp = (date - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
            1, "s"
        )
        return datetime.utcfromtimestamp(timestamp)

    def compute_total_samples(self):
        """
        Loop through all years and count the total number of samples
        """
        
        first_year = sorted(self.years)[0]
        last_year = sorted(self.years)[-1]
        
        test_datetime_start = self.params.train_dates[0]
        test_datetime_last = self.params.train_dates[1]

        first_sample = datetime.strptime(
            test_datetime_start, "%Y/%m/%d") \
                .replace(hour=0, minute=0, second=0)
        
        self.logger0.info("First sample is {}".format(first_sample))

        last_sample = datetime.strptime(
            test_datetime_last, "%Y/%m/%d") \
                .replace(hour=23, minute=0, second=0)

        self.logger0.info("Last sample is {}".format(last_sample))
        
        all_datetimes = [
            first_sample + timedelta(hours=x)
            for x in range(int((last_sample - first_sample).total_seconds() / 3600) + 1)
        ]

        missing_samples = set([])  # hash for faster lookup

        self.valid_samples = [
            x
            for x in all_datetimes
            if (x not in missing_samples)
                and (x + timedelta(hours=self.dt) <= last_sample)
                and ((x + timedelta(hours=self.dt)) not in missing_samples)
        ]

        self.logger0.info(
            "Total datetimes in training set are {} of which {} are valid".format(
                len(all_datetimes), len(self.valid_samples)
            )
        )

        return len(self.valid_samples)

    def normalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from physical units to normalized data."""
        if self.normalize:
            x -= self.means_LowRes
            x /= self.stds_LowRes
        return x

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from normalized data to physical units."""
        if self.normalize:
            x *= self.stds_LowRes
            x += self.means_LowRes
        return x

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from physical units to normalized data."""
        if self.normalize:
            x -= self.means_HighRes
            x /= self.stds_HighRes
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from normalized data to physical units."""
        if self.normalize:
            x *= self.stds_HighRes
            x += self.means_HighRes
        return x

    def _get_LowRes(self, ts_inp, ts_tar):
        """
        Retrieve LowRes samples from zarr files
        """

        ds_inp, ds_tar, adjacent = self._get_ds_handles(
            self.ds_LowRes, self.LowRes_paths, ts_inp, ts_tar
        )

        inp_field = ds_inp.sel(time=ts_inp, channel=self.kept_LowRes_channels).LowRes.values

        inp = self.normalize_background(inp_field)
        return torch.as_tensor(inp)

    def _get_HighRes(self, ts_inp, ts_tar):
        """
        Retrieve HighRes samples from zarr files
        """
        ds_inp, ds_tar, adjacent = self._get_ds_handles(
            self.ds_HighRes, self.HighRes_paths, ts_inp, ts_tar
        )

        inp_field = ds_inp.sel(time=ts_inp, channel=self.kept_HighRes_channels).HighRes.values
        tar_field = ds_tar.sel(time=ts_tar, channel=self.kept_HighRes_channels).HighRes.values

        inp, tar = self.normalize_state(inp_field), self.normalize_state(tar_field)

        return torch.as_tensor(inp), torch.as_tensor(tar)

    def __getitem__(self, global_idx):
        """
        Return data as a dict
        """
        time_pair = self._global_idx_to_datetime(global_idx)
        HighRes_pair = self._get_HighRes(*time_pair)
        LowRes_pair = self._get_LowRes(*time_pair)
        return {
            "background": LowRes_pair,
            "state": HighRes_pair,
        }

    def _global_idx_to_datetime(self, global_idx):
        """
        Parse a global sample index and return the input/target timstamps as datetimes
        """

        inp = self.valid_samples[global_idx]
        tar = inp + timedelta(hours=self.dt)

        return inp, tar

    def _get_ds_handles(self, handles, paths, ts_inp, ts_tar):
        """
        Return opened dataset handles for the appropriate year, and boolean indicating if they are from the same year
        """
        ds_handles = []
        for year in [ts_inp.year, ts_tar.year]:
            year_idx = self.years.index(year)
            ds_handles.append(handles[year_idx])
        return ds_handles[0], ds_handles[1], ds_handles[0] == ds_handles[1]
