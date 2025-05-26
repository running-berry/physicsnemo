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

from .dataset import StormCastDataset

logger = PythonLogger("dataset")


class DummyDataset(StormCastDataset):
    """
    Paired dataset object serving time-synchronized pairs of DummyLowRes and DummyHighRes samples
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

        self.kept_DummyLowRes_channels = (
            self.DummyLowRes_channels
            if params.kept_DummyLowRes_channels == "all"
            else params.kept_DummyLowRes_channels
        )
        self.kept_DummyHighRes_channels = (
            self.DummyHighRes_channels
            if params.kept_DummyHighRes_channels == "all"
            else params.kept_DummyHighRes_channels
        )
        kept_DummyLowRes_idx = [self.DummyLowRes_channels.index(c) for c in self.kept_DummyLowRes_channels]
        kept_DummyHighRes_idx = [self.DummyHighRes_channels.index(c) for c in self.kept_DummyHighRes_channels]

        self.means_DummyHighRes = np.load(
            os.path.join(
                self.location, "DummyHighRes", "stats", "means.npy"
            )
        )[kept_DummyHighRes_idx, None, None]
        self.stds_DummyHighRes = np.load(
            os.path.join(
                self.location, "DummyHighRes", "stats", "stds.npy"
            )
        )[kept_DummyHighRes_idx, None, None]
        self.means_DummyLowRes = np.load(
            os.path.join(self.location, "DummyLowRes", "stats", "means.npy")
        )[kept_DummyLowRes_idx, None, None]
        self.stds_DummyLowRes = np.load(
            os.path.join(self.location, "DummyLowRes", "stats", "stds.npy")
        )[kept_DummyLowRes_idx, None, None]
        self.invariants = params.invariants

    def background_channels(self):
        """Metadata for the background channels. A list of channel names, one for each channel"""
        return self.kept_DummyLowRes_channels

    def state_channels(self):
        """Metadata for the state channels. A list of channel names, one for each channel"""
        return self.kept_DummyHighRes_channels

    def image_shape(self):
        """Get the (height, width) of the data (same for input and output)."""
        return tuple(self.params.DummyHighRes_img_size)

    def get_invariants(self):
        """Return invariants used for training, or None if no invariants are used."""
        return None

    def _get_files_stats(self):
        """
        Scan directories and extract metadata for DummyLowRes and DummyHighRes
        """

        # DummyLowRes parsing
        self.DummyLowRes_paths = glob.glob(
            os.path.join(self.location, "DummyLowRes", "**", "????.zarr"), recursive=True
        )

        self.DummyLowRes_paths = sorted(
            self.DummyLowRes_paths, key=lambda x: int(os.path.basename(x).replace(".zarr", ""))
        )

        self.logger0.info(f"list of all DummyLowRes paths: {self.DummyLowRes_paths}")

        if self.train:
            # keep only years specified in the params.train_years list
            self.DummyLowRes_paths = [
                x
                for x in self.DummyLowRes_paths
                if int(os.path.basename(x).replace(".zarr", ""))
                in self.params.train_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".zarr", "")) for x in self.DummyLowRes_paths
            ]
        else:
            # keep only years specified in the params.valid_years list
            self.DummyLowRes_paths = [
                x
                for x in self.DummyLowRes_paths
                if int(os.path.basename(x).replace(".zarr", ""))
                in self.params.valid_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".zarr", "")) for x in self.DummyLowRes_paths
            ]

        self.logger0.info(f"list of all DummyLowRes paths after filtering: {self.DummyLowRes_paths}")
        self.n_years = len(self.DummyLowRes_paths)

        with xr.open_zarr(self.DummyLowRes_paths[0], consolidated=True) as ds:
            self.DummyLowRes_channels = list(ds.channel.values)
            self.DummyLowRes_lat = ds.latitude
            self.DummyLowRes_lon = ds.longitude

        self.n_samples_total = self.compute_total_samples()
        self.ds_DummyLowRes = [
            xr.open_zarr(self.DummyLowRes_paths[i], consolidated=True)
            for i in range(self.n_years)
        ]

        # DummyHighRes parsing
        self.DummyHighRes_paths = glob.glob(
            os.path.join(self.location, "DummyHighRes", "**", "????.zarr"),
            recursive=True,
        )
        self.logger0.info(f"list of all DummyHighRes paths: {self.DummyHighRes_paths}")
        self.DummyHighRes_paths = sorted(
            self.DummyHighRes_paths, key=lambda x: int(os.path.basename(x).replace(".zarr", ""))
        )
        if self.train:
            # keep only years specified in the params.train_years list
            self.DummyHighRes_paths = [
                x
                for x in self.DummyHighRes_paths
                if int(os.path.basename(x).replace(".zarr", ""))
                in self.params.train_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".zarr", "")) for x in self.DummyHighRes_paths
            ]
        else:
            # keep only years specified in the params.valid_years list
            self.DummyHighRes_paths = [
                x
                for x in self.DummyHighRes_paths
                if int(os.path.basename(x).replace(".zarr", ""))
                in self.params.valid_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".zarr", "")) for x in self.DummyHighRes_paths
            ]

        self.logger0.info(f"list of all DummyHighRes paths after filtering: {self.DummyHighRes_paths}")

        years = [int(os.path.basename(x).replace(".zarr", "")) for x in self.DummyHighRes_paths]
        self.logger0.info(f"years: {years}")
        self.logger0.info(f"self.years: {self.years}")
        
        assert (
            years == self.years
        ), "Number of years for DummyLowRes in %s and DummyHighRes in %s must match" % (
            os.path.join(self.location, "DummyLowRes/*.zarr"),
            os.path.join(self.location, "DummyHighRes/*.zarr"),
        )
        with xr.open_zarr(self.DummyHighRes_paths[0], consolidated=True) as ds:
            self.DummyHighRes_channels = list(ds.channel.values)
            self.DummyHighRes_lat = ds.latitude
            self.DummyHighRes_lon = ds.longitude
            
        self.ds_DummyHighRes = [
            xr.open_zarr(self.DummyHighRes_paths[i], consolidated=True, mask_and_scale=False)
            for i in range(self.n_years)
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

        first_sample = datetime(
            year=first_year, month=1, day=1, hour=0, minute=0, second=0
        )
        self.logger0.info("First sample is {}".format(first_sample))

        last_sample = datetime(
            year=last_year, month=12, day=31, hour=23, minute=0, second=0
        )
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
            x -= self.means_DummyLowRes
            x /= self.stds_DummyLowRes
        return x

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from normalized data to physical units."""
        if self.normalize:
            x *= self.stds_DummyLowRes
            x += self.means_DummyLowRes
        return x

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from physical units to normalized data."""
        if self.normalize:
            x -= self.means_DummyHighRes
            x /= self.stds_DummyHighRes
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from normalized data to physical units."""
        if self.normalize:
            x *= self.stds_DummyHighRes
            x += self.means_DummyHighRes
        return x

    def _get_DummyLowRes(self, ts_inp, ts_tar):
        """
        Retrieve DummyLowRes samples from zarr files
        """

        ds_inp, ds_tar, adjacent = self._get_ds_handles(
            self.ds_DummyLowRes, self.DummyLowRes_paths, ts_inp, ts_tar
        )

        inp_field = ds_inp.sel(time=ts_inp, channel=self.kept_DummyLowRes_channels).DummyLowRes.values

        inp = self.normalize_background(inp_field)
        return torch.as_tensor(inp)

    def _get_DummyHighRes(self, ts_inp, ts_tar):
        """
        Retrieve DummyHighRes samples from zarr files
        """
        ds_inp, ds_tar, adjacent = self._get_ds_handles(
            self.ds_DummyHighRes, self.DummyHighRes_paths, ts_inp, ts_tar
        )

        inp_field = ds_inp.sel(time=ts_inp, channel=self.kept_DummyHighRes_channels).DummyHighRes.values
        tar_field = ds_tar.sel(time=ts_tar, channel=self.kept_DummyHighRes_channels).DummyHighRes.values

        inp, tar = self.normalize_state(inp_field), self.normalize_state(tar_field)

        return torch.as_tensor(inp), torch.as_tensor(tar)

    def __getitem__(self, global_idx):
        """
        Return data as a dict
        """
        time_pair = self._global_idx_to_datetime(global_idx)
        DummyHighRes_pair = self._get_DummyHighRes(*time_pair)
        DummyLowRes_pair = self._get_DummyLowRes(*time_pair)
        return {
            "background": DummyLowRes_pair,
            "state": DummyHighRes_pair,
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
