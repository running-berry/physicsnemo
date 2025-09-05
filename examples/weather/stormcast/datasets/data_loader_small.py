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

debug_flag = False

class Dataset(StormCastDataset):
    """
    Paired dataset object serving time-synchronized pairs of LowRes and HighRes samples
    TODO
    """

    def __init__(self, params, train):

        dist = DistributedManager()
        self.logger0 = RankZeroLoggingWrapper(logger, dist)
        # Some backends expose only info/warning/error; provide a debug alias
        if not hasattr(self.logger0, "debug"):
            # fall back to info so debug logs don't crash workers
            self.logger0.debug = self.logger0.info  # type: ignore[attr-defined]

        dask.config.set(
            scheduler="synchronous"
        )  # for threadsafe multiworker dataloaders
        self.params = params
        self.location = self.params.location
        self.train = train
        self.path_suffix = "train" if train else "valid"
        self.date_ranges = params.train_dates if train else params.valid_dates
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

        # debug nan lowres
        # File to save NaN channel logs
        os.makedirs("logs", exist_ok=True)
        self.nan_log_file = os.path.join("logs", f"nan_channels_{'train' if train else 'valid'}.txt")
        with open(self.nan_log_file, "w") as f:
            f.write("timestamp,channels\n")  # CSV-style header

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

        invariants = xr.open_zarr(
            os.path.join(self.location, "invariants", "invariants.zarr")
        )

        invariant_channels_in_dataset = list(invariants.channel.values)

        for invariant in self.invariants:
            assert (
                invariant in invariant_channels_in_dataset
            ), f"Requested invariant {invariant} not in dataset"

        invariant_array = (
            invariants["HighRes_invariants"].sel(channel=self.invariants).values
        )

        return invariant_array

    def _get_files_stats(self):
        """
        Scan directories and extract metadata for LowRes and HighRes
        """

        # LowRes parsing
        self.LowRes_paths = glob.glob(
            os.path.join(self.location, "LowRes", "**", "*.zarr"), recursive=True
        )

        self.LowRes_paths = sorted(
            self.LowRes_paths, key=lambda p: os.path.basename(p).replace(".zarr", "")
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

        self.logger0.info(f"list of all LowRes paths after filtering: {self.LowRes_paths}")
        self.n_zarrs = len(self.LowRes_paths)

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
            self.HighRes_paths, key=lambda p: os.path.basename(p).replace(".zarr", "")
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

        self.logger0.info(f"list of all HighRes paths after filtering: {self.HighRes_paths}")
        
        assert (
            self.LowRes_zarrs == self.HighRes_zarrs
        ), "Number of zarrs for LowRes in %s and HighRes in %s must match" % (
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
        if self.train:
            test_datetime_start = self.params.train_dates[0]
            test_datetime_last = self.params.train_dates[1]
        else:
            test_datetime_start = self.params.valid_dates[0]
            test_datetime_last = self.params.valid_dates[1]

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
            # self.logger0.info(f"lowres- x: {x}, mean:{self.means_LowRes}, std:{self.stds_LowRes}")
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
        ds_inp, ds_tar, adjacent = self._get_ds_handles(
            self.ds_LowRes, self.LowRes_paths, ts_inp, ts_tar
        )
        try:
            inp_field = ds_inp.sel(time=ts_inp, channel=self.kept_LowRes_channels).LowRes.values
        except KeyError:
            self.logger0.error(
                f"[LowRes] Timestamp {ts_inp} not found in dataset index. "
                f"Available times range: {ds_inp.time.values[0]} → {ds_inp.time.values[-1]} "
                f"(len={len(ds_inp.time.values)})"
            )
            raise

        if debug_flag:
            self.logger0.debug(
                f"_get_LowRes ts_inp={ts_inp}, shape={inp_field.shape}, "
                f"channels={self.kept_LowRes_channels}"
            )
        inp = self.normalize_background(inp_field)

        nan_channels = [
            ch for i, ch in enumerate(self.kept_LowRes_channels)
            if np.isnan(inp[i]).any()
        ]
        if nan_channels:
            self.logger0.warning(
                f"[LowRes] NaNs detected after normalization at ts={ts_inp} in channels: {nan_channels}"
            )
            with open(self.nan_log_file, "a") as f:
                f.write(f"{ts_inp},{'|'.join(nan_channels)}\n")

        return torch.as_tensor(inp)

    def _get_HighRes(self, ts_inp, ts_tar):
        ds_inp, ds_tar, adjacent = self._get_ds_handles(
            self.ds_HighRes, self.HighRes_paths, ts_inp, ts_tar
        )
        try:
            inp_field = ds_inp.sel(time=ts_inp, channel=self.kept_HighRes_channels).HighRes.values
        except KeyError:
            self.logger0.error(
                f"[HighRes] Input timestamp {ts_inp} not found. "
                f"Range: {ds_inp.time.values[0]} → {ds_inp.time.values[-1]} "
                f"(len={len(ds_inp.time.values)})"
            )
            raise

        try:
            tar_field = ds_tar.sel(time=ts_tar, channel=self.kept_HighRes_channels).HighRes.values
        except KeyError:
            self.logger0.error(
                f"[HighRes] Target timestamp {ts_tar} not found. "
                f"Range: {ds_tar.time.values[0]} → {ds_tar.time.values[-1]} "
                f"(len={len(ds_tar.time.values)})"
            )
            raise

        # Normalize to produce tensors used below
        inp = self.normalize_state(inp_field)
        tar = self.normalize_state(tar_field)

        if debug_flag:
            self.logger0.debug(
                f"_get_HighRes ts_inp={ts_inp}, ts_tar={ts_tar}, "
                f"in_shape={inp_field.shape}, tar_shape={tar_field.shape}, "
                f"channels={self.kept_HighRes_channels}"
            )

        # Log NaNs
        for arr, tag in [(inp, "inp"), (tar, "tar")]:
            nan_channels = [
                ch for i, ch in enumerate(self.kept_HighRes_channels)
                if np.isnan(arr[i]).any()
            ]
            if nan_channels:
                self.logger0.warning(
                    f"[HighRes-{tag}] NaNs detected at ts={ts_inp}->{ts_tar} in channels: {nan_channels}"
                )
                with open(self.nan_log_file, "a") as f:
                    f.write(f"{ts_inp}->{ts_tar},{'|'.join(nan_channels)}\n")

        return torch.as_tensor(inp), torch.as_tensor(tar)

    def __getitem__(self, global_idx):
        """
        Return data as a dict
        """
        time_pair = self._global_idx_to_datetime(global_idx)
        ts_inp, ts_tar = time_pair

        if debug_flag:
            self.logger0.info(
                f"Fetching sample idx={global_idx}, inp={ts_inp}, tar={ts_tar}"
            )

        HighRes_pair = self._get_HighRes(*time_pair)
        LowRes_pair = self._get_LowRes(*time_pair)

        if debug_flag:
            # Log shapes and a quick summary
            if isinstance(LowRes_pair, torch.Tensor):
                self.logger0.info(
                    f"LowRes shape: {tuple(LowRes_pair.shape)} "
                    f"(min={LowRes_pair.min().item():.4f}, max={LowRes_pair.max().item():.4f})"
                )
            if isinstance(HighRes_pair, tuple):
                inp, tar = HighRes_pair
                self.logger0.info(
                    f"HighRes inp shape: {tuple(inp.shape)}, "
                    f"tar shape: {tuple(tar.shape)} "
                    f"(inp[min={inp.min().item():.4f}, max={inp.max().item():.4f}], "
                    f"tar[min={tar.min().item():.4f}, max={tar.max().item():.4f}])"
                )

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
        Return the two dataset‐handles whose date‐ranges contain ts_inp/ts_tar,
        plus a boolean indicating if they ended up in the same handle.
        """
        ds_handles = []

        # flat list of two strings → single tuple
        start_s, end_s = self.date_ranges
        fmt = "%Y/%m/%d"
        date_ranges = [
            (
                datetime.strptime(start_s, fmt).date(),
                datetime.strptime(end_s, fmt).date(),
            )
        ]

        for ts in (ts_inp, ts_tar):
            for idx, (start, end) in enumerate(date_ranges):
                if start <= ts.date() <= end:
                    ds_handles.append(handles[idx])
                    break
            else:
                raise ValueError(f"No dataset covers timestamp {ts!r}")
        
        return ds_handles[0], ds_handles[1], ds_handles[0] == ds_handles[1]
