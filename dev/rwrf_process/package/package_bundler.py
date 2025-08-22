import logging
import os
import pathlib
import shutil
from typing import Any, List, Protocol

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import SingleQuotedScalarString

from .metadata import create_metadata

logger = logging.getLogger(__name__)


class IFileWriter(Protocol):
    """
    Interface defining file system operations.
    Implementations should provide methods for writing files, copying files,
    zipping directories, creating directories, and removing directories.
    """

    def write(self, path: str, data: Any) -> None:
        """Write data to the specified file path."""
        ...

    def copy(self, src: str, dst: str) -> None:
        """Copy a file from src to dst."""
        ...

    def zipdir(self, src_dir: str, zip_path: str) -> None:
        """Zip the contents of src_dir into a zip file at zip_path."""
        ...

    def makedirs(self, path: str) -> None:
        """Create directories recursively at the specified path."""
        ...

    def rmdir(self, path: str) -> None:
        """Remove a directory and all its contents."""
        ...


class IPackageComponent(Protocol):
    """
    Interface defining a package component.
    """

    def save(self, writer: IFileWriter, location: str) -> None:
        """Save the component to the specified location using the provided writer."""
        ...


class LocalFileWriter:
    """Local Filesystem Writer implementation of IFileWriter."""

    def write(self, path: str, data: Any) -> None:
        with open(path, "w" if isinstance(data, str) else "wb") as f:
            f.write(data)

    def copy(self, src: str, dst: str) -> None:
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(src_path.read_bytes())

    def zipdir(self, src_dir: str, zip_path: str) -> None:
        shutil.make_archive(zip_path, "zip", src_dir)

    def makedirs(self, path: str) -> None:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    def rmdir(self, path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)


class FileComponent:
    """Component for copying a single file into the package.

    Parameters
    ----------
    src_path : str
        Path to the source file to copy.
    dst_filename : str
        Destination filename within the package.
    """

    def __init__(self, src_path: str, dst_filename: str):
        self._src_path = src_path
        self._dst_filename = dst_filename

    def save(self, writer: IFileWriter, location: str) -> None:
        dst_path = os.path.join(location, self._dst_filename)
        writer.copy(self._src_path, dst_path)


class ZipDirComponent:
    """Component for zipping the specified source directory and saves it as a zip file in the package location.
    Parameters
    ----------
    src_dir : str
        Path to the source directory to zip.
    dst_zip_name : str, optional
        Name of the resulting zip file in the package (default: "metadata.zarr").
    """

    def __init__(self, src_dir: str, dst_zip_name: str = "metadata.zarr"):
        self._src_dir = src_dir
        self._dst_zip_name = dst_zip_name

    def save(self, writer: IFileWriter, location: str) -> None:
        dst_dir = os.path.join(location, self._dst_zip_name)
        writer.zipdir(self._src_dir, dst_dir)


class YamlComponent:
    """Component for writing a YAML file from a file path or string, with the ability to  update/remove values from a dictionary.

    Parameters
    ----------
    yaml_data : str
        YAML content as a string, or a path to a YAML file.
    update_vals : dict, optional
        Dictionary of values to update in the YAML before saving.
    remove_vals : dict, optional
        Dictionary of values to remove from the YAML before saving.
    """

    def __init__(
        self, yaml_data: str, update_vals: dict = None, remove_vals: dict = None
    ):
        self._data = yaml_data
        self._update_vals = update_vals or {}
        self._remove_vals = remove_vals or {}

    @staticmethod
    def _flow_style_list(obj: Any) -> CommentedSeq | dict | list | None:
        """Recursively set flow style for all lists in the object and single-quote all elements."""
        if isinstance(obj, list):
            seq = CommentedSeq()
            seq.fa.set_flow_style()
            for v in obj:
                if isinstance(v, str):
                    seq.append(SingleQuotedScalarString(v))
                else:
                    seq.append(YamlComponent._flow_style_list(v))
            return seq
        elif isinstance(obj, dict):
            return {k: YamlComponent._flow_style_list(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def _update(d: dict, u: dict) -> None:
        """Recursively update dictionary d with values from u."""
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                YamlComponent._update(d[k], v)
            else:
                d[k] = v

    @staticmethod
    def _remove(d: dict, r: dict) -> None:
        """Recursively remove keys in r from d."""
        for k in r.keys():
            if k in d:
                if isinstance(d[k], dict) and isinstance(r[k], dict):
                    YamlComponent._remove(d[k], r[k])
                else:
                    del d[k]

    def save(self, writer: "IFileWriter", location: str) -> None:
        """
        Save the YAML file to the specified location, update/remove values as needed.

        Parameters
        ----------
        writer : IFileWriter
            The file writer to use for saving.
        location : str
            The root directory where the YAML file should be saved.
        """
        path = os.path.join(location, "model.yaml")
        yaml = YAML()
        yaml.width = 4096

        if os.path.isfile(self._data):
            with open(self._data, "r") as f:
                data = yaml.load(f)
        else:
            data = yaml.load(self._data)

        self._update(data, self._update_vals)
        self._remove(data, self._remove_vals)

        data = self._flow_style_list(data)

        from io import StringIO

        buf = StringIO()
        yaml.dump(data, buf)
        writer.write(path, buf.getvalue())


class ZarrStore:
    """Create a Zarr store for model metadata.

    Parameters
    ----------
    variable : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to variables to return. Must be in the RWRF lexicon.
    conditioning_variable : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to conditioning variables to return. Must be in the RWRF lexicon.
    invariant : str | list[str] | np.ndarray[str]
        String, list of strings or array of strings that refer to invariant to return. Must be in the RWRF lexicon.
    y : int
        The number of latitude grid points.
    x : int
        The number of longitude grid points.
    variable_file_path : str | None
        The file path for the variable data.
    conditioning_variable_file_path : str | None
        The file path for the conditioning variable data.
    invariant_file_path : str | None
        The file path for the invariant data.
    """

    def __init__(
        self,
        variable: str | list[str] | np.ndarray[str],
        conditioning_variable: str | list[str] | np.ndarray[str],
        invariant: str | list[str] | np.ndarray[str],
        y: int = 32,
        x: int = 32,
        variable_file_path: str | None = None,
        conditioning_variable_file_path: str | None = None,
        invariant_file_path: str | None = None,
    ):
        self.params = locals()
        self.params.pop("self")

    def create(self, target_path: str) -> None:
        """Creates the Zarr store at the given path."""
        ds = create_metadata(**self.params)
        ds.to_zarr(
            target_path,
            mode="w",
            consolidated=True,
            zarr_format=2,
        )


class PackageBundler:
    """Custom Package Bundler. Bundles up components into a package at location.

    Parameters
    ----------
    location : str
        The location where the package will be created.
    writer : IFileWriter
        The file writer to use for saving components.
    """

    def __init__(
        self,
        location: str,
        writer: IFileWriter = None,
    ):
        self.location = location
        self.writer = writer or LocalFileWriter()
        self.components: List[IPackageComponent] = []
        self.writer.makedirs(self.location)

    def add_component(self, component: IPackageComponent) -> None:
        """Adds a component to the package.

        Parameters
        ----------
        component : IPackageComponent
            The component to add to the package.
        """
        self.components.append(component)

    def build(self) -> None:
        """Executes the build process by saving all added components."""
        if not self.components:
            raise ValueError(
                "No components have been added to the package. Cannot build an empty package."
            )

        for component in self.components:
            component.save(self.writer, self.location)

        logger.info(f"Package successfully built at: {self.location}")

    def __call__(
        self,
        variable: str | list[str] | np.ndarray[str],
        conditioning_variable: str | list[str] | np.ndarray[str],
        invariant: str | list[str] | np.ndarray[str],
        y: int = 32,
        x: int = 32,
        checkpoints_src: str = "../../examples/weather/stormcast/rundir/stormcast-training/0/",
        regression_start_step: int = 0,
        regression_end_step: int = 160,
        diffusion_start_step: int = 0,
        diffusion_end_step: int = 100,
    ) -> None:
        """Prepares and adds all required components (edmprecond, storcastunet, metadata.zarr, model.yaml) for an e2s package, then builds the package.

        The following directory structure is expected for checkpoints_src:
        | <checkpoints_src>
        | -- .hydra
        | -- checkpoints_diffusion
            | checkpoint.<start_step>.<end_step>.pt
            | EDMPrecond.<start_step>.<end_step>.mdlus
        | -- checkpoints_diffusion
            | checkpoint.<start_step>.<end_step>.pt
            | StormCastUNet.<start_step>.<end_step>.mdlus

        Builds a package structured like below:
        | <location>
        | -- EDMPrecond.0.0.mdlus
        | -- StormCastUNet.0.0.mdlus
        | -- metadata.zarr.zip
        | -- model.yaml

        Parameters
        ----------
        variable : str | list[str] | np.ndarray[str]
            String, list of strings or array of strings that refer to variables to return. Must be in the RWRF lexicon.
        conditioning_variable : str | list[str] | np.ndarray[str]
            String, list of strings or array of strings that refer to conditioning variables to return. Must be in the RWRF lexicon.
        invariant : str | list[str] | np.ndarray[str]
            String, list of strings or array of strings that refer to invariant to return. Must be in the RWRF lexicon.
        y : int
            The number of latitude grid points.
        x : int
            The number of longitude grid points.
        checkpoints_src : str
            The source directory containing the checkpoint files. <path to>/stormcast/rundir/stormcast-training/0/
        regression_start_step : int
            The starting step for the regression checkpoints.
        regression_end_step : int
            The ending step for the regression checkpoints.
        diffusion_start_step : int
            The starting step for the diffusion checkpoints.
        diffusion_end_step : int
            The ending step for the diffusion checkpoints.
        """

        self.variable, self.conditioning_variable, self.invariant = (
            variable,
            conditioning_variable,
            invariant,
        )
        self.y, self.x = y, x

        stormcastunet_path = (
            pathlib.Path(checkpoints_src)
            / "checkpoints_regression"
            / f"StormCastUNet.{regression_start_step}.{regression_end_step}.mdlus"
        )  # regression checkpoint
        self.add_component(
            FileComponent(
                stormcastunet_path,
                "StormCastUNet.0.0.mdlus",
            )
        )

        edmprecond_path = (
            pathlib.Path(checkpoints_src)
            / "checkpoints_diffusion"
            / f"EDMPrecond.{diffusion_start_step}.{diffusion_end_step}.mdlus"
        )  # diffusion checkpoint
        self.add_component(
            FileComponent(
                edmprecond_path,
                "EDMPrecond.0.0.mdlus",
            )
        )

        store_path = "./tmp/"
        ZarrStore(
            variable=self.variable,
            conditioning_variable=self.conditioning_variable,
            invariant=self.invariant,
            y=self.y,
            x=self.x,
            variable_file_path="../data/HighRes/stats/",
            conditioning_variable_file_path="../data/LowRes/stats/",
            invariant_file_path="../data/invariants/invariants.zarr/",
        ).create(store_path)
        self.add_component(ZipDirComponent(store_path))

        self.add_component(
            YamlComponent(
                "./package/model.yaml",
                update_vals={
                    "data": {
                        "local_img_size": [self.y, self.x],
                        "n_local_channels": len(self.variable),
                        "n_conditioning_channels": len(self.conditioning_variable),
                        "n_invariant": len(self.invariant),
                        "local_channels": self.variable,
                        "conditioning_channels": self.conditioning_variable,
                        "invariants": self.invariant,
                    },
                    "regression_model": {
                        "img_resolution": self.y,
                        "model": {
                            "img_resolution": self.y,
                            "rwrf_resolution": [self.y, self.x],
                        },
                    },
                    "diffusion_model": {
                        "img_resolution": self.y,
                        "model": {
                            "img_resolution": self.y,
                            "rwrf_resolution": [self.y, self.x],
                        },
                    },
                },
                remove_vals={
                    "regression_model": {
                        "model": {
                            "hrrr_resolution": [self.y, self.x],
                        },
                    },
                    "diffusion_model": {
                        "model": {
                            "hrrr_resolution": [self.y, self.x],
                        },
                    },
                },
            )
        )
        self.build()
        self.writer.rmdir(store_path)
