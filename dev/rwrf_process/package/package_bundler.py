import logging
import os
import pathlib
import shutil
from typing import Any, List, Protocol

import numpy as np

from .metadata import create_metadata

logger = logging.getLogger(__name__)


class IFileWriter(Protocol):
    """Interface for abstracting file system operations."""

    def write(self, path: str, data: Any) -> None: ...
    def copy(self, src: str, dst: str) -> None: ...
    def zipdir(self, src_dir: str, zip_path: str) -> None: ...
    def makedirs(self, path: str) -> None: ...
    def rmdir(self, path: str) -> None: ...


class IPackageComponent(Protocol):
    """Interface defining a component of the package that can save itself."""

    def save(self, writer: IFileWriter, location: str) -> None: ...


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
    """A generic component for copying a file."""

    def __init__(self, src_path: str, dst_filename: str):
        self._src_path = src_path
        self._dst_filename = dst_filename

    def save(self, writer: IFileWriter, location: str) -> None:
        dst_path = os.path.join(location, self._dst_filename)
        writer.copy(self._src_path, dst_path)


class ZipDirComponent:
    """A generic component for zipping a directory."""

    def __init__(self, src_dir: str, dst_zip_name: str = "metadata.zarr"):
        self._src_dir = src_dir
        self._dst_zip_name = dst_zip_name

    def save(self, writer: IFileWriter, location: str) -> None:
        dst_dir = os.path.join(location, self._dst_zip_name)
        writer.zipdir(self._src_dir, dst_dir)


class YamlComponent:
    """A component for writing a YAML file from a string or file path."""

    def __init__(self, yaml_data: str):
        self.yaml_data = yaml_data

    def save(self, writer: IFileWriter, location: str) -> None:
        path = os.path.join(location, "model.yaml")
        if os.path.isfile(self.yaml_data):
            writer.copy(self.yaml_data, path)
        else:
            writer.write(path, self.yaml_data)


class ZarrStore:
    """Create a Zarr store for model metadata."""

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
    """
    A high-level director that orchestrates the creation of a model package.

    This class provides a simple, clean API to the user, hiding the
    complexity of component assembly and the build process.
    | <location>
    | -- EDMPrecond.0.0.mdlus
    | -- StormCastUNet.0.0.mdlus
    | -- metadata.zarr.zip
    | -- model.yaml

    Parameters
    ----------
    location : str
        The location where the package will be created.
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
    writer : IFileWriter
        The file writer to use for saving components.
    """

    def __init__(
        self,
        location: str,
        variable: str | list[str] | np.ndarray[str],
        conditioning_variable: str | list[str] | np.ndarray[str],
        invariant: str | list[str] | np.ndarray[str],
        y: int = 32,
        x: int = 32,
        writer: IFileWriter = None,
    ):
        self.location = location
        self.writer = writer or LocalFileWriter()
        self.components: List[IPackageComponent] = []
        self.writer.makedirs(self.location)

        self.variable, self.conditioning_variable, self.invariant = (
            variable,
            conditioning_variable,
            invariant,
        )
        self.y, self.x = y, x

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
        checkpoints_src: str = "../../examples/weather/stormcast/rundir/stormcast-training/0/",
        regression_start_step: int = 0,
        regression_end_step: int = 160,
        diffusion_start_step: int = 0,
        diffusion_end_step: int = 100,
    ) -> None:
        """Allows the builder to be called like a function.
        | <checkpoints_src>
        | -- .hydra
        | -- checkpoints_diffusion
            | checkpoint.<start_step>.<end_step>.pt
            | EDMPrecond.<start_step>.<end_step>.mdlus
        | -- checkpoints_diffusion
            | checkpoint.<start_step>.<end_step>.pt
            | StormCastUNet.<start_step>.<end_step>.mdlus

        Parameters
        ----------
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

        self.add_component(YamlComponent("./package/model.yaml"))
        self.build()
        self.writer.rmdir(store_path)
