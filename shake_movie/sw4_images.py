"""SW4 surface images and the grid they sit on.

SW4 writes an `imagehdf5` z-plane as one patch of `ni * nj` values in Fortran
order, with `i` along SW4's x axis and `j` along its y axis (`Image.C`). The
grid's geographic placement follows `GeographicProjection.C`: grid point
`(x, y)` maps to projected coordinates

    xmap = x sin(az) + y cos(az) + x0
    ymap = x cos(az) - y sin(az) + y0

where `(x0, y0)` is the grid origin projected with the PROJ string SW4
assembles from the `grid` command. That string is rebuilt here exactly as SW4
builds it, so the movie lands where SW4 put the simulation (including SW4
passing `scale=` through as `+scale`, which PROJ ignores for `tmerc`).
"""

import dataclasses
import functools
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import numpy.typing as npt
import parse
from pyproj import CRS, Transformer

SW4_GEOGRAPHIC_CRS = "+proj=latlong +datum=NAD83"
"""The geographic CRS SW4 projects from (`sw4_geographic_crs`)."""

NZTM = "EPSG:2193"
"""The map projection the movie is drawn in."""

IMAGE_NAME = parse.compile("{prefix}.cycle={cycle:d}.z={coordinate}.{mode}.sw4img.h5")
"""SW4's image file name (`Image::compute_file_suffix`). The cycle's zero
padding depends on the number of time steps, so files are matched by pattern."""

GRID_PROJECTION_PARAMETERS = {
    "proj": ("+proj=", False),
    "ellps": (" +ellps=", False),
    "datum": (" +datum=", False),
    "lon_p": (" +lon_0=", True),
    "lat_p": (" +lat_0=", True),
    "scale": (" +scale=", True),
}
"""How SW4's `grid` command parameters become its PROJ string, in SW4's order,
with whether SW4 reads the value as a number. Numbers are written with `:g`,
which matches the six significant digits C++ streams give `atof`'s result."""


@dataclasses.dataclass(frozen=True)
class SurfaceImage:
    """One SW4 z-plane image."""

    time: float
    """Simulation time of the image (s)."""
    values: npt.NDArray[np.float32]
    """Image values, shape `(ni, nj)`: `i` along SW4's x, `j` along SW4's y."""
    grid_size: float
    """Spacing between image points (m)."""


def read_image(path: Path) -> SurfaceImage:
    """Read an SW4 `imagehdf5` z-plane image.

    Parameters
    ----------
    path : Path
        The `*.sw4img.h5` file.

    Returns
    -------
    SurfaceImage
        The image.

    Raises
    ------
    ValueError
        If the file is not a single-patch z-plane image.
    """
    with h5py.File(path, "r") as handle:
        if int(handle["plane"][0]) != 2 or int(handle["npatch"][0]) != 1:
            raise ValueError(f"{path} is not a single-patch SW4 z-plane image.")
        ni = int(handle["ni"][0])
        nj = int(handle["nj"][0])
        values = np.asarray(handle["patches"][:], dtype=np.float32)
        return SurfaceImage(
            time=float(handle["time"][0]),
            values=values.reshape((ni, nj), order="F"),
            grid_size=float(handle["grid_size"][0]),
        )


def read_image_time(path: Path) -> float:
    """Read only the simulation time of an SW4 image.

    Parameters
    ----------
    path : Path
        The `*.sw4img.h5` file.

    Returns
    -------
    float
        The image's simulation time (s).
    """
    with h5py.File(path, "r") as handle:
        return float(handle["time"][0])


def read_image_shape(path: Path) -> tuple[int, int]:
    """Read only the shape of an SW4 image.

    Parameters
    ----------
    path : Path
        The `*.sw4img.h5` file.

    Returns
    -------
    tuple[int, int]
        The image's shape, `(ni, nj)`.
    """
    with h5py.File(path, "r") as handle:
        return int(handle["ni"][0]), int(handle["nj"][0])


def find_images(directory: Path, prefix: str, mode: str) -> list[Path]:
    """Find every image SW4 wrote for one `imagehdf5` command, in cycle order.

    Parameters
    ----------
    directory : Path
        SW4's output directory (its `fileio` path).
    prefix : str
        The command's `file=` prefix, e.g. `surf_mag`.
    mode : str
        The image mode's file suffix, e.g. `mag`.

    Returns
    -------
    list[Path]
        The image files, ordered by cycle.
    """
    matches = []
    for path in directory.glob(f"{prefix}.cycle=*.{mode}.sw4img.h5"):
        name = IMAGE_NAME.parse(path.name)
        if isinstance(name, parse.Result) and name["prefix"] == prefix:
            matches.append((name["cycle"], path))
    return [path for _, path in sorted(matches)]


class ImageSequence:
    """A time series of SW4 images, sampled at any time by interpolation.

    Parameters
    ----------
    paths : Sequence[Path]
        Image files from a single `imagehdf5` command.

    Raises
    ------
    ValueError
        If fewer than two images are given.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        """Index a sequence of images by their simulation times."""
        if len(paths) < 2:
            raise ValueError("An image sequence needs at least two images.")
        times = np.array([read_image_time(path) for path in paths])
        order = np.argsort(times, kind="stable")
        self.paths = [paths[i] for i in order]
        self.times = times[order]
        self._cache: dict[int, npt.NDArray[np.float32]] = {}

    def _values(self, index: int) -> npt.NDArray[np.float32]:
        """Read one image, keeping the two most recent in memory."""
        if index not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(max(self._cache, key=lambda k: abs(k - index)))
            self._cache[index] = read_image(self.paths[index]).values
        return self._cache[index]

    @property
    def end_time(self) -> float:
        """When the sequence ends.

        Returns
        -------
        float
            The time of the last image (s).
        """
        return float(self.times[-1])

    def at(self, time: float) -> npt.NDArray[np.float32]:
        """Interpolate the image linearly in time.

        Parameters
        ----------
        time : float
            Simulation time (s), clamped to the sequence's span.

        Returns
        -------
        np.ndarray
            The interpolated image, shape `(ni, nj)`.
        """
        time = float(np.clip(time, self.times[0], self.times[-1]))
        index = int(
            np.clip(np.searchsorted(self.times, time) - 1, 0, len(self.times) - 2)
        )
        t0, t1 = self.times[index], self.times[index + 1]
        weight = 0.0 if t1 == t0 else (time - t0) / (t1 - t0)
        before = self._values(index)
        after = self._values(index + 1)
        return before + np.float32(weight) * (after - before)

    def maximum(self) -> npt.NDArray[np.float32]:
        """Take the pointwise maximum over every image.

        Returns
        -------
        np.ndarray
            The maximum image, shape `(ni, nj)`.
        """
        peak = read_image(self.paths[0]).values.copy()
        for path in self.paths[1:]:
            np.maximum(peak, read_image(path).values, out=peak)
        return peak


def parse_sw4_input(path: Path) -> dict[str, list[dict[str, str]]]:
    """Read an SW4 input file into a map of command name to its occurrences.

    Parameters
    ----------
    path : Path
        The SW4 input file.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Each command's parameters, in file order, keyed by command name.
    """
    commands: dict[str, list[dict[str, str]]] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, *parameters = shlex.split(line)
        commands.setdefault(name, []).append(
            dict(parameter.split("=", 1) for parameter in parameters)
        )
    return commands


@dataclasses.dataclass(frozen=True)
class GridGeometry:
    """Where SW4's grid sits on the Earth."""

    lon: float
    """Longitude of the grid origin."""
    lat: float
    """Latitude of the grid origin."""
    azimuth: float
    """Bearing of SW4's x axis, degrees clockwise from north."""
    projection: str
    """The PROJ string SW4 projects with."""

    @classmethod
    def from_grid_command(cls, grid: dict[str, str]) -> Self:
        """Build the geometry from the parameters of SW4's `grid` command.

        Parameters
        ----------
        grid : dict[str, str]
            The `grid` command's parameters.

        Returns
        -------
        GridGeometry
            The grid's geometry.

        Raises
        ------
        ValueError
            If the grid has no map projection or origin.
        """
        if "proj" not in grid or "lon" not in grid or "lat" not in grid:
            raise ValueError(
                "The SW4 grid command must set proj, lon and lat to place the"
                " movie on a map."
            )
        projection = "".join(
            prefix + (f"{float(grid[key]):g}" if numeric else grid[key])
            for key, (prefix, numeric) in GRID_PROJECTION_PARAMETERS.items()
            if key in grid
        )
        return cls(
            lon=float(grid["lon"]),
            lat=float(grid["lat"]),
            azimuth=float(grid.get("az", 0.0)),
            projection=projection,
        )

    @classmethod
    def from_sw4_input(cls, path: Path) -> Self:
        """Build the geometry from an SW4 input file.

        Parameters
        ----------
        path : Path
            The SW4 input file.

        Returns
        -------
        GridGeometry
            The grid's geometry.
        """
        return cls.from_grid_command(parse_sw4_input(path)["grid"][0])

    @functools.cached_property
    def _transformers(self) -> tuple[Transformer, Transformer, tuple[float, float]]:
        """Transformers between SW4's projection and NZTM, and the origin."""
        crs = CRS.from_proj4(self.projection)
        to_sw4 = Transformer.from_crs(
            CRS.from_proj4(SW4_GEOGRAPHIC_CRS), crs, always_xy=True
        )
        origin = to_sw4.transform(self.lon, self.lat)
        sw4_to_nztm = Transformer.from_crs(crs, NZTM, always_xy=True)
        nztm_to_sw4 = Transformer.from_crs(NZTM, crs, always_xy=True)
        return sw4_to_nztm, nztm_to_sw4, origin

    def to_nztm(
        self, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map SW4 grid coordinates to NZTM.

        Parameters
        ----------
        x : array-like
            SW4 x coordinates (m).
        y : array-like
            SW4 y coordinates (m).

        Returns
        -------
        easting : np.ndarray
            NZTM easting (m).
        northing : np.ndarray
            NZTM northing (m).
        """
        sw4_to_nztm, _, (x0, y0) = self._transformers
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        az = np.radians(self.azimuth)
        xmap = x * np.sin(az) + y * np.cos(az) + x0
        ymap = x * np.cos(az) - y * np.sin(az) + y0
        easting, northing = sw4_to_nztm.transform(xmap, ymap)
        return np.asarray(easting), np.asarray(northing)

    def from_nztm(
        self, easting: npt.ArrayLike, northing: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map NZTM coordinates to SW4 grid coordinates.

        Parameters
        ----------
        easting : array-like
            NZTM easting (m).
        northing : array-like
            NZTM northing (m).

        Returns
        -------
        x : np.ndarray
            SW4 x coordinates (m).
        y : np.ndarray
            SW4 y coordinates (m).
        """
        _, nztm_to_sw4, (x0, y0) = self._transformers
        xmap, ymap = nztm_to_sw4.transform(
            np.asarray(easting, dtype=np.float64),
            np.asarray(northing, dtype=np.float64),
        )
        dx = np.asarray(xmap) - x0
        dy = np.asarray(ymap) - y0
        az = np.radians(self.azimuth)
        return dx * np.sin(az) + dy * np.cos(az), dx * np.cos(az) - dy * np.sin(az)
