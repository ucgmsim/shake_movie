"""Write files in the layout SW4 and the workflow produce, for tests.

The image layout copies SW4's `Image::writeImagePlane_2` (checked against the
real SW4 output in `tests/data/sw4_images`): one patch of `ni * nj` values in
Fortran order, with its metadata stored as one-element datasets.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

MODES = {"topo": 18, "mag": 29, "hmax": 31}
"""SW4's integer codes for the image modes the movie reads."""


def write_image(
    directory: Path,
    prefix: str,
    mode: str,
    cycle: int,
    time: float,
    values: npt.NDArray[np.floating],
    spacing: float,
) -> Path:
    """Write one SW4 z = 0 image.

    Parameters
    ----------
    directory : Path
        SW4's output directory.
    prefix : str
        The `imagehdf5` command's `file=` prefix.
    mode : str
        The image mode, one of `MODES`.
    cycle : int
        The time step.
    time : float
        Simulation time (s).
    values : np.ndarray
        Image values, `(ni, nj)`.
    spacing : float
        Grid spacing (m).

    Returns
    -------
    Path
        The written file.
    """
    ni, nj = values.shape
    path = directory / f"{prefix}.cycle={cycle:05d}.z=0.{mode}.sw4img.h5"
    with h5py.File(path, "w") as handle:
        for name, value in {
            "npatch": np.int32(1),
            "plane": np.int32(2),
            "mode": np.int32(MODES[mode]),
            "gridinfo": np.int32(0),
            "ni": np.int32(ni),
            "nj": np.int32(nj),
        }.items():
            handle.create_dataset(name, data=np.array([value]))
        for name, value in {
            "time": time,
            "coordinate": -spacing,
            "grid_size": spacing,
            "zmin": -spacing,
        }.items():
            handle.create_dataset(name, data=np.array([value], dtype=np.float64))
        handle.create_dataset(
            "patches", data=np.asarray(values, dtype=np.float32).ravel(order="F")
        )
    return path


def write_srf(
    path: Path,
    planes: list[
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
    ],
) -> Path:
    """Write an SW4 `rupturehdf5` SRF with the fields the movie reads.

    Parameters
    ----------
    path : Path
        The SRF HDF5 file.
    planes : list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Per plane: longitude, latitude, depth (km) and rupture time (s), each
        `(ndip, nstk)` with the first row shallowest.

    Returns
    -------
    Path
        The written file.
    """
    point_dtype = np.dtype([(name, "<f4") for name in ("LON", "LAT", "DEP", "TINIT")])
    plane_dtype = np.dtype([("NSTK", "<i4"), ("NDIP", "<i4")])
    points = []
    headers = []
    for lon, lat, depth, tinit in planes:
        ndip, nstk = lon.shape
        block = np.zeros(ndip * nstk, dtype=point_dtype)
        for name, values in zip(
            ("LON", "LAT", "DEP", "TINIT"), (lon, lat, depth, tinit), strict=True
        ):
            block[name] = values.ravel()
        points.append(block)
        headers.append((nstk, ndip))
    with h5py.File(path, "w") as handle:
        handle.attrs["PLANE"] = np.array(headers, dtype=plane_dtype)
        handle.create_dataset("POINTS", data=np.concatenate(points))
    return path


def write_realisation(
    path: Path,
    name: str,
    magnitudes: dict[str, float],
    domain: list[tuple[float, float]],
    coastline: Path | None = None,
) -> Path:
    """Write the parts of a workflow realisation the movie reads.

    Parameters
    ----------
    path : Path
        The realisation JSON file.
    name : str
        Realisation name.
    magnitudes : dict[str, float]
        Moment magnitude of each fault.
    domain : list[tuple[float, float]]
        Domain corners as (latitude, longitude).
    coastline : Path | None
        nzcvm coastline file named in the velocity model's layers.

    Returns
    -------
    Path
        The written file.
    """
    layers = (
        []
        if coastline is None
        else [{"type": "coastline", "coastline": str(coastline)}]
    )
    path.write_text(
        json.dumps(
            {
                "metadata": {"name": name, "version": "1"},
                "magnitudes": {"magnitudes": magnitudes},
                "domain": {
                    "domain": [
                        {"latitude": lat, "longitude": lon} for lat, lon in domain
                    ]
                },
                "nzcvm": {"layers": layers},
            }
        )
    )
    return path


def write_sw4_input(
    path: Path,
    output_directory: Path,
    srf: Path,
    grid: dict[str, float | str],
) -> Path:
    """Write an SW4 input file with the commands the movie reads.

    Parameters
    ----------
    path : Path
        The SW4 input file.
    output_directory : Path
        SW4's `fileio` path.
    srf : Path
        The SRF.
    grid : dict[str, float | str]
        The `grid` command's parameters.

    Returns
    -------
    Path
        The written file.
    """
    grid_line = " ".join(f"{key}={value}" for key, value in grid.items())
    path.write_text(
        "\n".join(
            [
                f"fileio path={output_directory} verbose=2",
                f"grid {grid_line}",
                f"rupturehdf5 file={srf}",
                "imagehdf5 mode=topo z=0.0 file=topo cycle=0 precision=float",
                "imagehdf5 mode=mag z=0.0 file=surf_mag timeInterval=0.5 precision=float",
                "imagehdf5 mode=hmax z=0.0 file=surf_hmax precision=float",
                "",
            ]
        )
    )
    return path
