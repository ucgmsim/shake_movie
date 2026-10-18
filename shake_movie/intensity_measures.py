"""Broadband peak ground velocity from the workflow's intensity measures.

The movie's waves are SW4's low-frequency motion, which badly underestimates
how strongly small earthquakes are felt. The final map instead uses the
broadband intensity measures `im-calc` writes (`intensity_measures.h5`, an
xarray netCDF4 file): group `/PGV` holds 1-D datasets over its stations,
including `rotd50` (cm/s), `latitude`, `longitude` and `supergrid_depth_gp`.
The stations are the real ones plus a dense virtual grid over the domain.
Stations inside SW4's absorbing layer (`supergrid_depth_gp > 0`) are not
ground-motion predictions and are left out.
"""

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from pyproj import Transformer
from scipy import interpolate, spatial

from shake_movie.sw4_images import NZTM

COVERAGE_DISTANCE = 10e3
"""Furthest a point may be from its nearest station to be given a value (m).
The virtual stations are a few kilometres apart at most."""


@dataclasses.dataclass(frozen=True)
class Stations:
    """Broadband peak ground velocity at stations."""

    easting: npt.NDArray[np.float64]
    """NZTM easting of each station (m)."""
    northing: npt.NDArray[np.float64]
    """NZTM northing of each station (m)."""
    pgv: npt.NDArray[np.float64]
    """Peak ground velocity (RotD50, cm/s)."""


def read_pgv(path: Path) -> Stations:
    """Read broadband PGV at the stations SW4 simulated properly.

    Parameters
    ----------
    path : Path
        The workflow's `intensity_measures.h5`.

    Returns
    -------
    Stations
        Stations outside SW4's absorbing layer with a finite, positive PGV.

    Raises
    ------
    ValueError
        If no station is left.
    """
    with h5py.File(path, "r") as handle:
        group = handle["PGV"]
        pgv = np.asarray(group["rotd50"][:], dtype=np.float64)
        latitude = np.asarray(group["latitude"][:], dtype=np.float64)
        longitude = np.asarray(group["longitude"][:], dtype=np.float64)
        sponge = (
            np.asarray(group["supergrid_depth_gp"][:], dtype=np.float64)
            if "supergrid_depth_gp" in group
            else np.zeros_like(pgv)
        )
    # NaN depth means the solver did not say, which is not the same as inside.
    keep = (
        ~(sponge > 0)
        & np.isfinite(pgv)
        & (pgv > 0)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
    )
    if not keep.any():
        raise ValueError(f"{path} has no PGV outside SW4's absorbing layer.")
    easting, northing = Transformer.from_crs(
        "EPSG:4326", NZTM, always_xy=True
    ).transform(longitude[keep], latitude[keep])
    return Stations(np.asarray(easting), np.asarray(northing), pgv[keep])


def interpolate_pgv(
    stations: Stations,
    easting: npt.NDArray[np.float64],
    northing: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """Interpolate PGV between stations, linearly in its logarithm.

    Parameters
    ----------
    stations : Stations
        Broadband PGV at stations.
    easting : np.ndarray
        NZTM easting of the points wanted (m).
    northing : np.ndarray
        NZTM northing of the points wanted (m).

    Returns
    -------
    np.ndarray
        PGV (cm/s) at the points, shaped like `easting`; NaN outside the
        stations' coverage (beyond their hull, or `COVERAGE_DISTANCE` from
        any station).
    """
    known = np.column_stack([stations.easting, stations.northing])
    wanted = np.column_stack([np.ravel(easting), np.ravel(northing)])
    log_pgv = interpolate.LinearNDInterpolator(known, np.log10(stations.pgv))(wanted)
    distance, _ = spatial.KDTree(known).query(wanted)
    log_pgv[distance > COVERAGE_DISTANCE] = np.nan
    return (10.0**log_pgv).reshape(np.shape(easting)).astype(np.float32)
