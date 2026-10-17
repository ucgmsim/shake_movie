"""Shared fixtures: a small SW4 run built around SW4's own output files."""

import shutil
from pathlib import Path

import fake_sw4
import numpy as np
import pytest
from pyproj import Transformer

from shake_movie.sw4_images import GridGeometry

DATA = Path(__file__).parent / "data" / "sw4_images"
SPACING = 1000.0


@pytest.fixture
def geometry() -> GridGeometry:
    """The geometry of SW4's fixture grid (40 x 50 km, rotated 30 degrees)."""
    return GridGeometry.from_sw4_input(DATA / "input.sw4")


def _lon_lat(geometry: GridGeometry, x: float, y: float) -> tuple[float, float]:
    """Longitude and latitude of an SW4 grid point."""
    easting, northing = geometry.to_nztm(x, y)
    lon, lat = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True).transform(
        easting, northing
    )
    return float(lon), float(lat)


def write_square_coastline(path: Path, geometry: GridGeometry) -> Path:
    """Write a GMT coastline making the grid's x < 25 km side land."""
    corners = [(-5e3, -5e3), (25e3, -5e3), (25e3, 55e3), (-5e3, 55e3), (-5e3, -5e3)]
    lines = ["# test coastline", ">"]
    lines += [
        " ".join(f"{v:.6f}" for v in _lon_lat(geometry, x, y)) for x, y in corners
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def sw4_run(tmp_path: Path, geometry: GridGeometry) -> dict[str, Path]:
    """A complete SW4 run: SW4's fixture images, an SRF, a realisation and
    an input file pointing at them."""
    images = tmp_path / "images"
    images.mkdir()
    for path in DATA.glob("*.sw4img.h5"):
        if path.name.split(".")[0] in ("topo", "surf_mag", "surf_hmax"):
            shutil.copy(path, images / path.name)
    # A 10 km fault through the source at (22, 17) km, rupturing from its middle.
    y = np.linspace(12e3, 22e3, 11)
    lon, lat = np.transpose([_lon_lat(geometry, 22e3, value) for value in y])
    depth = np.array([1.0, 5.0])[:, None] * np.ones((1, y.size))
    tinit = np.abs(y - 17e3)[None, :] / 2800.0 + 0.0 * depth
    srf = fake_sw4.write_srf(
        tmp_path / "rupture.srf.h5",
        [(np.tile(lon, (2, 1)), np.tile(lat, (2, 1)), depth, tinit)],
    )
    corners = [
        _lon_lat(geometry, x, y)
        for x, y in [(10e3, 10e3), (30e3, 10e3), (30e3, 40e3), (10e3, 40e3)]
    ]
    coastline = write_square_coastline(tmp_path / "coast.gmt", geometry)
    realisation = fake_sw4.write_realisation(
        tmp_path / "realisation.json",
        "test_run",
        {"fault": 6.0},
        [(lat, lon) for lon, lat in corners],
        coastline,
    )
    grid = {
        "x": 40000,
        "y": 50000,
        "z": 24000,
        "h": 1000,
        "lat": -43.5,
        "lon": 172.5,
        "az": 30,
        "proj": "tmerc",
        "ellps": "GRS80",
        "lon_p": 173,
        "lat_p": 0,
        "scale": 0.9996,
    }
    sw4_input = fake_sw4.write_sw4_input(tmp_path / "sw4.in", images, srf, grid)
    return {
        "sw4_input": sw4_input,
        "realisation": realisation,
        "coastline": coastline,
        "srf": srf,
        "images": images,
    }
