"""Tests for reading SW4 surface images and placing them on the map.

The fixtures in `tests/data/sw4_images` were written by SW4 itself
(`input.sw4`, a 40 x 50 km grid at 1 km spacing, rotated 30 degrees, with a
Gaussian hill at (x, y) = (14, 33) km and a point source at (22, 17) km).
SW4's own `lat` and `lon` images pin down the georeferencing.
"""

from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from shake_movie import sw4_images

DATA = Path(__file__).parent / "data" / "sw4_images"
SPACING = 1000.0


@pytest.fixture
def geometry() -> sw4_images.GridGeometry:
    """The fixture grid's geometry."""
    return sw4_images.GridGeometry.from_sw4_input(DATA / "input.sw4")


def test_image_orientation_matches_sw4_axes() -> None:
    """The topography peaks over the Gaussian hill, at i = x / h, j = y / h."""
    topo = sw4_images.read_image(DATA / "topo.cycle=00.z=0.topo.sw4img.h5")
    assert topo.values.shape == (41, 51)
    assert topo.grid_size == SPACING
    i, j = np.unravel_index(np.argmax(topo.values), topo.values.shape)
    assert (i, j) == (14, 33)
    assert topo.values[i, j] == pytest.approx(1000.0, rel=1e-3)


def test_shaking_is_centred_on_the_source() -> None:
    """Early ground motion peaks above the source at (22, 17) km."""
    image = sw4_images.read_image(DATA / "surf_mag.cycle=04.z=0.mag.sw4img.h5")
    i, j = np.unravel_index(np.argmax(image.values), image.values.shape)
    assert abs(i - 22) <= 1
    assert abs(j - 17) <= 1


def test_find_images_orders_by_cycle() -> None:
    """Images are found by prefix and mode, in cycle order."""
    paths = sw4_images.find_images(DATA, "surf_mag", "mag")
    assert [path.name.split(".")[1] for path in paths] == [
        "cycle=04",
        "cycle=08",
        "cycle=16",
    ]
    assert sw4_images.find_images(DATA, "surf", "mag") == []


def test_image_sequence_interpolates_in_time() -> None:
    """Sampling between two images blends them by time."""
    paths = sw4_images.find_images(DATA, "surf_mag", "mag")
    sequence = sw4_images.ImageSequence(paths)
    first, second = (sw4_images.read_image(path) for path in paths[:2])
    middle = 0.5 * (first.time + second.time)
    np.testing.assert_allclose(
        sequence.at(middle), 0.5 * (first.values + second.values), rtol=1e-5
    )
    np.testing.assert_allclose(sequence.at(-10.0), first.values)
    assert sequence.end_time == pytest.approx(4.0)
    peak = sequence.maximum()
    assert np.all(peak >= first.values)


def test_projection_string_is_built_like_sw4(
    geometry: sw4_images.GridGeometry,
) -> None:
    """The PROJ string matches the one SW4 assembles from the grid command."""
    assert (
        geometry.projection
        == "+proj=tmerc +ellps=GRS80 +lon_0=173 +lat_0=0 +scale=0.9996"
    )
    assert geometry.azimuth == 30.0


def test_georeferencing_matches_sw4_lat_lon(
    geometry: sw4_images.GridGeometry,
) -> None:
    """Every grid point lands within float32 rounding of SW4's own lat/lon."""
    lat = sw4_images.read_image(DATA / "lat.cycle=00.z=0.lat.sw4img.h5").values
    lon = sw4_images.read_image(DATA / "lon.cycle=00.z=0.lon.sw4img.h5").values
    i, j = np.meshgrid(*map(np.arange, lat.shape), indexing="ij")
    easting, northing = geometry.to_nztm(i * SPACING, j * SPACING)
    to_nztm = Transformer.from_crs(
        sw4_images.SW4_GEOGRAPHIC_CRS, sw4_images.NZTM, always_xy=True
    )
    sw4_easting, sw4_northing = to_nztm.transform(lon, lat)
    error = np.hypot(easting - sw4_easting, northing - sw4_northing)
    assert error.max() < 2.0


def test_from_nztm_inverts_to_nztm(geometry: sw4_images.GridGeometry) -> None:
    """Grid coordinates survive a round trip through NZTM."""
    x = np.array([0.0, 12_345.0, 40_000.0])
    y = np.array([0.0, 31_000.0, 50_000.0])
    back_x, back_y = geometry.from_nztm(*geometry.to_nztm(x, y))
    np.testing.assert_allclose(back_x, x, atol=1e-3)
    np.testing.assert_allclose(back_y, y, atol=1e-3)


def test_grid_without_projection_is_rejected() -> None:
    """A grid with no map projection cannot be placed on a map."""
    with pytest.raises(ValueError, match="proj"):
        sw4_images.GridGeometry.from_grid_command({"x": "1000", "h": "100"})
