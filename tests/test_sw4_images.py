"""Tests for reading SW4 surface images and placing them on the map.

The fixtures in `tests/data/sw4_images` were written by SW4 itself
(`input.sw4`, a 40 x 50 km grid at 1 km spacing, rotated 30 degrees, with a
Gaussian hill at (x, y) = (14, 33) km and a point source at (22, 17) km).
SW4's own `lat` and `lon` images pin down the georeferencing.
"""

from pathlib import Path

import fake_sw4
import h5py
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


def _write_ramp(directory: Path, cycle: int, time: float) -> Path:
    """Write a small image whose values are `time + 10 i + j`."""
    i, j = np.meshgrid(*map(np.arange, (3, 4)), indexing="ij")
    return fake_sw4.write_image(
        directory, "surf_mag", "mag", cycle, time, time + 10.0 * i + j, SPACING
    )


def test_read_image_returns_float32_in_sw4_layout(tmp_path: Path) -> None:
    """Values come back as float32 `(ni, nj)` even from double-precision files."""
    path = _write_ramp(tmp_path, 1, 0.25)
    with h5py.File(path, "r+") as handle:
        patches = handle["patches"][:]
        del handle["patches"]
        handle.create_dataset("patches", data=patches.astype(np.float64))
    image = sw4_images.read_image(path)
    assert image.time == 0.25
    assert image.grid_size == SPACING
    assert image.values.dtype == np.float32
    assert image.values.shape == (3, 4)
    assert image.values[2, 1] == pytest.approx(0.25 + 21.0)
    assert sw4_images.read_image_time(path) == 0.25
    assert sw4_images.read_image_shape(path) == (3, 4)


@pytest.mark.parametrize(("plane", "npatch"), [(0, 1), (2, 3), (1, 2)])
def test_read_image_rejects_other_planes_and_patches(
    tmp_path: Path, plane: int, npatch: int
) -> None:
    """Only single-patch z-plane images can be read."""
    path = _write_ramp(tmp_path, 1, 0.0)
    with h5py.File(path, "r+") as handle:
        handle["plane"][0] = plane
        handle["npatch"][0] = npatch
    with pytest.raises(ValueError, match="is not a single-patch SW4 z-plane image"):
        sw4_images.read_image(path)


def test_find_images_skips_names_that_do_not_parse(tmp_path: Path) -> None:
    """Files matching the glob but not SW4's naming scheme are ignored."""
    paths = [_write_ramp(tmp_path, cycle, 0.1 * cycle) for cycle in (20, 3)]
    (tmp_path / "surf_mag.cycle=last.mag.sw4img.h5").touch()
    assert sw4_images.find_images(tmp_path, "surf_mag", "mag") == paths[::-1]


def test_image_sequence_needs_two_images(tmp_path: Path) -> None:
    """One image is rejected; two are enough to interpolate between."""
    first = _write_ramp(tmp_path, 0, 0.0)
    with pytest.raises(ValueError, match="^An image sequence needs at least two"):
        sw4_images.ImageSequence([first])
    second = _write_ramp(tmp_path, 1, 2.0)
    sequence = sw4_images.ImageSequence([second, first])
    assert sequence.paths == [first, second]
    np.testing.assert_allclose(sequence.times, [0.0, 2.0])
    np.testing.assert_allclose(
        sequence.at(0.5), sw4_images.read_image(first).values + 0.5
    )


def test_image_sequence_keeps_input_order_for_equal_times(tmp_path: Path) -> None:
    """Images sharing a time stay in the order they were given."""
    paths = [_write_ramp(tmp_path, cycle, float(cycle // 10)) for cycle in range(40)]
    given = paths[::-1]
    sequence = sw4_images.ImageSequence(given)
    expected = [
        path
        for time in range(4)
        for path in given
        if path in paths[10 * time : 10 * time + 10]
    ]
    assert sequence.paths == expected


def test_image_sequence_clamps_after_the_last_image(tmp_path: Path) -> None:
    """Times at or past the end give the last image, not an extrapolation."""
    paths = [_write_ramp(tmp_path, cycle, float(cycle)) for cycle in range(4)]
    sequence = sw4_images.ImageSequence(paths)
    last = sw4_images.read_image(paths[-1]).values
    np.testing.assert_allclose(sequence.at(3.0), last)
    np.testing.assert_allclose(sequence.at(10.0), last)
    np.testing.assert_allclose(sequence.at(2.25), last - 0.75)


def test_image_sequence_caches_the_two_nearest_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the two images nearest the requested one stay cached."""
    paths = [_write_ramp(tmp_path, cycle, float(cycle)) for cycle in range(3)]
    sequence = sw4_images.ImageSequence(paths)
    reads: list[int] = []
    read_image = sw4_images.read_image

    def counting_read_image(path: Path) -> sw4_images.SurfaceImage:
        reads.append(paths.index(path))
        return read_image(path)

    monkeypatch.setattr(sw4_images, "read_image", counting_read_image)
    base = read_image(paths[0]).values
    for time in (0.5, 1.5, 1.75, 0.25):
        np.testing.assert_allclose(sequence.at(time), base + time, rtol=1e-6)
    assert reads == [0, 1, 2, 0]


def test_parse_sw4_input_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    """Comments and blank lines are ignored; repeated commands are kept in order."""
    path = tmp_path / "input.sw4"
    path.write_text(
        "# a comment\n"
        "\n"
        "   \n"
        "grid h=100 proj=tmerc  # trailing comment\n"
        "imagehdf5 mode=mag file='a b'\n"
        "imagehdf5 mode=hmax expr=x=1\n"
    )
    assert sw4_images.parse_sw4_input(path) == {
        "grid": [{"h": "100", "proj": "tmerc"}],
        "imagehdf5": [
            {"mode": "mag", "file": "a b"},
            {"mode": "hmax", "expr": "x=1"},
        ],
    }
