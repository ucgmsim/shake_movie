"""Tests for preparing the scene."""

import gzip
from pathlib import Path

import fake_sw4
import numpy as np
import pytest
import shapely

from shake_movie import scene
from shake_movie.realisation import Realisation
from shake_movie.sw4_images import GridGeometry, read_image

DATA = Path(__file__).parent / "data" / "sw4_images"


def test_rupture_follows_the_srf(sw4_run: dict[str, Path]) -> None:
    """The fault is the SRF's top row, rupturing when its column first slips."""
    rupture = scene.Rupture.from_srf(sw4_run["srf"])
    assert len(rupture.segments) == 1
    line, start = rupture.segments[0]
    assert line.shape == (11, 2)
    assert start[5] == pytest.approx(0.0)
    np.testing.assert_allclose(rupture.hypocentre, line[5], atol=1.0)
    assert rupture.end_time == pytest.approx(5000 / 2800, rel=1e-5)
    assert not rupture.is_point
    assert rupture.ruptured(0.0) == []
    (middle,) = rupture.ruptured(1.0)
    assert len(middle) == 5


def test_tiny_ruptures_are_points(tmp_path: Path) -> None:
    """A rupture shorter than a few kilometres is drawn as an epicentre."""
    lon = np.array([[172.0, 172.001]])
    srf = fake_sw4.write_srf(
        tmp_path / "point.h5",
        [(lon, np.full_like(lon, -43.0), np.ones_like(lon), np.zeros_like(lon))],
    )
    assert scene.Rupture.from_srf(srf).is_point


def test_multiple_planes_become_segments(tmp_path: Path) -> None:
    """Each SRF plane becomes its own fault segment."""
    planes = []
    for start in (172.0, 172.2):
        lon = np.tile(np.linspace(start, start + 0.15, 6), (3, 1))
        lat = np.full_like(lon, -42.5)
        planes.append(
            (
                lon,
                lat,
                np.ones_like(lon),
                np.arange(lon.size, dtype=float).reshape(lon.shape),
            )
        )
    rupture = scene.Rupture.from_srf(fake_sw4.write_srf(tmp_path / "two.h5", planes))
    assert len(rupture.segments) == 2
    assert all(len(line) == 6 for line, _ in rupture.segments)


def test_gmt_and_wkb_coastlines_agree(tmp_path: Path, geometry: GridGeometry) -> None:
    """Both coastline formats load as NZTM land polygons."""
    from conftest import write_square_coastline

    gmt = scene.load_coastline(write_square_coastline(tmp_path / "coast.gmt", geometry))
    wkb = tmp_path / "coast.wkb.gz"
    with gzip.open(wkb, "wb") as handle:
        handle.write(shapely.to_wkb(gmt))
    assert scene.load_coastline(wkb).equals(gmt)
    assert gmt.area == pytest.approx(30e3 * 60e3, rel=0.01)
    min_e, min_n, _, _ = gmt.bounds
    assert 1.4e6 < min_e < 1.6e6 and 5.1e6 < min_n < 5.2e6


def test_towns_are_loaded_largest_first() -> None:
    """The vendored town list loads in NZTM, largest first."""
    towns = scene.load_towns()
    assert towns[0].population >= towns[-1].population
    christchurch = next(town for town in towns if town.name == "Christchurch")
    assert christchurch.easting == pytest.approx(1570700, abs=3000)
    assert christchurch.northing == pytest.approx(5180200, abs=3000)
    assert any(town.name == "Kaikōura" for town in towns)


def test_lakes_are_large_zero_patches_on_land() -> None:
    """Lagoons at 0 m inside the coastline are lakes; the sea and puddles are not."""
    elevation = np.full((60, 60), 50.0, dtype=np.float32)
    land = np.ones_like(elevation)
    land[:, 50:] = 0.0
    elevation[:, 50:] = 0.0
    elevation[10:35, 10:35] = 0.0
    elevation[40:42, 40:42] = 0.0
    lake = scene.lakes(elevation, land, cell_area=100.0**2)
    assert lake[20, 20] > 0.9
    assert lake[41, 41] < 0.1
    assert lake[20, 55] < 0.1


def test_scene_on_sw4_output(sw4_run: dict[str, Path], geometry: GridGeometry) -> None:
    """The scene lines SW4's terrain, the domain and the map up."""
    topo = read_image(DATA / "topo.cycle=00.z=0.topo.sw4img.h5")
    info = Realisation.read(sw4_run["realisation"])
    built = scene.build_scene(
        geometry,
        topo.values,
        topo.grid_size,
        info.domain,
        scene.load_coastline(sw4_run["coastline"]),
        scene.Rupture.from_srf(sw4_run["srf"]),
        "Mw 6.0 earthquake",
        "Somewhere",
        False,
    )
    assert built.terrain_texture.shape == (*topo.values.shape, 3)
    # The domain covers x 10-30 km, y 10-40 km of the 1 km grid.
    assert built.wave_inside.sum() == 21 * 31
    hill_e, hill_n = geometry.to_nztm(14e3, 33e3)
    # Draping heights are resampled at 500 m, which rounds off the 2 km wide peak.
    assert built.height_at(hill_e, hill_n) == pytest.approx(1000.0, rel=0.15)
    assert built.height_at(*geometry.to_nztm(14e3, 38e3)) < 200.0
    sea_e, sea_n = geometry.to_nztm(35e3, 33e3)
    assert built.height_at(sea_e, sea_n) == 0.0
    # Shaking is faint at sea and absent outside the domain.
    levels = np.array([0.0, scene.palette.SEA_SHAKING_OPACITY, 1.0])
    assert np.isclose(built.wave_opacity[..., None], levels).any(axis=-1).all()
    assert (built.wave_opacity == 1.0).any() and np.isclose(
        built.wave_opacity, levels[1]
    ).any()
