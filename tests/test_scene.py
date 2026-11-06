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


def _slipping_plane(
    tmp_path: Path, nstk: int = 4, ndip: int = 3
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write an SRF plane whose points slip 1 m more per column, one second
    after another, each in a 1 s pulse."""
    lon = np.tile(np.linspace(172.0, 172.1, nstk), (ndip, 1))
    lat = np.tile(np.linspace(-43.0, -43.02, ndip)[:, None], (1, nstk))
    tinit = np.tile(np.arange(nstk, dtype=float), (ndip, 1))
    slip = np.tile(np.arange(1.0, nstk + 1), (ndip, 1))
    srf = fake_sw4.write_srf(
        tmp_path / "slip.h5",
        [(lon, lat, np.ones_like(lon), tinit)],
        slip=[(slip, np.ones_like(slip))],
    )
    return srf, tinit, slip


def test_slip_follows_each_points_pulse(tmp_path: Path) -> None:
    """Each cell slips from its onset through its pulse to its total."""
    srf, _, slip = _slipping_plane(tmp_path)
    history = scene.Rupture.from_srf(srf).slip
    assert history is not None
    (corners,) = history.corners
    assert corners.shape == (4, 5, 2)
    assert history.peak == pytest.approx(4.0)
    np.testing.assert_allclose(history.at(-1.0), 0.0)
    np.testing.assert_allclose(history.at(100.0), slip.ravel(), rtol=1e-4)
    # Halfway through a symmetric pulse, half the slip is done.
    midway = history.at(1.5).reshape(slip.shape)
    np.testing.assert_allclose(midway[:, 1], slip[:, 1] / 2, rtol=0.05)
    np.testing.assert_allclose(midway[:, 0], slip[:, 0], rtol=1e-4)
    np.testing.assert_allclose(midway[:, 2:], 0.0)


def test_large_srfs_slip_in_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past `MAX_SLIP_CELLS` points, cells are blocks with their mean slip."""
    monkeypatch.setattr(scene, "MAX_SLIP_CELLS", 4)
    srf, _, _ = _slipping_plane(tmp_path, nstk=6, ndip=4)
    history = scene.Rupture.from_srf(srf).slip
    assert history is not None
    (corners,) = history.corners
    # 24 points over at most 4 cells: blocks of 3 x 3, the last ones short.
    assert corners.shape == (3, 3, 2)
    np.testing.assert_allclose(history.at(100.0), [2.0, 5.0, 2.0, 5.0], rtol=1e-4)
    assert history.peak == pytest.approx(5.0)


def test_srfs_without_slip_rates_have_no_slip(tmp_path: Path) -> None:
    """An SRF with no slip-rate pulses still gives the fault, without slip."""
    lon = np.tile(np.linspace(172.0, 172.1, 4), (2, 1))
    srf = fake_sw4.write_srf(
        tmp_path / "plain.h5",
        [(lon, np.full_like(lon, -43.0), np.ones_like(lon), np.zeros_like(lon))],
    )
    assert scene.Rupture.from_srf(srf).slip is None


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


def test_planes_outline_their_surface_projection(tmp_path: Path) -> None:
    """Each plane's outline runs along its top edge and back along its bottom."""
    lon = np.tile(np.linspace(172.0, 172.15, 6), (3, 1))
    lat = np.array([[-42.5], [-42.52], [-42.54]]) + np.zeros_like(lon)
    srf = fake_sw4.write_srf(
        tmp_path / "dipping.h5",
        [(lon, lat, np.arange(3.0)[:, None] + np.zeros_like(lon), np.zeros_like(lon))],
    )
    rupture = scene.Rupture.from_srf(srf)
    (outline,) = rupture.planes
    line, _ = rupture.segments[0]
    assert outline.shape == (12, 2)
    np.testing.assert_allclose(outline[:6], line)
    # The bottom edge comes back the other way, south of the top.
    assert outline[6, 0] > outline[11, 0]
    assert np.all(outline[6:, 1] < outline[:6, 1].min())


def test_footprint_outlines_neighbouring_planes_as_one(tmp_path: Path) -> None:
    """Planes meeting at a bend, with a small gap, share one outline."""
    planes = []
    for start, end in ((172.0, 172.15), (172.1505, 172.3)):
        lon = np.tile(np.linspace(start, end, 6), (3, 1))
        lat = np.array([[-42.5], [-42.52], [-42.54]]) + np.zeros_like(lon)
        if start > 172.0:
            lat = lat + np.linspace(0.0, 0.05, 6)
        planes.append(
            (lon, lat, np.arange(3.0)[:, None] + np.zeros_like(lon), np.zeros_like(lon))
        )
    rupture = scene.Rupture.from_srf(fake_sw4.write_srf(tmp_path / "bend.h5", planes))
    (ring,) = rupture.footprint()
    np.testing.assert_allclose(ring[0], ring[-1])
    spacing = np.hypot(*np.diff(ring, axis=0).T)
    assert spacing.max() <= scene.FOOTPRINT_SPACING + 1.0


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


def test_vendored_lakes_cover_inland_and_coastal_lakes() -> None:
    """The vendored lakes include high inland lakes and coastal lagoons."""
    lakes = scene.load_lakes()
    assert lakes.is_valid
    taupo = shapely.Point(1_855_800, 5_700_800)
    te_waihora = shapely.Point(1_560_400, 5_151_600)
    assert lakes.contains(taupo) and lakes.contains(te_waihora)
    assert not lakes.contains(shapely.Point(1_570_700, 5_180_200))  # Christchurch


def test_vendored_water_and_roads() -> None:
    """Rivers join the lakes, and roads are the state highway network."""
    water = scene.load_water()
    assert water.is_valid
    assert water.contains(shapely.Point(1_514_400, 5_221_070))  # Waimakariri
    assert water.contains(shapely.Point(1_855_800, 5_700_800))  # Taupō
    roads = scene.load_roads()
    assert roads.geom_type == "MultiLineString"
    assert 10_000e3 < roads.length < 12_500e3
    assert roads.distance(shapely.Point(1_570_700, 5_180_200)) < 3e3  # Christchurch


def test_line_coverage_draws_lines_at_width() -> None:
    """A road across the grid covers its own pixels and none far away."""
    road = shapely.LineString([(0, 30), (60, 30)])

    def _to_pixels(easting: np.ndarray, northing: np.ndarray) -> tuple:
        """Use metres as pixels, rows running south."""
        return easting, 59 - northing

    covered = scene.line_coverage(road, (0, 0, 60, 60), _to_pixels, (60, 60), 3.0)
    assert covered[29, 30] > 0.9
    assert covered[20, 30] == 0.0
    assert covered[:, 30].sum() == pytest.approx(3.0, abs=0.5)


def test_coverage_rasterises_holes() -> None:
    """Islands in a lake are left uncovered."""
    lake = shapely.Polygon(
        [(0, 0), (60, 0), (60, 60), (0, 60)], [[(20, 20), (40, 20), (40, 40), (20, 40)]]
    )

    def _to_pixels(easting: np.ndarray, northing: np.ndarray) -> tuple:
        """Use metres as pixels, rows running south."""
        return easting, 59 - northing

    covered = scene.coverage(lake, (0, 0, 60, 60), _to_pixels, (60, 60))
    assert covered[5, 5] > 0.9
    assert covered[30, 30] < 0.1


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
        scene.load_water(),
        scene.load_roads(),
        scene.Rupture.from_srf(sw4_run["srf"]),
        "Mw 6.0 earthquake",
        "Somewhere",
        False,
    )
    assert built.terrain_texture.shape == (*topo.values.shape, 3)
    # The domain covers x 10-30 km, y 10-40 km of the 1 km grid.
    assert built.wave_inside.sum() == 21 * 31
    # Relief is drawn over the domain only; the absorbing layer is flat map.
    i, j = built.mesh_index
    assert i.min() >= 10 and i.max() <= 30 and j.min() >= 10 and j.max() <= 40
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


def test_inset_outline_keeps_only_large_islands_on_the_mainland_map() -> None:
    """Islands off the inset or too small to see are dropped."""
    coastline = shapely.MultiPolygon(
        [
            shapely.box(1_500_000, 5_000_000, 1_600_000, 5_100_000),  # on the map
            shapely.box(1_700_000, 5_200_000, 1_701_000, 5_201_000),  # too small
            shapely.box(2_500_000, 5_000_000, 2_600_000, 5_100_000),  # the Chathams
        ]
    )
    outline = scene.inset_outline(coastline)
    assert len(outline) == 1
    assert outline[0].shape[1] == 2
    assert outline[0][:, 0].min() == pytest.approx(1_500_000)
