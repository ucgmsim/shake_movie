"""Tests for reading broadband PGV from the workflow's intensity measures."""

from pathlib import Path

import fake_sw4
import h5py
import numpy as np
import numpy.typing as npt
import pytest
from conftest import SPONGE, write_measures

from shake_movie import intensity_measures
from shake_movie.sw4_images import GridGeometry


def _log_linear(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """PGV rising tenfold every 20 km along SW4's x axis (cm/s)."""
    return 10.0 ** (-1.0 + x / 20e3) + 0.0 * y


def test_stations_in_the_sponge_are_left_out(
    tmp_path: Path, geometry: GridGeometry
) -> None:
    """Stations inside SW4's absorbing layer are not ground-motion predictions."""
    stations = intensity_measures.read_pgv(
        write_measures(tmp_path / "im.h5", geometry, _log_linear)
    )
    x, y = geometry.from_nztm(stations.easting, stations.northing)
    assert stations.pgv.max() < 1e3
    assert x.min() >= SPONGE - 1.0 and y.max() <= 50e3 - SPONGE + 1.0
    # 21 x 26 stations, of which the inner 15 x 20 are outside the sponge.
    assert len(stations.pgv) == 15 * 20


def test_pgv_is_interpolated_in_its_logarithm(
    tmp_path: Path, geometry: GridGeometry
) -> None:
    """Between stations PGV follows its logarithm linearly, and nothing is
    made up beyond them."""
    stations = intensity_measures.read_pgv(
        write_measures(tmp_path / "im.h5", geometry, _log_linear)
    )
    x = np.array([[11.3e3, 20.0e3, 29.1e3], [3e3, 20e3, -40e3]])
    y = np.array([[17.7e3, 25.0e3, 40.2e3], [20e3, 47e3, 20e3]])
    easting, northing = geometry.to_nztm(x, y)
    pgv = intensity_measures.interpolate_pgv(stations, easting, northing)
    assert pgv.shape == x.shape
    np.testing.assert_allclose(pgv[0], _log_linear(x[0], y[0]), rtol=2e-3)
    assert np.isnan(pgv[1]).all()


def test_coverage_ends_away_from_stations(
    tmp_path: Path, geometry: GridGeometry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside the stations' hull, a point far from every station has no value."""
    stations = intensity_measures.read_pgv(
        write_measures(tmp_path / "im.h5", geometry, _log_linear)
    )
    easting, northing = geometry.to_nztm(np.array([21e3]), np.array([25e3]))
    monkeypatch.setattr(intensity_measures, "COVERAGE_DISTANCE", 500.0)
    assert np.isnan(intensity_measures.interpolate_pgv(stations, easting, northing))


def _write_stations(
    path: Path,
    pgv: list[float],
    sponge: list[float] | None,
    dtype: type = np.float64,
) -> Path:
    """Write stations about a kilometre apart near Christchurch, storing PGV
    and position as `dtype`, optionally without `supergrid_depth_gp`."""
    count = len(pgv)
    latitude = np.full(count, -43.5, dtype=dtype)
    longitude = (172.6 + 0.0125 * np.arange(count)).astype(dtype)
    fake_sw4.write_intensity_measures(
        path,
        latitude,
        longitude,
        np.asarray(pgv, dtype=dtype),
        np.zeros(count, dtype=dtype) if sponge is None else np.asarray(sponge, dtype),
    )
    with h5py.File(path, "a") as handle:
        group = handle["PGV"]
        if sponge is None:
            del group["supergrid_depth_gp"]
        # The fake writer stores double precision; store it as asked instead.
        for name in ("rotd50", "latitude", "longitude"):
            values = np.asarray(group[name][:], dtype=dtype)
            del group[name]
            group.create_dataset(name, data=values)
    return path


def test_only_stations_with_positive_pgv_outside_the_sponge_are_kept(
    tmp_path: Path,
) -> None:
    """A station a single grid point into the sponge is inside it, an unknown
    (NaN) depth is not, and zero, negative or non-finite PGV is dropped."""
    stations = intensity_measures.read_pgv(
        _write_stations(
            tmp_path / "im.h5",
            pgv=[1.0, 2.0, 3.0, 0.0, -1.0, np.nan, np.inf, 4.0],
            sponge=[0.0, 1.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.5],
        )
    )
    np.testing.assert_array_equal(stations.pgv, [1.0, 3.0])
    # The kept stations are the 1st and 3rd, 2 x 0.0125 degrees of longitude
    # (about 2 km at 43.5 S) apart along a parallel.
    assert stations.easting[1] - stations.easting[0] == pytest.approx(2.02e3, rel=1e-2)
    assert abs(stations.northing[1] - stations.northing[0]) < 50.0


def test_stations_are_all_kept_without_sponge_depths(tmp_path: Path) -> None:
    """A file without `supergrid_depth_gp` has no stations in the sponge."""
    stations = intensity_measures.read_pgv(
        _write_stations(tmp_path / "im.h5", pgv=[1.0, 2.0, 3.0], sponge=None)
    )
    np.testing.assert_array_equal(stations.pgv, [1.0, 2.0, 3.0])
    assert stations.easting.shape == stations.northing.shape == (3,)


def test_single_precision_measures_are_read_as_double(tmp_path: Path) -> None:
    """Stations hold double precision whatever precision the file stores."""
    stations = intensity_measures.read_pgv(
        _write_stations(
            tmp_path / "im.h5", pgv=[1.5, 2.5], sponge=[0.0, 0.0], dtype=np.float32
        )
    )
    assert stations.pgv.dtype == np.float64
    assert stations.easting.dtype == stations.northing.dtype == np.float64
    np.testing.assert_array_equal(stations.pgv, [1.5, 2.5])


def test_no_station_left_is_an_error(tmp_path: Path) -> None:
    """A file whose every station is in the sponge or has no PGV is refused,
    naming the file."""
    path = _write_stations(
        tmp_path / "im.h5", pgv=[1.0, 0.0, np.nan], sponge=[3.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError, match=r"im\.h5 has no PGV outside SW4's"):
        intensity_measures.read_pgv(path)


def test_coverage_includes_its_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A point exactly `COVERAGE_DISTANCE` from its nearest station still has a
    value, and the result is single precision."""
    stations = intensity_measures.Stations(
        easting=np.array([0.0, 2000.0, 0.0, 2000.0]),
        northing=np.array([0.0, 0.0, 2000.0, 2000.0]),
        pgv=np.array([1.0, 100.0, 1.0, 100.0]),
    )
    monkeypatch.setattr(intensity_measures, "COVERAGE_DISTANCE", 1000.0)
    pgv = intensity_measures.interpolate_pgv(
        stations, np.array([1000.0, 1000.0]), np.array([0.0, 1000.0])
    )
    assert pgv.dtype == np.float32
    # Halfway in easting is halfway in log10(PGV): 10 cm/s. The second point
    # is about 1414 m from every station, so it has no value.
    assert pgv[0] == pytest.approx(10.0, rel=1e-6)
    assert np.isnan(pgv[1])
