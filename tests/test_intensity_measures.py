"""Tests for reading broadband PGV from the workflow's intensity measures."""

from pathlib import Path

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
