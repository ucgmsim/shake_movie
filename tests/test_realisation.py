"""Tests for reading realisations."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyproj import Transformer

from shake_movie.realisation import Realisation, moment_magnitude


def test_moment_magnitude_sums_moments() -> None:
    """Two equal faults release twice the moment: 0.2 magnitude units more."""
    assert moment_magnitude([7.0]) == pytest.approx(7.0)
    assert moment_magnitude([7.0, 7.0]) == pytest.approx(
        7.0 + 2 / 3 * 0.30103, abs=1e-4
    )


def test_read_realisation(sw4_run: dict[str, Path]) -> None:
    """The name, magnitude, domain and coastline come from the realisation."""
    info = Realisation.read(sw4_run["realisation"])
    assert info.name == "test_run"
    assert info.magnitude == pytest.approx(6.0)
    assert info.domain.shape == (4, 2)
    assert info.coastline == sw4_run["coastline"]


def _write_realisation(path: Path, nzcvm: dict[str, Any] | None) -> Path:
    """Write a minimal realisation with the given (optional) `nzcvm` section."""
    config: dict[str, Any] = {
        "metadata": {"name": "minimal"},
        "magnitudes": {"magnitudes": {"a": 6.0, "b": 6.0}},
        "domain": {
            "domain": [
                {"latitude": -43.0, "longitude": 172.0},
                {"latitude": -43.0, "longitude": 173.0},
                {"latitude": -44.0, "longitude": 173.0},
                {"latitude": -44.0, "longitude": 172.0},
            ]
        },
    }
    if nzcvm is not None:
        config["nzcvm"] = nzcvm
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize(
    "nzcvm",
    [
        None,
        {},
        {"layers": []},
        {"layers": [{"type": "surface", "coastline": "not/this.geojson"}]},
        {"layers": [{"type": "coastline"}]},
    ],
    ids=["no-nzcvm", "no-layers", "empty-layers", "other-type", "no-path"],
)
def test_read_realisation_without_coastline(
    tmp_path: Path, nzcvm: dict[str, Any] | None
) -> None:
    """Without a coastline-typed layer naming a path, the coastline is None."""
    info = Realisation.read(_write_realisation(tmp_path / "r.json", nzcvm))
    assert info.coastline is None


def test_read_realisation_picks_first_coastline_layer(tmp_path: Path) -> None:
    """The first layer typed coastline that names a path is the coastline."""
    nzcvm = {
        "layers": [
            {"type": "surface", "coastline": "wrong.geojson"},
            {"type": "coastline"},
            {"type": "coastline", "coastline": "right.geojson"},
            {"type": "coastline", "coastline": "later.geojson"},
        ]
    }
    info = Realisation.read(_write_realisation(tmp_path / "r.json", nzcvm))
    assert info.coastline == Path("right.geojson")


def test_read_realisation_values(tmp_path: Path) -> None:
    """Name, combined magnitude and NZTM domain corners are read exactly."""
    info = Realisation.read(_write_realisation(tmp_path / "r.json", None))
    assert info.name == "minimal"
    assert info.magnitude == pytest.approx(6.0 + 2 / 3 * np.log10(2))
    to_nztm = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    easting, northing = to_nztm.transform(
        [172.0, 173.0, 173.0, 172.0], [-43.0, -43.0, -44.0, -44.0]
    )
    np.testing.assert_allclose(info.domain, np.column_stack([easting, northing]))
    # Longitude maps to easting and latitude to northing.
    assert info.domain[1, 0] > info.domain[0, 0]
    assert info.domain[2, 1] < info.domain[1, 1]
