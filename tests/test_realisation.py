"""Tests for reading realisations."""

from pathlib import Path

import pytest

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
