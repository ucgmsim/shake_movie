"""Tests for the shaking scale."""

import numpy as np
import pytest

from shake_movie import palette


@pytest.mark.parametrize(
    ("pgv", "mmi"),
    [(1.0, 3.78), (10.0, 2.89 + 3.16), (100.0, 2.89 + 2 * 3.16)],
)
def test_pgv_to_mmi_follows_worden_2012(pgv: float, mmi: float) -> None:
    """Intensities match Worden et al. (2012) on both branches."""
    assert palette.pgv_to_mmi(pgv) == pytest.approx(mmi, abs=1e-4)


def test_pgv_to_mmi_is_clipped() -> None:
    """Intensity stays within the MMI scale."""
    assert palette.pgv_to_mmi(0.0) == 1.0
    assert palette.pgv_to_mmi(1e6) == 10.0


@pytest.mark.parametrize(
    ("mmi", "word"),
    [(2.9, None), (3.0, "Weak"), (4.5, "Light"), (6.2, "Strong"), (9.5, "Extreme")],
)
def test_shaking_words(mmi: float, word: str | None) -> None:
    """Intensities are named in GeoNet's words."""
    assert palette.shaking_word(mmi) == word


def test_banded_centres_each_word() -> None:
    """Felt shaking steps to the middle of its band, so colours match words."""
    banded = palette.banded(np.array([2.5, 3.1, 5.9, 8.7], dtype=np.float32))
    np.testing.assert_allclose(banded, [2.5, 3.5, 5.5, 8.5])


def test_weak_shaking_is_see_through() -> None:
    """Shaking below "Weak" is invisible and the strongest is opaque."""
    lut = palette.shaking_lut()
    low, high = palette.MMI_RANGE
    index = int((3.0 - low) / (high - low) * (len(lut) - 1))
    assert lut[index, 3] == 0
    assert lut[-1, 3] == 255
