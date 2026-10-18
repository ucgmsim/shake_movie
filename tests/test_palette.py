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


def test_velocity_scale_spans_two_decades_below_the_peak() -> None:
    """The peak is the top, a hundredth of it the bottom, a tenth halfway."""
    scale = palette.VelocityScale(2.0)
    assert scale.low == pytest.approx(0.02)
    level = scale.level([2.0, 0.2, 0.02, 0.0, np.nan, 50.0])
    np.testing.assert_allclose(level, [1.0, 0.5, 0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_shaking_is_magma_and_see_through_at_the_bottom() -> None:
    """The bottom of the scale is invisible, the top opaque, and opacity only
    rises; the colours are magma's."""
    lut = palette.shaking_lut()
    assert lut[0, 3] == 0 and lut[-1, 3] == 255
    assert np.all(np.diff(lut[:, 3].astype(int)) >= 0)
    from matplotlib import colormaps

    top = np.round(np.asarray(colormaps["magma"](1.0)[:3]) * 255)
    np.testing.assert_allclose(lut[-1, :3], top, atol=2)


def test_rgba_ignores_missing_shaking() -> None:
    """NaN (no station coverage) colours as the see-through bottom."""
    rgba = palette.shaking_rgba(np.array([np.nan, 1.0], dtype=np.float32))
    assert rgba[0, 3] == 0 and rgba[1, 3] == 255
