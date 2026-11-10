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


def test_velocity_scale_levels_are_float32_computed_in_double() -> None:
    """Levels come back as float32 whatever the input, and half-precision
    velocities are placed as precisely as double-precision ones."""
    scale = palette.VelocityScale(2.0)
    velocity = np.array([0.05, 0.2, 0.7, 1.3], dtype=np.float16)
    level = scale.level(velocity)
    assert level.dtype == np.float32
    exact = np.log10(velocity.astype(np.float64) / 0.02) / 2.0
    np.testing.assert_allclose(level, exact, rtol=0, atol=1e-6)
    assert scale.level([2, 0]).dtype == np.float32


def test_shaking_cmap_is_magma_range_with_opacity_rising_over_the_lower_scale() -> None:
    """Each level takes magma's colour from the top 88% of magma, and an
    opacity of (level / 0.4) ** 0.8, fully opaque from 0.4 up."""
    from matplotlib import colormaps

    cmap = palette.shaking_cmap()
    levels = np.linspace(0.0, 1.0, 256)
    rgba = cmap(levels)
    magma = colormaps["magma"]
    np.testing.assert_allclose(
        rgba[:, :3], magma(0.12 + 0.88 * levels)[:, :3], atol=1e-6
    )
    np.testing.assert_allclose(
        rgba[:, 3], np.clip(levels / 0.4, 0.0, 1.0) ** 0.8, atol=1e-6
    )
    np.testing.assert_allclose(cmap(0.0)[:3], magma(0.12)[:3], atol=1e-6)
    assert cmap(0.0)[3] == 0.0
    # Half way up the opacity rise is 0.5 ** 0.8 opaque, not 0.5 or less.
    assert cmap(0.2)[3] == pytest.approx(0.5**0.8, abs=0.01)
    assert np.all(cmap(np.linspace(0.4, 1.0, 50))[:, 3] == 1.0)


@pytest.mark.parametrize("size", [2, 16, 256])
def test_shaking_lut_tabulates_the_colour_map_in_bytes(size: int) -> None:
    """The table has one RGBA byte row per entry, bottom to top of the scale."""
    lut = palette.shaking_lut(size)
    assert lut.shape == (size, 4)
    assert lut.dtype == np.uint8
    expected = (palette.shaking_cmap()(np.linspace(0.0, 1.0, size)) * 255).astype(
        np.uint8
    )
    np.testing.assert_array_equal(lut, expected)


def test_shaking_lut_defaults_to_256_entries() -> None:
    """The default table has 256 entries."""
    assert palette.shaking_lut().shape == (256, 4)


def test_shaking_rgba_uses_the_given_table_and_clips_levels() -> None:
    """A given table is looked up instead of the default; levels are spread
    over its entries, and levels off the scale take its ends."""
    lut = np.arange(5 * 4, dtype=np.uint8).reshape(5, 4)
    level = np.array([[0.0, 0.25, 0.5], [1.0, -0.5, 1.5]], dtype=np.float32)
    rgba = palette.shaking_rgba(level, lut)
    assert rgba.shape == (2, 3, 4)
    np.testing.assert_array_equal(rgba[0], lut[[0, 1, 2]])
    np.testing.assert_array_equal(rgba[1], lut[[4, 0, 4]])
