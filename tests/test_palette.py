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


@pytest.mark.parametrize(
    ("pgv", "mmi"),
    [
        (0.1, 3.78 - 1.47),
        (0.05, 3.78 + 1.47 * np.log10(0.05)),
        (2.0, 3.78 + 1.47 * np.log10(2.0)),
    ],
)
def test_pgv_to_mmi_low_branch_slope(pgv: float, mmi: float) -> None:
    """Below the break, intensity rises 1.47 per decade of PGV."""
    assert palette.pgv_to_mmi(pgv) == pytest.approx(mmi, abs=1e-4)


def test_pgv_to_mmi_break_belongs_to_the_lower_branch() -> None:
    """At log10(PGV) = 0.53 exactly the lower relation applies (3.78 +
    1.47 * 0.53), not the upper (2.89 + 3.16 * 0.53), which is 0.0057 higher.
    """
    # The single-precision PGV whose single-precision log10 is exactly 0.53.
    pgv = np.float32(3.388441324234009)
    assert np.log10(pgv) == np.float32(0.53)
    assert palette.pgv_to_mmi(pgv) == pytest.approx(3.78 + 1.47 * 0.53, abs=1e-4)
    # Just above the break the upper relation takes over. One float32 step in
    # PGV moves log10 by less than half a float32 step at 0.53, so step until
    # the single-precision log10 actually clears the break.
    above = pgv
    while np.log10(above) <= np.float32(0.53):
        above = np.nextafter(above, np.float32(10.0))
    assert palette.pgv_to_mmi(above) == pytest.approx(2.89 + 3.16 * 0.53, abs=1e-4)


def test_pgv_to_mmi_returns_float32_of_the_input_shape() -> None:
    """Intensities come back as single precision whatever the input."""
    pgv = np.array([[0.5, 5.0], [50.0, 0.0]], dtype=np.float64)
    mmi = palette.pgv_to_mmi(pgv)
    assert mmi.dtype == np.float32
    assert mmi.shape == (2, 2)
    assert palette.pgv_to_mmi([1, 10]).dtype == np.float32


def test_colour_maps_are_named() -> None:
    """The colour maps carry their names, as matplotlib reports them."""
    assert palette.land_cmap().name == "land"
    assert palette.shaking_cmap().name == "shaking"


def test_land_cmap_runs_through_the_elevation_stops() -> None:
    """Each stop's colour sits at its elevation over the highest stop."""
    from matplotlib.colors import to_rgba

    cmap = palette.land_cmap()
    top = palette.LAND_STOPS[-1][0]
    for elevation, colour in palette.LAND_STOPS:
        np.testing.assert_allclose(cmap(elevation / top), to_rgba(colour), atol=1 / 255)
