"""Colours and the public shaking scale.

Land is tinted by elevation from a lowland green that never turns blue, so
low-lying land never reads as sea; where the sea is comes from the coastline,
not from elevation.

Shaking is ground velocity in cm/s on one colour scale for the whole movie
(`VelocityScale`), in magma, see-through at its low end so weak shaking leaves
the map readable. The scale tops out at the strongest broadband shaking. The
travelling waves are SW4's low-frequency velocity scaled so their strongest
reaches that same peak: SW4's low-frequency motion from a Mw 4 is far weaker
than what is felt, and drawn as it is it would not show. Towns on the final map
are named in GeoNet's public words, from intensity (MMI).
"""

import dataclasses

import numpy as np
import numpy.typing as npt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

LAND_STOPS = [
    (0.0, "#8fa77f"),
    (120.0, "#a2b386"),
    (400.0, "#bab98f"),
    (800.0, "#c6b28d"),
    (1300.0, "#b09a7e"),
    (1800.0, "#9b8b80"),
    (2200.0, "#b3ada8"),
    (2600.0, "#dddad6"),
    (3700.0, "#fbfbfa"),
]
"""Land colour by elevation (m)."""

SEA_NEAR = "#5f8197"
"""Sea colour at the coast."""
SEA_FAR = "#3e5f75"
"""Sea colour `SEA_FADE_DISTANCE` offshore."""
SEA_FADE_DISTANCE = 40e3
"""Distance offshore (m) over which the sea darkens, standing in for depth."""
LAKE = "#6a8ea3"
"""Lake colour."""

SHAKING_CMAP = "magma"
"""Colour map for ground velocity, the whole movie through."""
SHAKING_CMAP_RANGE = (0.12, 1.0)
"""Part of `SHAKING_CMAP` used: its near-black end would read as a hole."""
SHAKING_OPACITY_RISE = 0.4
"""How far up the scale (0 to 1) shaking takes to rise from see-through."""
VELOCITY_DECADES = 2.0
"""Ground velocities the colour scale spans: from the broadband peak down to a
hundredth of it."""

SEA_SHAKING_OPACITY = 0.3
"""Shaking over the sea means little to viewers; it is drawn faintly there."""

SHAKING_WORDS = [
    (3.0, "Weak"),
    (4.0, "Light"),
    (5.0, "Moderate"),
    (6.0, "Strong"),
    (7.0, "Severe"),
    (8.0, "Extreme"),
]
"""GeoNet's public words for shaking, with the intensity each starts at."""


def land_cmap() -> LinearSegmentedColormap:
    """Build the land colour map over `LAND_STOPS`.

    Returns
    -------
    LinearSegmentedColormap
        The colour map, normalised to the highest stop.
    """
    top = LAND_STOPS[-1][0]
    return LinearSegmentedColormap.from_list(
        "land", [(elevation / top, colour) for elevation, colour in LAND_STOPS]
    )


@dataclasses.dataclass(frozen=True)
class VelocityScale:
    """The movie's colour scale: ground velocity on a log scale.

    It tops out at the strongest broadband shaking and spans
    `VELOCITY_DECADES` below it, so the end card fills it, and the waves,
    scaled so their strongest is that peak, fill it too.
    """

    high: float
    """Ground velocity (cm/s) at the top of the scale."""

    @property
    def low(self) -> float:
        """Ground velocity (cm/s) at the bottom of the scale."""
        return self.high / 10**VELOCITY_DECADES

    def level(self, velocity: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Place ground velocity on the scale.

        Parameters
        ----------
        velocity : array-like
            Ground velocity (cm/s); NaN for none.

        Returns
        -------
        np.ndarray
            0 at the bottom of the scale (or below, or NaN), 1 at the top.
        """
        velocity = np.nan_to_num(np.asarray(velocity, dtype=np.float64), nan=0.0)
        ratio = np.maximum(velocity, 1e-30) / self.low
        return np.clip(np.log10(ratio) / VELOCITY_DECADES, 0.0, 1.0).astype(np.float32)


def shaking_cmap() -> LinearSegmentedColormap:
    """Build the shaking colour map: `SHAKING_CMAP` with a rising opacity.

    Returns
    -------
    LinearSegmentedColormap
        The colour map over a `VelocityScale` level, 0 to 1.
    """
    levels = np.linspace(0.0, 1.0, 256)
    rgba = colormaps[SHAKING_CMAP](np.interp(levels, [0, 1], SHAKING_CMAP_RANGE))
    rgba[:, 3] = np.clip(levels / SHAKING_OPACITY_RISE, 0.0, 1.0) ** 0.8
    return LinearSegmentedColormap.from_list("shaking", rgba)


def shaking_lut(size: int = 256) -> npt.NDArray[np.uint8]:
    """Tabulate the shaking colour map as RGBA bytes.

    Parameters
    ----------
    size : int
        Number of entries from the bottom of the scale to the top.

    Returns
    -------
    np.ndarray
        RGBA lookup table, shape `(size, 4)`.
    """
    return (shaking_cmap()(np.linspace(0.0, 1.0, size)) * 255).astype(np.uint8)


def shaking_rgba(
    level: npt.NDArray[np.float32], lut: npt.NDArray[np.uint8] | None = None
) -> npt.NDArray[np.uint8]:
    """Colour levels on the velocity scale.

    Parameters
    ----------
    level : np.ndarray
        Levels from `VelocityScale.level`, 0 to 1.
    lut : np.ndarray | None
        A table from `shaking_lut`, to reuse; built if not given.

    Returns
    -------
    np.ndarray
        RGBA bytes, shape `level.shape + (4,)`.
    """
    lut = shaking_lut() if lut is None else lut
    index = np.clip(np.nan_to_num(level) * (len(lut) - 1), 0, len(lut) - 1)
    return lut[index.astype(np.int32)]


def pgv_to_mmi(pgv: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Convert peak ground velocity to intensity (Worden et al., 2012).

    Parameters
    ----------
    pgv : array-like
        Peak ground velocity (cm/s).

    Returns
    -------
    np.ndarray
        Modified Mercalli intensity, clipped to [1, 10].
    """
    log_pgv = np.log10(np.maximum(np.asarray(pgv, dtype=np.float32), 1e-4))
    mmi = np.where(log_pgv <= 0.53, 3.78 + 1.47 * log_pgv, 2.89 + 3.16 * log_pgv)
    return np.clip(mmi, 1.0, 10.0).astype(np.float32)


def shaking_word(mmi: float) -> str | None:
    """Name the shaking at an intensity in GeoNet's public words.

    Parameters
    ----------
    mmi : float
        Modified Mercalli intensity.

    Returns
    -------
    str | None
        The word, or None for shaking too weak to feel.
    """
    word = None
    for start, name in SHAKING_WORDS:
        if mmi >= start:
            word = name
    return word
