"""Colours and the public shaking scale.

Land is tinted by elevation from a lowland green that never turns blue, so
low-lying land never reads as sea; where the sea is comes from the coastline,
not from elevation. Shaking uses warm colours only (so it cannot be mistaken
for water) and is see-through below "Weak", so weak shaking leaves the map
readable.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

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

SHAKING_STOPS = [
    (2.0, "#fff3a8", 0.0),
    (3.2, "#fff3a8", 0.0),
    (3.8, "#ffe27a", 0.32),
    (4.5, "#ffc04a", 0.55),
    (5.0, "#ffa840", 0.72),
    (5.5, "#ff8a36", 0.82),
    (6.5, "#ec4f33", 0.9),
    (7.5, "#c4213f", 0.94),
    (8.5, "#8a1050", 0.97),
    (10.0, "#4d0a4a", 1.0),
]
"""Shaking colour and opacity by intensity (MMI)."""

MMI_RANGE = (2.0, 10.0)
"""Intensities spanned by the shaking colour map."""

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


def shaking_cmap() -> LinearSegmentedColormap:
    """Build the shaking colour map over `SHAKING_STOPS`.

    Returns
    -------
    LinearSegmentedColormap
        The colour map, with opacity, normalised over `MMI_RANGE`.
    """
    low, high = MMI_RANGE
    return LinearSegmentedColormap.from_list(
        "shaking",
        [
            ((mmi - low) / (high - low), (*to_rgb(colour), alpha))
            for mmi, colour, alpha in SHAKING_STOPS
        ],
    )


def shaking_lut(size: int = 256) -> npt.NDArray[np.uint8]:
    """Tabulate the shaking colour map as RGBA bytes.

    Parameters
    ----------
    size : int
        Number of entries across `MMI_RANGE`.

    Returns
    -------
    np.ndarray
        RGBA lookup table, shape `(size, 4)`.
    """
    return (shaking_cmap()(np.linspace(0.0, 1.0, size)) * 255).astype(np.uint8)


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


def banded(mmi: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Step intensities to the middle of their word's band.

    The final map is drawn in bands so the word printed at each town matches
    the colour under it.

    Parameters
    ----------
    mmi : np.ndarray
        Modified Mercalli intensity.

    Returns
    -------
    np.ndarray
        Intensities of at least `SHAKING_WORDS[0]` moved to their band's centre.
    """
    first = SHAKING_WORDS[0][0]
    return np.where(mmi >= first, np.floor(mmi) + 0.5, mmi).astype(np.float32)
