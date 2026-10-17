"""The few parts of a workflow realisation a shake movie needs.

Realisations are read as plain JSON (format version 1, see
`ucgmsim/workflow`'s `realisations.py`) so this package does not depend on
the workflow.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from shake_movie.sw4_images import NZTM


def moment_magnitude(magnitudes: list[float]) -> float:
    """Combine per-fault moment magnitudes into one.

    Parameters
    ----------
    magnitudes : list[float]
        Moment magnitude of each fault.

    Returns
    -------
    float
        The magnitude of the summed moment.
    """
    moment = sum(10 ** (1.5 * magnitude) for magnitude in magnitudes)
    return float(np.log10(moment) / 1.5)


@dataclasses.dataclass(frozen=True)
class Realisation:
    """What a shake movie reads from a realisation."""

    name: str
    """The realisation's name."""
    magnitude: float
    """Moment magnitude of the whole rupture."""
    domain: npt.NDArray[np.float64]
    """NZTM corners of the simulated domain (inside SW4's sponge), `(4, 2)`."""
    coastline: Path | None
    """The coastline the velocity model was built with, if it names one."""

    @classmethod
    def read(cls, path: Path) -> "Realisation":
        """Read a realisation file.

        Parameters
        ----------
        path : Path
            The realisation JSON file.

        Returns
        -------
        Realisation
            The parts of the realisation the movie needs.
        """
        config = json.loads(path.read_text())
        to_nztm = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
        corners = config["domain"]["domain"]
        easting, northing = to_nztm.transform(
            [corner["longitude"] for corner in corners],
            [corner["latitude"] for corner in corners],
        )
        coastline = next(
            (
                Path(layer["coastline"])
                for layer in config.get("nzcvm", {}).get("layers", [])
                if layer.get("type") == "coastline" and "coastline" in layer
            ),
            None,
        )
        return cls(
            name=config["metadata"]["name"],
            magnitude=moment_magnitude(
                list(config["magnitudes"]["magnitudes"].values())
            ),
            domain=np.column_stack([easting, northing]),
            coastline=coastline,
        )
