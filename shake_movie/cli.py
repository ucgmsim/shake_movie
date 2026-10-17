"""Render a shake movie from a finished SW4 run.

Run it as `shake-movie SW4_INPUT REALISATION [--output MOVIE.mp4]`.
Everything else is read from the SW4 input file: where SW4 wrote its images
(`fileio`), which `imagehdf5` commands hold the terrain (`topo`), the ground
velocity (`mag`, every 0.5 s in the workflow's defaults) and its peak
(`hmax`), and the SRF it ran (`rupturehdf5`). SW4 driven by an SRF slip
*rate* outputs velocity where its modes say displacement, so `mag` is ground
velocity and `hmax` peak horizontal ground velocity.

Next to the movie, a JSON file records its title, place and date for the
upload.
"""

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer

from shake_movie import camera, naming, palette, render, scene
from shake_movie.realisation import Realisation
from shake_movie.sw4_images import (
    GridGeometry,
    ImageSequence,
    find_images,
    parse_sw4_input,
    read_image,
    read_image_shape,
)

app = typer.Typer(add_completion=False)
logger = logging.getLogger("shake_movie")

FELT_POINTS = 4000
"""Most shaking points the camera considers per sample."""


def _image_prefix(commands: dict[str, list[dict[str, str]]], mode: str) -> str | None:
    """Find the file prefix of the surface `imagehdf5` command for a mode."""
    for command in commands.get("imagehdf5", []):
        if command.get("mode") == mode and float(command.get("z", "nan")) == 0.0:
            return command.get("file")
    return None


def _images(
    commands: dict[str, list[dict[str, str]]], directory: Path, mode: str
) -> list[Path]:
    """Find the surface images SW4 wrote for a mode."""
    prefix = _image_prefix(commands, mode)
    return [] if prefix is None else find_images(directory, prefix, mode)


def _matching(paths: list[Path], shape: tuple[int, int]) -> list[Path]:
    """Keep the images on a grid of the given shape.

    A retried SW4 run on a resized domain leaves the earlier attempt's images
    behind, at cycles the retry doesn't overwrite when its time step differs.
    """
    kept = [path for path in paths if read_image_shape(path) == shape]
    if len(kept) < len(paths):
        logger.warning(
            "Skipping %d images that don't match the %d x %d terrain,"
            " left by another SW4 run",
            len(paths) - len(kept),
            *shape,
        )
    return kept


def _coastline(given: Path | None, realisation: Realisation) -> Path:
    """Choose the coastline: given, the velocity model's, or LINZ's."""
    if given is not None:
        return given
    if realisation.coastline is not None and realisation.coastline.exists():
        return realisation.coastline
    logger.info("Downloading LINZ's coastline (cached for later movies)")
    return scene.download_coastline()


def _town_words(
    shake_scene: scene.Scene,
    geometry: GridGeometry,
    spacing: float,
    peak_mmi: npt.NDArray[np.float32],
) -> dict[str, str | None]:
    """Name the strongest shaking at each town inside the domain."""
    stride = shake_scene.wave_stride
    words: dict[str, str | None] = {}
    for town in shake_scene.towns[: render.LABEL_CANDIDATES]:
        x, y = geometry.from_nztm(town.easting, town.northing)
        i, j = (
            round(float(x) / (spacing * stride)),
            round(float(y) / (spacing * stride)),
        )
        if (
            0 <= i < peak_mmi.shape[0]
            and 0 <= j < peak_mmi.shape[1]
            and shake_scene.wave_inside[i, j]
        ):
            words[town.name] = palette.shaking_word(float(peak_mmi[i, j]))
    return words


@dataclasses.dataclass(frozen=True)
class MoviePlan:
    """Everything worked out before rendering."""

    realisation: Realisation
    """The realisation."""
    naming: naming.Naming
    """The title and place."""
    scene: scene.Scene
    """The scene."""
    frames: list[camera.Frame]
    """Every frame of the movie."""
    velocity: list[Path]
    """`surf_mag` images (ground velocity)."""
    peak_mmi: npt.NDArray[np.float32]
    """Strongest shaking intensity at shaking texture pixels."""
    words: dict[str, str | None]
    """Strongest shaking at each town, in words."""

    def metadata(self) -> dict[str, object]:
        """Describe the movie for its upload.

        Returns
        -------
        dict[str, object]
            The title, place, time, magnitude and a description.
        """
        name = self.naming
        return {
            "title": name.title,
            "location": name.location,
            "time": None if name.time is None else name.time.isoformat(),
            "magnitude": round(self.realisation.magnitude, 1),
            "realisation": self.realisation.name,
            "description": (
                f"{name.title}, {name.location}. A computer simulation of the ground"
                " shaking, from the SW4 wave propagation code. Real shaking may differ."
            ),
        }


def plan_movie(
    sw4_input: Path,
    realisation: Path,
    settings: render.Settings,
    images: Path | None = None,
    srf: Path | None = None,
    coastline: Path | None = None,
    online: bool = True,
) -> MoviePlan:
    """Read an SW4 run and work out the movie: scene, names and camera.

    Parameters
    ----------
    sw4_input : Path
        The SW4 input file of the run.
    realisation : Path
        The run's realisation file.
    settings : render.Settings
        How the movie will be rendered.
    images : Path | None
        Where SW4 wrote its images, if not its `fileio` path.
    srf : Path | None
        The SRF SW4 ran, if not its `rupturehdf5` file.
    coastline : Path | None
        Land polygons; defaults to the velocity model's, else LINZ's.
    online : bool
        Whether to ask GeoNet and Nominatim for the date and place.

    Returns
    -------
    MoviePlan
        The planned movie.

    Raises
    ------
    typer.BadParameter
        If the SW4 run lacks the images a movie needs.
    """
    commands = parse_sw4_input(sw4_input)
    geometry = GridGeometry.from_grid_command(commands["grid"][0])
    directory = images or Path(commands["fileio"][0]["path"])
    topo = _images(commands, directory, "topo")
    shape = read_image_shape(topo[0]) if topo else (0, 0)
    velocity = _matching(_images(commands, directory, "mag"), shape)
    peak = _matching(_images(commands, directory, "hmax"), shape)
    if not topo or len(velocity) < 2:
        raise typer.BadParameter(
            f"{directory} needs a surface `topo` image and a series of surface `mag` images."
        )
    info = Realisation.read(realisation)

    logger.info("Preparing the scene")
    rupture = scene.Rupture.from_srf(srf or Path(commands["rupturehdf5"][0]["file"]))
    name = naming.name_earthquake(
        info.name, info.magnitude, rupture.hypocentre, scene.load_towns(), online=online
    )
    logger.info("%s, %s", name.title, name.location)
    terrain = read_image(topo[0])
    shake_scene = scene.build_scene(
        geometry,
        terrain.values,
        terrain.grid_size,
        info.domain,
        scene.load_coastline(_coastline(coastline, info)),
        rupture,
        name.title,
        name.location,
        name.is_real,
    )

    logger.info("Planning the camera")
    sequence = ImageSequence(velocity)
    stride = shake_scene.wave_stride

    def _felt(time: float) -> npt.NDArray[np.float64]:
        """NZTM points shaking at `camera.FELT_MMI` or above at a time."""
        mmi = render.intensity(sequence, time, stride)
        mask = (mmi >= camera.FELT_MMI) & shake_scene.wave_inside
        points = np.column_stack(
            [shake_scene.wave_easting[mask], shake_scene.wave_northing[mask]]
        )
        return points[:: max(1, len(points) // FELT_POINTS)]

    path = camera.plan_camera(
        rupture,
        info.domain,
        shake_scene.towns,
        _felt,
        sequence.end_time,
        settings.aspect,
    )
    frames = camera.timeline(
        path, info.domain, sequence.end_time, settings.fps, settings.aspect
    )
    strongest = read_image(peak[-1]).values if peak else sequence.maximum()
    peak_mmi = palette.pgv_to_mmi(strongest[::stride, ::stride] * render.CMS_PER_MS)
    words = _town_words(shake_scene, geometry, terrain.grid_size, peak_mmi)
    return MoviePlan(info, name, shake_scene, frames, velocity, peak_mmi, words)


@app.command()
def main(
    sw4_input: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="The SW4 input file of the run."
        ),
    ],
    realisation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="The run's realisation file."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="The movie to write. Defaults to <realisation name>.mp4."),
    ] = None,
    images: Annotated[
        Path | None,
        typer.Option(help="Where SW4 wrote its images, if not its fileio path."),
    ] = None,
    srf: Annotated[
        Path | None, typer.Option(help="The SRF SW4 ran, if not its rupturehdf5 file.")
    ] = None,
    coastline: Annotated[
        Path | None,
        typer.Option(
            help="Land polygons (nzcvm .wkb.gz or GMT .gmt). Defaults to the velocity model's, else LINZ's."
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(help="Worker processes. Defaults to the CPUs available."),
    ] = None,
    width: Annotated[int, typer.Option(help="Frame width.")] = 1920,
    height: Annotated[int, typer.Option(help="Frame height.")] = 1080,
    fps: Annotated[int, typer.Option(help="Frames per second.")] = 30,
    offline: Annotated[
        bool,
        typer.Option(help="Do not ask GeoNet or Nominatim for the date and place."),
    ] = False,
) -> None:
    """Render a public-facing video of an SW4 simulation.

    Parameters
    ----------
    sw4_input : Path
        The SW4 input file of the run.
    realisation : Path
        The run's realisation file.
    output : Path | None
        The movie to write; defaults to `<realisation name>.mp4`.
    images : Path | None
        Where SW4 wrote its images, if not its `fileio` path.
    srf : Path | None
        The SRF SW4 ran, if not its `rupturehdf5` file.
    coastline : Path | None
        Land polygons; defaults to the velocity model's, else LINZ's.
    workers : int | None
        Worker processes; defaults to the CPUs available.
    width : int
        Frame width.
    height : int
        Frame height.
    fps : int
        Frames per second.
    offline : bool
        Skip the date and place lookups.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = render.Settings(width=width, height=height, fps=fps)
    plan = plan_movie(
        sw4_input, realisation, settings, images, srf, coastline, online=not offline
    )
    output = output or Path(f"{plan.realisation.name}.mp4")
    count = workers or len(os.sched_getaffinity(0))
    logger.info("Rendering %d frames on %d workers", len(plan.frames), count)
    render.render_movie(
        plan.scene,
        plan.frames,
        plan.velocity,
        plan.peak_mmi,
        plan.words,
        settings,
        output,
        count,
    )
    output.with_suffix(".json").write_text(json.dumps(plan.metadata(), indent=2))
    logger.info("Wrote %s", output)
