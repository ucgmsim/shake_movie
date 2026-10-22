"""Render a shake movie from a finished SW4 run.

Run it as `shake-movie SW4_INPUT REALISATION [--intensity-measures
intensity_measures.h5] [--output MOVIE.mp4]`.
Everything else is read from the SW4 input file: where SW4 wrote its images
(`fileio`), which `imagehdf5` commands hold the terrain (`topo`), the ground
velocity (`mag`, every 0.5 s in the workflow's defaults) and its peak
(`hmax`), and the SRF it ran (`rupturehdf5`). SW4 driven by an SRF slip
*rate* outputs velocity where its modes say displacement, so `mag` is ground
velocity and `hmax` peak horizontal ground velocity.

Shaking is drawn in ground velocity (cm/s) on one colour scale for the whole
movie, topped by the strongest broadband shaking from the workflow's intensity
measures (`--intensity-measures`, `im-calc`'s `intensity_measures.h5`). The
final map is that broadband shaking, and the waves are SW4's low-frequency
velocity scaled so their strongest reaches the same peak. Without the
intensity measures, SW4's low-frequency motion stands in for the broadband,
which underestimates how strongly smaller earthquakes are felt.

Next to the movie, a JSON file records its title, place and date for the
upload.
"""

import dataclasses
import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import psutil
import typer

from shake_movie import camera, intensity_measures, naming, palette, render, scene
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


def _cores() -> int:
    """Count the physical cores this process may run on.

    The affinity mask lists logical CPUs, so on a node with hyperthreading a
    Slurm job asking for 16 cores sees 32. Each worker holds its own scene,
    renderer and encoder, so running one per hyperthread doubles the memory
    for little speed.
    """
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    return max(1, len(psutil.Process().cpu_affinity()) * physical // logical)


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


def end_card(
    shake_scene: scene.Scene,
    geometry: GridGeometry,
    spacing: float,
    strongest: npt.NDArray[np.float32],
    measures: Path | None,
) -> render.EndCard:
    """Work out the final map of the strongest shaking, and the colour scale.

    Parameters
    ----------
    shake_scene : scene.Scene
        The scene.
    geometry : GridGeometry
        Where SW4's grid sits.
    spacing : float
        SW4 image spacing (m).
    strongest : np.ndarray
        SW4's strongest low-frequency ground velocity (m/s) on its image grid,
        for when there are no broadband intensity measures.
    measures : Path | None
        The workflow's `intensity_measures.h5`, if it has one.

    Returns
    -------
    render.EndCard
        Peak ground velocity at the shaking texture pixels on land (NaN at
        sea and beyond the stations' coverage), broadband where given, the
        colour scale it tops out, and the words for each town.
    """
    if measures is None:
        logger.info(
            "No intensity measures given: the final map shows SW4's"
            " low-frequency motion, which underestimates smaller earthquakes"
        )
        stride = shake_scene.wave_stride
        pgv = (strongest[::stride, ::stride] * render.CMS_PER_MS).astype(np.float32)
    else:
        pgv = intensity_measures.interpolate_pgv(
            intensity_measures.read_pgv(measures),
            shake_scene.wave_easting,
            shake_scene.wave_northing,
        ).astype(np.float32)
    # The final map is of shaking on land: clip it to the coastline.
    pgv = np.where(shake_scene.wave_opacity >= 1.0, pgv, np.nan).astype(np.float32)
    mmi = np.where(np.isfinite(pgv), palette.pgv_to_mmi(pgv), 1.0).astype(np.float32)
    covered = np.isfinite(pgv)
    peak = float(pgv[covered].max()) if covered.any() else 1.0
    if not peak > 0:
        peak = 1.0
    logger.info(
        "Strongest %s shaking on land: %.3g cm/s (MMI %.1f)",
        "low-frequency" if measures is None else "broadband",
        peak,
        float(palette.pgv_to_mmi(peak)),
    )
    return render.EndCard(
        pgv,
        palette.VelocityScale(peak),
        _town_words(shake_scene, geometry, spacing, mmi),
        weak=measures is not None
        and palette.shaking_word(float(mmi[covered].max(initial=1.0))) is None,
    )


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
    reference: float
    """The strongest low-frequency ground velocity (m/s), which the waves are
    scaled to the top of the colour scale by."""
    end_card: render.EndCard
    """The final map, and the colour scale."""

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
    measures: Path | None = None,
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
    measures : Path | None
        The workflow's broadband `intensity_measures.h5`, for the final map.

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
    moving = sequence.maximum()
    inside = moving[::stride, ::stride][shake_scene.wave_inside]
    reference = float(inside.max()) if inside.size else float(moving.max())
    if not reference > 0:
        reference = 1.0
    logger.info("Strongest low-frequency ground velocity: %.3g cm/s", reference * 100)

    # Waves out at sea are left to run: the camera follows what people feel.
    on_land = shake_scene.wave_inside & (shake_scene.wave_opacity >= 1.0)

    def _felt(time: float) -> npt.NDArray[np.float64]:
        """Land points moving at `camera.FELT_FRACTION` of the strongest or more."""
        now = sequence.at(time)[::stride, ::stride]
        mask = (now >= camera.FELT_FRACTION * reference) & on_land
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
    strongest = read_image(peak[-1]).values if peak else moving
    card = end_card(shake_scene, geometry, terrain.grid_size, strongest, measures)
    return MoviePlan(info, name, shake_scene, frames, velocity, reference, card)


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
    measures: Annotated[
        Path | None,
        typer.Option(
            "--intensity-measures",
            exists=True,
            dir_okay=False,
            help="The workflow's broadband intensity_measures.h5, for the final map. Without it, the map shows SW4's low-frequency motion.",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            help="Worker processes. Defaults to the physical cores available."
        ),
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
    measures : Path | None
        The workflow's broadband `intensity_measures.h5`, for the final map.
    workers : int | None
        Worker processes; defaults to the physical cores available.
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
        sw4_input,
        realisation,
        settings,
        images,
        srf,
        coastline,
        online=not offline,
        measures=measures,
    )
    output = output or Path(f"{plan.realisation.name}.mp4")
    count = workers or _cores()
    logger.info("Rendering %d frames on %d workers", len(plan.frames), count)
    render.render_movie(
        plan.scene,
        plan.frames,
        plan.velocity,
        plan.reference,
        plan.end_card,
        settings,
        output,
        count,
    )
    output.with_suffix(".json").write_text(json.dumps(plan.metadata(), indent=2))
    logger.info("Wrote %s", output)
