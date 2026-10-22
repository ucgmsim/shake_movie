"""Render a shake movie on CPUs.

Each worker process owns one off-screen VTK window. The terrain texture is
uploaded once; each frame only swaps a small shaking texture, draped on a
second copy of the terrain, and moves the camera. Labels, the fault and the
on-screen text are drawn over the 3D frame with matplotlib, and frames are
piped straight into ffmpeg, one segment per worker, then joined without
re-encoding.

Shaking is coloured on one ground velocity scale for the whole movie
(`palette.VelocityScale`), topped by the strongest broadband shaking. The waves
are SW4's low-frequency velocity scaled so their strongest reaches that peak;
the final map is the broadband shaking itself (`EndCard`), with towns named in
GeoNet's words.

Without a GPU, VTK renders through Mesa (EGL or OSMesa, llvmpipe). Mesh size
dominates that cost, so a coarser terrain mesh takes over once the camera is
far out.
"""

import dataclasses
import itertools
import os
import shutil
import subprocess
import tempfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text

from shake_movie import palette
from shake_movie.camera import VERTICAL_FOV, Frame, Shot
from shake_movie.scene import Scene, Town
from shake_movie.sw4_images import ImageSequence

FAR_LOD_DISTANCE = 180e3
"""Camera distance (m) beyond which the coarse terrain mesh is drawn."""
OVERLAY_LIFT = 40.0
"""Height (m, before exaggeration) the shaking sits above the terrain."""
NEAR_FRACTION = 0.5
"""Near clipping plane as a fraction of the camera's height above the highest
ground: a fixed near plane leaves far-out shots too little depth precision to
tell the terrain, the shaking and the map beneath apart."""
FAR_CLIP = 3e6
"""Far clipping plane (m)."""
OVERLAY_OFFSET = (-1.0, -1.0)
"""Polygon offset (slope factor, depth units) pulling the shaking toward the
camera: lifting it alone leaves steep terrain poking through it far out."""
PLANE_FILL, FINAL_PLANE_FILL = 0.12, 0.4
"""Opacity of the fault planes' fill while the waves play, faint so they show
through, and on the final map, where the planes are narrow."""
FINAL_TRACE = 0.5
"""How much thinner the fault's trace is drawn on the final map, so the
planes dipping away from it show."""
LABEL_FADE = 0.4
"""Time (s) a town label takes to fade in or out."""
MAX_LABELS, MAX_FINAL_LABELS = 8, 14
"""Town labels shown during the simulation and on the final map."""
LABEL_CANDIDATES = 80
"""Largest towns considered for labels each frame."""
FINAL_OPACITY = 0.8
"""Opacity of the strongest shaking map, so the terrain shows through."""
BACKGROUND = "#3f4a50"
"""Colour beyond the map's edge."""
CAPTION = "Computer simulation. Real shaking may differ."
"""Shown on every frame."""
CREDITS = "Simulated with SW4  ·  Coastline: LINZ (CC BY)  ·  Places © OpenStreetMap contributors"
"""Shown on the final map."""
CMS_PER_MS = 100.0
"""SW4 velocities are m/s; intensities are computed from cm/s."""
WEAK_NOTE = "Strongest shaking: too weak for most people to notice"
"""Shown on a final map where no shaking on land reaches a word, in place of
"Strongest shaking felt"."""
EPICENTRE_COLOUR = "#ffcc1a"
"""Fill of the star marking the epicentre."""
VELOCITY_TITLE = "Ground velocity (cm/s)"
"""Heading of the legend."""
DIGIT_WIDTH = 0.6
"""Width of a digit as a fraction of its font size, for spacing tick labels."""
TICK_GAP = 8.0
"""Gap (points) two neighbouring tick labels need to read as two."""
SKIRT = "#2c3236"
"""Colour of the wall dropping from the domain's edge to the map beneath."""

FONT = "DejaVu Sans"
HALO = [patheffects.withStroke(linewidth=3.2, foreground="#1a1a1a")]
HALO_LARGE = [patheffects.withStroke(linewidth=4, foreground="#1a1a1a")]
TITLE_CARD_WIDTH = 0.9
"""Most of the frame's width a title card line may take."""
HUD_TEXT_WIDTH = 0.68
"""Most of the frame's width the title and place at the top left may take,
clear of the caption at the top right."""


@dataclasses.dataclass(frozen=True)
class Settings:
    """How to render."""

    width: int = 1920
    """Frame width (pixels)."""
    height: int = 1080
    """Frame height (pixels)."""
    fps: int = 30
    """Frames per second."""
    exaggeration: float = 2.0
    """Vertical exaggeration of the terrain."""
    crf: int = 20
    """x264 quality (lower is better)."""

    @property
    def aspect(self) -> float:
        """The frame's aspect ratio.

        Returns
        -------
        float
            Frame width over height.
        """
        return self.width / self.height


# ----------------------------------------------------------------- geometry


def camera_pose(
    scene: Scene, shot: Shot, exaggeration: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Place the camera for a shot.

    Parameters
    ----------
    scene : Scene
        The scene, for the ground height under the looked-at point.
    shot : Shot
        The shot.
    exaggeration : float
        Vertical exaggeration.

    Returns
    -------
    position : np.ndarray
        Camera position.
    focal_point : np.ndarray
        The point looked at.
    up : np.ndarray
        The camera's up vector, which never lines up with its view.
    """
    h, p = np.radians(shot.heading), np.radians(shot.pitch)
    ground = float(scene.height_at(shot.easting, shot.northing)) * exaggeration
    focal_point = np.array([shot.easting, shot.northing, 0.5 * ground])
    forward = np.array([np.sin(h) * np.cos(p), np.cos(h) * np.cos(p), -np.sin(p)])
    up = np.array([np.sin(h) * np.sin(p), np.cos(h) * np.sin(p), np.cos(p)])
    return focal_point - shot.distance * forward, focal_point, up


class Projector:
    """Pinhole projection matching VTK's camera, for placing overlays.

    Parameters
    ----------
    position : np.ndarray
        Camera position.
    focal_point : np.ndarray
        The point looked at.
    up : np.ndarray
        Camera up vector.
    settings : Settings
        Frame size.
    """

    def __init__(
        self,
        position: npt.NDArray[np.float64],
        focal_point: npt.NDArray[np.float64],
        up: npt.NDArray[np.float64],
        settings: Settings,
    ) -> None:
        """Set up the projection for one camera."""
        self.position = position
        forward = focal_point - position
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, up)
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.forward)
        self.tan = np.tan(np.radians(VERTICAL_FOV) / 2)
        self.settings = settings

    def __call__(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project world points to pixels.

        Parameters
        ----------
        points : np.ndarray
            World points, `(n, 3)`.

        Returns
        -------
        np.ndarray
            Pixel coordinates `(n, 2)`, from the top left; NaN behind the camera.
        """
        offset = np.atleast_2d(points) - self.position
        depth = offset @ self.forward
        depth = np.where(depth > 1.0, depth, np.nan)
        x = (offset @ self.right) / (depth * self.tan * self.settings.aspect)
        y = (offset @ self.up) / (depth * self.tan)
        return np.column_stack(
            [(x + 1) / 2 * self.settings.width, (1 - y) / 2 * self.settings.height]
        )


def visible(
    scene: Scene,
    camera: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    exaggeration: float,
    samples: int = 40,
) -> npt.NDArray[np.bool_]:
    """Test which ground points the terrain leaves in view.

    Parameters
    ----------
    scene : Scene
        The scene.
    camera : np.ndarray
        Camera position.
    points : np.ndarray
        Ground points `(n, 2)` in NZTM.
    exaggeration : float
        Vertical exaggeration.
    samples : int
        Steps along each sight line.

    Returns
    -------
    np.ndarray
        Whether each point can be seen.
    """
    points = np.atleast_2d(points)
    target_z = scene.height_at(points[:, 0], points[:, 1]) * exaggeration + 50.0
    s = np.linspace(0.02, 0.97, samples)[None, :]
    e = camera[0] + (points[:, :1] - camera[0]) * s
    n = camera[1] + (points[:, 1:2] - camera[1]) * s
    z = camera[2] + (target_z[:, None] - camera[2]) * s
    return np.all(scene.height_at(e, n) * exaggeration < z, axis=1)


# ------------------------------------------------------------------- labels


@dataclasses.dataclass
class Label:
    """A town label drawn on a frame."""

    town: Town
    """The town."""
    position: tuple[float, float]
    """Pixel position of the town's dot."""
    alpha: float
    """Opacity, as it fades in or out."""
    word: str | None
    """The shaking felt there, on the final map."""


def _tier(population: int) -> int:
    """Rank a town for its label size."""
    return 2 if population >= 100_000 else 1 if population >= 10_000 else 0


class LabelPlacer:
    """Place town labels, keeping them steady from frame to frame.

    Towns already showing keep their place ahead of larger newcomers, and
    labels fade in and out rather than popping.

    Parameters
    ----------
    scene : Scene
        The scene.
    settings : Settings
        Frame size and rate.
    """

    def __init__(self, scene: Scene, settings: Settings) -> None:
        """Start with no labels showing."""
        from matplotlib.path import Path as MplPath

        self.scene = scene
        self.settings = settings
        self.alpha: dict[str, float] = {}
        domain = MplPath(scene.domain)
        self.candidates = scene.towns[:LABEL_CANDIDATES]
        self.in_domain = {
            town.name: bool(domain.contains_point((town.easting, town.northing)))
            for town in self.candidates
        }

    def place(
        self,
        projector: Projector,
        exaggeration: float,
        limit: int,
        words: dict[str, str | None] | None = None,
    ) -> list[Label]:
        """Place labels for one frame.

        Parameters
        ----------
        projector : Projector
            The frame's projection.
        exaggeration : float
            Vertical exaggeration.
        limit : int
            Most labels to show.
        words : dict[str, str | None] | None
            Shaking felt at each town, to print under its name; only towns
            inside the domain are labelled when given.

        Returns
        -------
        list[Label]
            Labels to draw, with their fade.
        """
        settings = self.settings
        scale = settings.width / 1920
        towns = [
            town
            for town in self.candidates
            if words is None or self.in_domain[town.name]
        ]
        if not towns:
            return []
        ground = np.array([[town.easting, town.northing] for town in towns])
        z = self.scene.height_at(ground[:, 0], ground[:, 1]) * exaggeration + 60.0
        screen = projector(np.column_stack([ground, z]))
        seen = visible(self.scene, projector.position, ground, exaggeration)
        order = sorted(
            range(len(towns)),
            key=lambda k: (
                self.alpha.get(towns[k].name, 0.0) == 0.0,
                -towns[k].population,
            ),
        )
        boxes: list[tuple[float, float, float, float]] = []
        shown: dict[str, tuple[float, float]] = {}
        for k in order:
            town = towns[k]
            sx, sy = screen[k]
            if len(shown) >= limit or not seen[k] or not np.isfinite(sx):
                continue
            if not (
                60 * scale < sx < settings.width - 260 * scale
                and 200 * scale < sy < settings.height - 150 * scale
            ):
                continue
            size = (13 + 3 * _tier(town.population)) * scale
            word = (words or {}).get(town.name)
            width = (
                0.62 * size * max(len(town.name), len(word or "")) * 1.35 + 16 * scale
            )
            height = size * (2.6 if word else 1.4) * 1.35
            box = (sx - 8 * scale, sy - height / 2, sx + width, sy + height / 2)
            if any(
                not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                for b in boxes
            ):
                continue
            boxes.append(box)
            shown[town.name] = (float(sx), float(sy))
        step = 1.0 / max(LABEL_FADE * settings.fps, 1.0)
        labels = []
        for k, town in enumerate(towns):
            alpha = self.alpha.get(town.name, 0.0)
            alpha = (
                min(1.0, alpha + step) if town.name in shown else max(0.0, alpha - step)
            )
            self.alpha[town.name] = alpha
            if alpha <= 0.0 or not np.isfinite(screen[k][0]):
                continue
            position = shown.get(town.name, (float(screen[k][0]), float(screen[k][1])))
            labels.append(Label(town, position, alpha, (words or {}).get(town.name)))
        return labels


# ----------------------------------------------------------------------- HUD


def _fit(text: Text, width: float) -> None:
    """Shrink a text's font until the text is at most `width` pixels wide.

    Parameters
    ----------
    text : Text
        The drawn text.
    width : float
        The widest it may be (pixels).
    """
    extent = text.get_window_extent()
    if extent.width > width:
        text.set_fontsize(float(text.get_fontsize()) * width / extent.width)


def draw_hud(
    image: npt.NDArray[np.uint8],
    scene: Scene,
    frame: Frame,
    labels: list[Label],
    fault: list[tuple[npt.NDArray[np.float64], str, float]],
    epicentre: npt.NDArray[np.float64],
    settings: Settings,
    weak: bool = False,
    *,
    scale: palette.VelocityScale,
    planes: list[npt.NDArray[np.float64]] | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw labels, the fault and on-screen text over a rendered frame.

    Parameters
    ----------
    image : np.ndarray
        The 3D frame, `(height, width, 3)`.
    scene : Scene
        The scene.
    frame : Frame
        The frame being drawn.
    labels : list[Label]
        Town labels.
    fault : list[tuple[np.ndarray, str, float]]
        Screen-space fault polylines with colour and line width.
    epicentre : np.ndarray
        Screen position of the epicentre.
    settings : Settings
        Frame size.
    weak : bool
        Whether the final map shows no shaking on land, and says so.
    scale : palette.VelocityScale
        The colour scale, for the legend.
    planes : list[np.ndarray] | None
        Screen-space outlines of the fault planes.

    Returns
    -------
    np.ndarray
        The finished frame.
    """
    w, h = settings.width, settings.height
    s = w / 1920
    figure = Figure(figsize=(w / 100, h / 100), dpi=100)
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.axis("off")
    # Soft shading behind the text, top left and along the bottom.
    gx = np.linspace(1, 0, 256)[None, :] ** 1.5
    gy = np.linspace(1, 0, 128)[:, None] ** 1.5
    ax.imshow(
        np.dstack([np.zeros((128, 256, 3)), 0.6 * gx * gy]),
        extent=(0, w * 0.6, h * 0.26, 0),
        aspect="auto",
    )
    ax.imshow(
        np.dstack([np.zeros((256, 1, 3)), np.linspace(0, 0.5, 256)[:, None, None]]),
        extent=(0, w, h, h * 0.8),
        aspect="auto",
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    # Text goes on its own axes above the map, so the title, legend and
    # captions always cover the fault, epicentre and town labels.
    overlay = figure.add_axes((0, 0, 1, 1))
    overlay.axis("off")
    overlay.set_xlim(0, w)
    overlay.set_ylim(h, 0)
    # The planes lie beneath the trace while it shows the rupture spreading,
    # and over its thinned line on the final map, where they are narrow.
    fill = PLANE_FILL + (FINAL_PLANE_FILL - PLANE_FILL) * frame.peak
    edge = 1.0 + 0.5 * frame.peak
    zorder = 4.5 if frame.peak > 0 else 3.5
    for outline in planes or []:
        ax.add_patch(
            Polygon(
                outline, closed=True, fc=(1.0, 1.0, 1.0, fill), ec="none", zorder=zorder
            )
        )
        # The outline is a line, not the patch's edge: a stroke effect on a
        # patch paints its face again.
        ring = np.vstack([outline, outline[:1]])
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color=(1.0, 1.0, 1.0, 0.8),
            lw=edge * s,
            solid_joinstyle="round",
            zorder=zorder,
            path_effects=[
                patheffects.withStroke(
                    linewidth=(edge + 2.5) * s, foreground="#1a1a1a", alpha=0.6
                ),
                patheffects.Normal(),
            ],
        )
    # Every halo goes beneath every line, so a fault of several planes reads
    # as one unbroken trace.
    thin = 1 - FINAL_TRACE * frame.peak
    fault = [(line, colour, width * thin) for line, colour, width in fault]
    for line, _, width in fault:
        ax.plot(
            line[:, 0],
            line[:, 1],
            color="#1a1a1a",
            lw=(width + 3) * s,
            solid_capstyle="round",
            zorder=3.9,
        )
    for line, colour, width in fault:
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=colour,
            lw=width * s,
            solid_capstyle="round",
            zorder=4,
        )
    if np.all(np.isfinite(epicentre)):
        ax.plot(
            *epicentre,
            marker="*",
            ms=26 * s,
            mfc=EPICENTRE_COLOUR,
            mec="#1a1a1a",
            mew=1.6 * s,
            zorder=5,
        )
    for label in labels:
        tier = _tier(label.town.population)
        x, y = label.position
        ax.plot(
            x,
            y,
            "o",
            ms=(7 + 3 * tier) * s,
            mfc="white",
            mec="#1a1a1a",
            mew=1.4 * s,
            alpha=label.alpha,
            zorder=6,
        )
        text = (
            label.town.name
            if label.word is None
            else f"{label.town.name}\n{label.word}"
        )
        ax.text(
            x + (10 + 2 * tier) * s,
            y - 4 * s,
            text,
            color="white",
            fontsize=(13 + 3 * tier) * s,
            fontweight="bold" if tier == 2 else "normal",
            va="center",
            family=FONT,
            path_effects=HALO_LARGE,
            alpha=label.alpha,
            zorder=7,
            linespacing=1.15,
        )
    title = scene.title
    if frame.title > 0:
        a = frame.title
        line = overlay.text(
            w / 2,
            h * 0.45,
            title,
            color="white",
            fontsize=64 * s,
            fontweight="bold",
            ha="center",
            va="center",
            family=FONT,
            path_effects=HALO_LARGE,
            alpha=a,
        )
        _fit(line, TITLE_CARD_WIDTH * w)
        line = overlay.text(
            w / 2,
            h * 0.55,
            scene.location,
            color="#f1ede4",
            fontsize=30 * s,
            ha="center",
            va="center",
            family=FONT,
            path_effects=HALO_LARGE,
            alpha=a,
        )
        _fit(line, TITLE_CARD_WIDTH * w)
    elif frame.kind != "intro":
        heading = overlay.text(
            48 * s,
            62 * s,
            title,
            color="white",
            fontsize=34 * s,
            fontweight="bold",
            va="center",
            family=FONT,
            path_effects=HALO,
        )
        _fit(heading, HUD_TEXT_WIDTH * w - 48 * s)
        place = overlay.text(
            50 * s,
            108 * s,
            scene.location,
            color="#e9e6df",
            fontsize=19 * s,
            va="center",
            family=FONT,
            path_effects=HALO,
        )
        _fit(place, HUD_TEXT_WIDTH * w - 50 * s)
        if frame.peak < 1.0:
            status = f"{frame.time:.0f} seconds after it began"
        else:
            status = WEAK_NOTE if weak else "Strongest shaking felt"
        overlay.text(
            50 * s,
            150 * s,
            status,
            color="#ffe9b0",
            fontsize=19 * s,
            va="center",
            family=FONT,
            path_effects=HALO,
        )
        _legend(overlay, scene, s, h, scale)
    _north_arrow(overlay, -_heading(frame), s, w, h)
    overlay.text(
        w - 48 * s,
        46 * s,
        CAPTION,
        color="#e9e6df",
        fontsize=13 * s,
        ha="right",
        va="center",
        family=FONT,
        path_effects=HALO,
    )
    if frame.peak >= 1.0:
        overlay.text(
            w - 110 * s,
            h - 40 * s,
            CREDITS,
            color="#d7d3cb",
            fontsize=11 * s,
            ha="right",
            va="center",
            family=FONT,
            path_effects=HALO,
        )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _heading(frame: Frame) -> float:
    """The frame's camera heading."""
    return frame.shot.heading


def velocity_ticks(
    scale: palette.VelocityScale, width: float, fontsize: float
) -> list[float]:
    """Pick round velocities (cm/s) to label the legend with.

    Boundaries at 1, 2 and 5 of every decade the scale spans, thinned to as
    many as the bar can carry without their labels touching, measuring the
    widest label rather than assuming a count, and always keeping the largest
    (eqvis_workflow's colour bar ticks).

    Parameters
    ----------
    scale : palette.VelocityScale
        The colour scale.
    width : float
        Length of the bar (points).
    fontsize : float
        Size of the tick labels (points).

    Returns
    -------
    list[float]
        Velocities to label, ascending.
    """
    low, high = scale.low, scale.high
    decades = np.arange(np.floor(np.log10(low)), np.ceil(np.log10(high)) + 1)
    boundaries = [
        float(f"{f * 10**d:.6g}")
        for d in decades
        for f in (1.0, 2.0, 5.0)
        if low <= f * 10**d <= high
    ]
    if not boundaries:
        return []
    per_tick = max(len(f"{b:g}") for b in boundaries) * DIGIT_WIDTH * fontsize
    room = max(2, int(width / (per_tick + TICK_GAP)))
    count = len(boundaries)
    stride = max(1, -(-(count - 1) // max(1, room - 1)))
    kept = list(range(0, count, stride))
    if kept[-1] != count - 1:
        if len(kept) > 1 and (count - 1) - kept[-1] < stride:
            kept.pop()
        kept.append(count - 1)
    return [boundaries[i] for i in kept]


def _legend(
    ax: Axes, scene: Scene, s: float, h: float, scale: palette.VelocityScale
) -> None:
    """Draw the ground velocity scale and the fault key."""
    x0, y0, width, height = 48 * s, h - 92 * s, 620 * s, 18 * s
    title = VELOCITY_TITLE
    colours = palette.shaking_cmap()(np.linspace(0.0, 1.0, 512))
    colours[..., 3] = np.maximum(colours[..., 3], 0.35)
    ax.imshow(
        colours[None], extent=(x0, x0 + width, y0 + height, y0), aspect="auto", zorder=8
    )
    ax.add_patch(
        Rectangle((x0, y0), width, height, fill=False, ec="white", lw=1.2 * s, zorder=9)
    )
    ticks = []
    for velocity in velocity_ticks(scale, width, 14 * s):
        x = x0 + float(scale.level(velocity)) * width
        ax.plot(
            [x, x],
            [y0 + height, y0 + height + 6 * s],
            color="white",
            lw=1.2 * s,
            solid_capstyle="butt",
            zorder=9,
        )
        ticks.append((x, f"{velocity:g}", "center"))
    for x, word, align in ticks:
        ax.text(
            x,
            y0 + height + 22 * s,
            word,
            color="white",
            fontsize=14 * s,
            ha=align,
            va="center",
            family=FONT,
            path_effects=HALO,
        )
    ax.text(
        x0,
        y0 - 18 * s,
        title,
        color="white",
        fontsize=15 * s,
        fontweight="bold",
        va="center",
        family=FONT,
        path_effects=HALO,
    )
    if not scene.rupture.is_point:
        key_x, key_y = x0 + width + 40 * s, y0 + height / 2
        ax.plot(
            [key_x, key_x + 40 * s],
            [key_y] * 2,
            color="white",
            lw=4 * s,
            path_effects=[
                patheffects.withStroke(linewidth=7 * s, foreground="#1a1a1a")
            ],
        )
        ax.text(
            key_x + 50 * s,
            key_y,
            "Fault rupturing",
            color="white",
            fontsize=14 * s,
            va="center",
            family=FONT,
            path_effects=HALO,
        )


def _north_arrow(ax: Axes, rotation: float, s: float, w: float, h: float) -> None:
    """Draw a north arrow turned by the camera heading."""
    cx, cy, r = w - 70 * s, h - 80 * s, 28 * s
    a = np.radians(rotation)
    ax.add_patch(Circle((cx, cy), r, color="#1a1a1a", alpha=0.45))
    ax.annotate(
        "",
        xy=(cx + r * 0.8 * np.sin(a), cy - r * 0.8 * np.cos(a)),
        xytext=(cx - r * 0.5 * np.sin(a), cy + r * 0.5 * np.cos(a)),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "white",
            "lw": 2 * s,
            "mutation_scale": 18 * s,
        },
    )
    ax.text(
        cx + r * 1.35 * np.sin(a),
        cy - r * 1.35 * np.cos(a),
        "N",
        color="white",
        fontsize=13 * s,
        ha="center",
        va="center",
        fontweight="bold",
        family=FONT,
        path_effects=HALO,
    )


# ------------------------------------------------------------------ renderer


def _ring(grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Walk a 2-D grid's edge once round, closing the loop."""
    return np.concatenate([grid[0, :], grid[1:, -1], grid[-1, -2::-1], grid[-2::-1, 0]])


class Renderer:
    """An off-screen VTK window showing the scene.

    Parameters
    ----------
    scene : Scene
        The scene.
    settings : Settings
        Frame size and exaggeration.
    """

    def __init__(self, scene: Scene, settings: Settings) -> None:
        """Build the meshes and upload the static textures."""
        import pyvista as pv

        self.pv = pv
        self.scene = scene
        self.settings = settings
        ex = settings.exaggeration
        plotter = pv.Plotter(
            off_screen=True,
            window_size=[settings.width, settings.height],
            lighting="none",
        )
        plotter.background_color = pv.Color(BACKGROUND)
        light = pv.Light(position=(-1, 1, 0.9), light_type="scene light", intensity=0.9)
        light.positional = False
        plotter.add_light(light)

        min_e, min_n, max_e, max_n = scene.context_bounds
        floor = -30.0 * ex
        context = pv.StructuredGrid(
            np.array([[min_e, max_e], [min_e, max_e]]),
            np.array([[max_n, max_n], [min_n, min_n]]),
            np.full((2, 2), floor),
        )
        context.active_texture_coordinates = np.array(
            [[0, 1], [0, 0], [1, 1], [1, 0]], dtype=float
        )
        plotter.add_mesh(
            context,
            texture=self._texture(scene.context_texture, mipmap=True),
            ambient=1.0,
            diffuse=0.0,
        )

        terrain = self._texture(scene.terrain_texture, mipmap=True)
        rows, columns = scene.wave_opacity.shape
        stride = scene.wave_stride
        self.levels = []
        self.skirts = []
        for step in (1, 2):
            e = scene.mesh_easting[::step, ::step]
            n = scene.mesh_northing[::step, ::step]
            z = scene.mesh_elevation[::step, ::step].astype(np.float64) * ex
            base = pv.StructuredGrid(e, n, z)
            base.active_texture_coordinates = scene.mesh_texture_uv[
                ::step, ::step
            ].reshape(-1, 2, order="F")
            over = pv.StructuredGrid(e, n, z + OVERLAY_LIFT * ex * step)
            i = scene.mesh_index[0][::step, ::step] / stride
            j = scene.mesh_index[1][::step, ::step] / stride
            uv = np.stack([(j + 0.5) / columns, 1.0 - (i + 0.5) / rows], axis=-1)
            over.active_texture_coordinates = uv.reshape(-1, 2, order="F")
            base_actor = plotter.add_mesh(
                base, texture=terrain, ambient=0.62, diffuse=0.5, smooth_shading=True
            )
            over_actor = plotter.add_mesh(
                over,
                texture=self._texture(np.zeros((2, 2, 4), np.uint8)),
                ambient=1.0,
                diffuse=0.0,
                smooth_shading=True,
            )
            # VTK's switch for polygon offsets is global, the offset per mapper.
            over_actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            over_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                *OVERLAY_OFFSET
            )
            self.levels.append((base_actor, over_actor))
            # The domain's terrain stands on the flat map; a wall closes the
            # gap beneath its edge when the camera looks across it.
            ring = _ring(e), _ring(n), _ring(z)
            skirt = pv.StructuredGrid(
                np.stack([ring[0]] * 2, axis=1),
                np.stack([ring[1]] * 2, axis=1),
                np.stack([ring[2], np.full_like(ring[2], floor)], axis=1),
            )
            self.skirts.append(
                plotter.add_mesh(skirt, color=SKIRT, ambient=1.0, diffuse=0.0)
            )
        self.plotter = plotter
        # The highest surface drawn: the shaking over the coarse mesh.
        self.top = float(scene.mesh_elevation.max()) * ex + 2 * OVERLAY_LIFT * ex
        self.lut = palette.shaking_lut()

    def _texture(self, image: npt.NDArray[np.uint8], mipmap: bool = False) -> "object":
        """Make a clamped, filtered VTK texture."""
        texture = self.pv.Texture(np.ascontiguousarray(image))
        texture.interpolate = True
        texture.mipmap = mipmap
        texture.repeat = False
        texture.edge_clamp = True
        return texture

    def shaking_texture(
        self, level: npt.NDArray[np.float32] | None, opacity: float
    ) -> npt.NDArray[np.uint8] | None:
        """Colour the shaking for the overlay.

        Parameters
        ----------
        level : np.ndarray | None
            Levels on the velocity scale (`palette.VelocityScale.level`) at
            shaking texture pixels, or None for no shaking.
        opacity : float
            Overall opacity.

        Returns
        -------
        np.ndarray | None
            RGBA texture, or None.
        """
        if level is None:
            return None
        rgba = palette.shaking_rgba(level, self.lut)
        rgba[..., 3] = (rgba[..., 3] * self.scene.wave_opacity * opacity).astype(
            np.uint8
        )
        return rgba

    def render(
        self, shot: Shot, shaking: npt.NDArray[np.uint8] | None
    ) -> npt.NDArray[np.uint8]:
        """Render the 3D view.

        Parameters
        ----------
        shot : Shot
            The camera.
        shaking : np.ndarray | None
            RGBA shaking texture, or None.

        Returns
        -------
        np.ndarray
            The frame, `(height, width, 3)`.
        """
        far = 1 if shot.distance > FAR_LOD_DISTANCE else 0
        for level, (base, over) in enumerate(self.levels):
            base.visibility = level == far
            over.visibility = level == far and shaking is not None
            self.skirts[level].visibility = level == far
        if shaking is not None:
            self.levels[far][1].texture = self._texture(shaking)
        position, focal_point, up = camera_pose(
            self.scene, shot, self.settings.exaggeration
        )
        camera = self.plotter.camera
        camera.position, camera.focal_point, camera.up = (
            tuple(position),
            tuple(focal_point),
            tuple(up),
        )
        camera.view_angle = VERTICAL_FOV
        camera.clipping_range = (
            max(1000.0, NEAR_FRACTION * (position[2] - self.top)),
            FAR_CLIP,
        )
        # Render explicitly: an off-screen plotter's screenshot otherwise
        # returns the first frame again.
        window = self.plotter.render_window
        assert window is not None
        window.Render()
        image = self.plotter.screenshot(return_img=True)
        assert image is not None
        return image[..., :3]

    def close(self) -> None:
        """Release the VTK window."""
        self.plotter.close()


# -------------------------------------------------------------------- frames


@dataclasses.dataclass(frozen=True)
class EndCard:
    """The final map: the strongest broadband shaking."""

    pgv: npt.NDArray[np.float32]
    """Peak ground velocity (cm/s) at shaking texture pixels; NaN for none."""
    scale: palette.VelocityScale
    """The movie's colour scale, topped by the strongest of `pgv` on land."""
    words: dict[str, str | None]
    """Strongest shaking at each town, in words."""
    weak: bool = False
    """Whether no shaking on land reaches a word, so the map says so."""


@dataclasses.dataclass(frozen=True)
class Job:
    """One worker's share of the movie: a contiguous run of frames."""

    scene: Scene
    """The scene."""
    frames: list[Frame]
    """Every frame of the movie."""
    start: int
    """First frame to encode."""
    stop: int
    """One past the last frame to encode."""
    velocity: list[Path]
    """`surf_mag` images (ground velocity magnitude)."""
    reference: float
    """The strongest low-frequency ground velocity (m/s), which the waves are
    scaled to the top of the colour scale by."""
    end_card: EndCard
    """The final map, and the colour scale."""
    settings: Settings
    """How to render."""
    output: Path
    """Where to write this segment."""


def waves(
    sequence: ImageSequence,
    time: float,
    stride: int,
    reference: float,
    scale: palette.VelocityScale,
) -> npt.NDArray[np.float32]:
    """The travelling waves at a time, on the movie's velocity scale.

    SW4's low-frequency velocity is scaled so its strongest (`reference`)
    lands on the top of the scale, the strongest broadband shaking.

    Parameters
    ----------
    sequence : ImageSequence
        `surf_mag` images.
    time : float
        Simulation time (s).
    stride : int
        Image points per texture pixel.
    reference : float
        The strongest low-frequency ground velocity (m/s).
    scale : palette.VelocityScale
        The colour scale.

    Returns
    -------
    np.ndarray
        Levels on the scale (`palette.VelocityScale.level`).
    """
    velocity = sequence.at(time)[::stride, ::stride]
    return scale.level(velocity * (scale.high / max(reference, 1e-30)))


def fault_lines(
    scene: Scene, projector: Projector, time: float, exaggeration: float
) -> list[tuple[npt.NDArray[np.float64], str, float]]:
    """Project the fault, split where terrain hides it.

    Parameters
    ----------
    scene : Scene
        The scene.
    projector : Projector
        The frame's projection.
    time : float
        Simulation time (s); the ruptured part is drawn bright.
    exaggeration : float
        Vertical exaggeration.

    Returns
    -------
    list[tuple[np.ndarray, str, float]]
        Screen polylines with colour and line width.
    """
    if scene.rupture.is_point:
        return []
    lines = []
    for line, start in scene.rupture.segments:
        z = scene.height_at(line[:, 0], line[:, 1]) * exaggeration + 80.0
        screen = projector(np.column_stack([line, z]))
        seen = visible(scene, projector.position, line, exaggeration) & np.isfinite(
            screen[:, 0]
        )
        for mask, colour, width in (
            (seen, "#6b6660", 2.5),
            (seen & (start <= time), "#ffffff", 5.0),
        ):
            indices = np.flatnonzero(mask)
            gaps = np.flatnonzero(np.diff(indices) > 1) + 1
            lines.extend(
                (screen[run], colour, width)
                for run in np.split(indices, gaps)
                if run.size > 1
            )
    return lines


def fault_planes(
    scene: Scene, projector: Projector, exaggeration: float
) -> list[npt.NDArray[np.float64]]:
    """Project the fault planes' surface outlines, split where terrain hides them.

    Each plane is cut along strike into the stretches whose top and bottom
    edges are both in view.

    Parameters
    ----------
    scene : Scene
        The scene.
    projector : Projector
        The frame's projection.
    exaggeration : float
        Vertical exaggeration.

    Returns
    -------
    list[np.ndarray]
        Screen outlines, `(n, 2)` each; none for a point source.
    """
    if scene.rupture.is_point:
        return []
    outlines = []
    for outline in scene.rupture.planes:
        z = scene.height_at(outline[:, 0], outline[:, 1]) * exaggeration + 80.0
        screen = projector(np.column_stack([outline, z]))
        seen = visible(scene, projector.position, outline, exaggeration) & np.isfinite(
            screen[:, 0]
        )
        n = len(outline) // 2
        top, bottom = screen[:n], screen[n:][::-1]
        indices = np.flatnonzero(seen[:n] & seen[n:][::-1])
        gaps = np.flatnonzero(np.diff(indices) > 1) + 1
        outlines.extend(
            np.concatenate([top[run], bottom[run][::-1]])
            for run in np.split(indices, gaps)
            if run.size > 1
        )
    return outlines


def render_segment(job: Job) -> Path:
    """Render and encode one worker's frames.

    Parameters
    ----------
    job : Job
        The frames to render.

    Returns
    -------
    Path
        The encoded segment.
    """
    os.environ.setdefault("LP_NUM_THREADS", "1")
    scene, settings = job.scene, job.settings
    ex = settings.exaggeration
    sequence = ImageSequence(job.velocity)
    renderer = Renderer(scene, settings)
    placer = LabelPlacer(scene, settings)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{settings.width}x{settings.height}",
            "-r",
            str(settings.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(settings.crf),
            "-g",
            str(2 * settings.fps),
            "-pix_fmt",
            "yuv420p",
            str(job.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    # Settle the labels on the frames before this segment so fades carry over.
    for frame in job.frames[
        max(0, job.start - round(LABEL_FADE * settings.fps)) : job.start
    ]:
        projector = Projector(*camera_pose(scene, frame.shot, ex), settings)
        final = frame.peak >= 1.0
        placer.place(
            projector,
            ex,
            MAX_FINAL_LABELS if final else MAX_LABELS,
            job.end_card.words if final else None,
        )
    for frame in job.frames[job.start : job.stop]:
        final = frame.peak >= 1.0
        if frame.kind in ("intro", "dive"):
            level = None
        else:
            scale = job.end_card.scale
            level = waves(sequence, frame.time, scene.wave_stride, job.reference, scale)
            if frame.peak > 0:
                strongest = scale.level(job.end_card.pgv)
                level = level * (1 - frame.peak) + strongest * frame.peak
        shaking = renderer.shaking_texture(level, FINAL_OPACITY if final else 1.0)
        image = renderer.render(frame.shot, shaking)
        projector = Projector(*camera_pose(scene, frame.shot, ex), settings)
        labels = placer.place(
            projector,
            ex,
            MAX_FINAL_LABELS if final else MAX_LABELS,
            job.end_card.words if final else None,
        )
        fault = fault_lines(
            scene, projector, frame.time if frame.peak == 0 else np.inf, ex
        )
        epicentre_z = scene.height_at(*scene.rupture.hypocentre) * ex + 300.0
        epicentre = projector(
            np.array([[*scene.rupture.hypocentre, float(epicentre_z)]])
        )[0]
        encoder.stdin.write(
            draw_hud(
                image,
                scene,
                frame,
                labels,
                fault,
                epicentre,
                settings,
                job.end_card.weak,
                scale=job.end_card.scale,
                planes=fault_planes(scene, projector, ex),
            ).tobytes()
        )
    encoder.stdin.close()
    renderer.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to encode {job.output}")
    return job.output


def render_movie(
    scene: Scene,
    frames: list[Frame],
    velocity: list[Path],
    reference: float,
    end_card: EndCard,
    settings: Settings,
    output: Path,
    workers: int,
) -> None:
    """Render every frame across worker processes and write the movie.

    Parameters
    ----------
    scene : Scene
        The scene.
    frames : list[Frame]
        Every frame of the movie.
    velocity : list[Path]
        `surf_mag` images.
    reference : float
        The strongest low-frequency ground velocity (m/s), which the waves are
        scaled to the top of the colour scale by.
    end_card : EndCard
        The final map.
    settings : Settings
        How to render.
    output : Path
        The MP4 to write.
    workers : int
        Worker processes.

    Raises
    ------
    RuntimeError
        If ffmpeg is missing or fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to encode the movie, but is not on PATH.")
    workers = max(1, min(workers, len(frames) // (2 * settings.fps) or 1))
    bounds = np.linspace(0, len(frames), workers + 1).round().astype(int)
    with tempfile.TemporaryDirectory(prefix="shake_movie_") as directory:
        jobs = [
            Job(
                scene,
                frames,
                int(a),
                int(b),
                velocity,
                reference,
                end_card,
                settings,
                Path(directory) / f"segment{k:03d}.mp4",
            )
            for k, (a, b) in enumerate(itertools.pairwise(bounds))
        ]
        with get_context("spawn").Pool(workers) as pool:
            segments = pool.map(render_segment, jobs)
        listing = Path(directory) / "segments.txt"
        listing.write_text("".join(f"file '{segment}'\n" for segment in segments))
        joined = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=False,
        )
        if joined.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to join the segments into {output}")
