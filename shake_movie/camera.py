"""The camera and the movie's timeline, worked out from the simulation alone.

Every shot comes from the rupture and the shaking; nothing is set per event:

1. Heading: look across the fault from the side where more people live, so
   towns sit in the foreground, keeping north within 60 degrees of screen-up.
2. Framing: the part of the fault that has ruptured, plus everywhere shaking at
   "Light" or above, padded by 30% and at least `MIN_WIDTH` across.
3. The camera only ever pulls back: the framed size is a running maximum.
4. The path is smoothed with look-ahead (the movie is rendered after the
   run), then evaluated at every frame through monotone splines, so the
   camera glides rather than stepping at SW4's image interval.
5. The pitch eases from `PITCH_TIGHT` to `PITCH_WIDE` as the camera pulls
   back, and the heading drifts slowly for parallax.
6. The movie ends by tilting up to a north-up map of the strongest shaking.
"""

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import interpolate, ndimage

from shake_movie.scene import Rupture, Town

VERTICAL_FOV = 30.0
"""Camera vertical field of view (degrees)."""
MIN_WIDTH = 70e3
"""Narrowest view of the ground (m), so small events still show context."""
FRAMING_PAD = 1.3
"""Padding around what is framed."""
PITCH_TIGHT, PITCH_WIDE = 40.0, 55.0
"""Camera pitch below the horizon (degrees) when close and when wide."""
PITCH_MAP = 89.9
"""Pitch of the final, straight-down map shot (degrees)."""
SMOOTHING = 2.5
"""Look-ahead smoothing of the camera path (s)."""
HEADING_DRIFT = 12.0
"""Total slow turn of the camera over the simulation, for parallax (degrees)."""
MAX_NORTH_OFFSET = 60.0
"""Furthest north may turn from screen-up (degrees)."""
FELT_MMI = 4.0
"""Shaking framed by the camera: "Light" and above."""
PLAN_INTERVAL = 0.5
"""Spacing of the samples the camera path is planned from (s)."""

INTRO, DIVE, PULL_UP, HOLD = 4.0, 2.0, 3.0, 6.0
"""Durations (s) of the title shot, the dive to the epicentre, the tilt up to
the final map and the hold on it."""


@dataclasses.dataclass(frozen=True)
class Shot:
    """A camera placement."""

    easting: float
    """NZTM easting of the point looked at (m)."""
    northing: float
    """NZTM northing of the point looked at (m)."""
    distance: float
    """Distance from the camera to that point (m)."""
    pitch: float
    """Angle below the horizon (degrees)."""
    heading: float
    """Bearing the camera faces (degrees clockwise from north)."""


def blend(a: Shot, b: Shot, s: float) -> Shot:
    """Ease between two shots.

    Parameters
    ----------
    a : Shot
        The shot at `s = 0`.
    b : Shot
        The shot at `s = 1`.
    s : float
        Progress in [0, 1], eased with a smoothstep.

    Returns
    -------
    Shot
        The blended shot. Distance blends geometrically and heading by the
        shorter turn.
    """
    s = s * s * (3 - 2 * s)
    turn = (b.heading - a.heading + 180.0) % 360.0 - 180.0
    return Shot(
        easting=a.easting + (b.easting - a.easting) * s,
        northing=a.northing + (b.northing - a.northing) * s,
        distance=float(
            np.exp(np.log(a.distance) + (np.log(b.distance) - np.log(a.distance)) * s)
        ),
        pitch=a.pitch + (b.pitch - a.pitch) * s,
        heading=a.heading + turn * s,
    )


def fit_distance(width: float, depth: float, pitch: float, aspect: float) -> float:
    """Find how far back the camera must be to fit an area of ground.

    Parameters
    ----------
    width : float
        Ground width across the screen (m).
    depth : float
        Ground depth into the screen (m).
    pitch : float
        Camera pitch (degrees below the horizon).
    aspect : float
        Frame width over height.

    Returns
    -------
    float
        Camera distance (m).
    """
    half_v = np.radians(VERTICAL_FOV) / 2
    half_h = np.arctan(np.tan(half_v) * aspect)
    return float(
        max(
            (width / 2) / np.tan(half_h),
            (depth * np.sin(np.radians(pitch)) / 2) / np.tan(half_v),
        )
    )


def _axes(heading: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Ground directions facing into and across the screen for a heading."""
    h = np.radians(heading)
    return np.array([np.sin(h), np.cos(h)]), np.array([np.cos(h), -np.sin(h)])


def extents(
    points: npt.NDArray[np.float64], heading: float
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Measure points as the camera sees them.

    Parameters
    ----------
    points : np.ndarray
        NZTM points, `(n, 2)`.
    heading : float
        Camera heading (degrees).

    Returns
    -------
    centre : np.ndarray
        NZTM centre of the points' screen-aligned bounding box.
    width : float
        Extent across the screen (m).
    depth : float
        Extent into the screen (m).
    """
    forward, right = _axes(heading)
    across = points @ right
    into = points @ forward
    centre = (
        right * (across.min() + across.max()) / 2
        + forward * (into.min() + into.max()) / 2
    )
    return centre, float(np.ptp(across)), float(np.ptp(into))


def choose_heading(
    rupture: Rupture, towns: list[Town], domain: npt.NDArray[np.float64]
) -> float:
    """Face across the fault from the side where more people live.

    Parameters
    ----------
    rupture : Rupture
        The rupture.
    towns : list[Town]
        Candidate towns.
    domain : np.ndarray
        NZTM corners of the simulated domain.

    Returns
    -------
    float
        Camera heading (degrees). Point sources face north.
    """
    if rupture.is_point:
        return 0.0
    line = np.concatenate([segment for segment, _ in rupture.segments])
    strike = line[-1] - line[0]
    strike = strike / np.hypot(*strike)
    normal = np.array([strike[1], -strike[0]])
    middle = line.mean(axis=0)
    from matplotlib.path import Path as MplPath

    inside = MplPath(domain)
    score = sum(
        np.sign((np.array([town.easting, town.northing]) - middle) @ normal)
        * np.log10(town.population)
        for town in towns
        if inside.contains_point((town.easting, town.northing))
    )
    # The camera stands on the populated side and looks back across the fault.
    facing = -normal if score > 0 else normal
    heading = np.degrees(np.arctan2(facing[0], facing[1]))
    offset = (heading + 180.0) % 360.0 - 180.0
    return float(np.clip(offset, -MAX_NORTH_OFFSET, MAX_NORTH_OFFSET) % 360.0)


class CameraPath:
    """A smooth camera path through the simulation, sampled at any time.

    Parameters
    ----------
    times : np.ndarray
        Sample times (s).
    easting : np.ndarray
        Looked-at easting at each sample (m).
    northing : np.ndarray
        Looked-at northing at each sample (m).
    distance : np.ndarray
        Camera distance at each sample (m).
    pitch : np.ndarray
        Pitch at each sample (degrees).
    heading : np.ndarray
        Heading at each sample (degrees).
    domain_shot : Shot
        A shot of the whole domain, used for the title.
    """

    def __init__(
        self,
        times: npt.NDArray[np.float64],
        easting: npt.NDArray[np.float64],
        northing: npt.NDArray[np.float64],
        distance: npt.NDArray[np.float64],
        pitch: npt.NDArray[np.float64],
        heading: npt.NDArray[np.float64],
        domain_shot: Shot,
    ) -> None:
        """Fit monotone splines through planned camera samples."""
        self.times = times
        self.domain_shot = domain_shot
        self._splines = {
            "easting": interpolate.PchipInterpolator(times, easting),
            "northing": interpolate.PchipInterpolator(times, northing),
            "distance": interpolate.PchipInterpolator(times, np.log(distance)),
            "pitch": interpolate.PchipInterpolator(times, pitch),
            "heading": interpolate.PchipInterpolator(times, heading),
        }

    def at(self, time: float) -> Shot:
        """Place the camera at a simulation time.

        Parameters
        ----------
        time : float
            Simulation time (s), clamped to the planned span.

        Returns
        -------
        Shot
            The camera.
        """
        t = float(np.clip(time, self.times[0], self.times[-1]))
        values = {name: float(spline(t)) for name, spline in self._splines.items()}
        values["distance"] = float(np.exp(values["distance"]))
        return Shot(**values)


def plan_camera(
    rupture: Rupture,
    domain: npt.NDArray[np.float64],
    towns: list[Town],
    felt: Callable[[float], npt.NDArray[np.float64]],
    end_time: float,
    aspect: float,
) -> CameraPath:
    """Plan the camera path from the rupture and the shaking.

    Parameters
    ----------
    rupture : Rupture
        The rupture.
    domain : np.ndarray
        NZTM corners of the simulated domain.
    towns : list[Town]
        Candidate towns, for the heading.
    felt : Callable[[float], np.ndarray]
        NZTM points shaking at `FELT_MMI` or above at a time, `(n, 2)`.
    end_time : float
        End of the simulation (s).
    aspect : float
        Frame width over height.

    Returns
    -------
    CameraPath
        The camera path.
    """
    heading = choose_heading(rupture, towns, domain)
    domain_centre, domain_width, domain_depth = extents(domain, heading)
    domain_distance = fit_distance(
        domain_width * 1.05, domain_depth * 1.05, PITCH_WIDE, aspect
    )
    times = np.arange(0.0, end_time + PLAN_INTERVAL, PLAN_INTERVAL)
    centres, widths, depths = [], [], []
    for t in times:
        framed = [rupture.hypocentre[None, :]]
        if not rupture.is_point:
            framed.extend(rupture.ruptured(t))
        framed.append(felt(t))
        centre, width, depth = extents(np.concatenate(framed), heading)
        centres.append(centre)
        widths.append(max(width * FRAMING_PAD, MIN_WIDTH))
        depths.append(max(depth * FRAMING_PAD, MIN_WIDTH / aspect))
    # Only ever pull back.
    widths = np.maximum.accumulate(widths)
    depths = np.maximum.accumulate(depths)
    distance = np.minimum(
        [
            fit_distance(w, d, PITCH_TIGHT, aspect)
            for w, d in zip(widths, depths, strict=True)
        ],
        domain_distance,
    )
    # Drift toward the domain's centre as the view widens to take it all in.
    span = max(domain_distance - distance[0], 1.0)
    widening = np.clip((distance - distance[0]) / span, 0.0, 1.0)[:, None]
    centre = np.asarray(centres) * (1 - widening) + domain_centre * widening
    sigma = SMOOTHING / PLAN_INTERVAL
    centre = ndimage.gaussian_filter1d(centre, sigma, axis=0, mode="nearest")
    distance = np.maximum.accumulate(
        np.exp(ndimage.gaussian_filter1d(np.log(distance), sigma, mode="nearest"))
    )
    pitch = ndimage.gaussian_filter1d(
        PITCH_TIGHT + (PITCH_WIDE - PITCH_TIGHT) * widening[:, 0], sigma, mode="nearest"
    )
    drift = heading - HEADING_DRIFT / 2 + HEADING_DRIFT * times / max(times[-1], 1.0)
    domain_shot = Shot(
        float(domain_centre[0]),
        float(domain_centre[1]),
        domain_distance,
        PITCH_WIDE,
        heading - HEADING_DRIFT / 2,
    )
    return CameraPath(
        times, centre[:, 0], centre[:, 1], distance, pitch, drift, domain_shot
    )


def map_shot(domain: npt.NDArray[np.float64], aspect: float) -> Shot:
    """Frame the whole domain from straight above, north up.

    Parameters
    ----------
    domain : np.ndarray
        NZTM corners of the simulated domain.
    aspect : float
        Frame width over height.

    Returns
    -------
    Shot
        The map shot.
    """
    low, high = domain.min(axis=0), domain.max(axis=0)
    centre = (low + high) / 2
    width, depth = high - low
    distance = fit_distance(width * 1.08, depth * 1.12, PITCH_MAP, aspect)
    return Shot(float(centre[0]), float(centre[1]), distance, PITCH_MAP, 0.0)


@dataclasses.dataclass(frozen=True)
class Frame:
    """One frame of the movie."""

    kind: str
    """"intro", "dive", "simulation", "pull_up" or "final"."""
    time: float
    """Simulation time shown (s)."""
    shot: Shot
    """Where the camera is."""
    title: float = 0.0
    """Opacity of the title card."""
    peak: float = 0.0
    """How far the shaking has turned into the strongest shaking map."""


def timeline(
    path: CameraPath,
    domain: npt.NDArray[np.float64],
    end_time: float,
    fps: int,
    aspect: float,
) -> list[Frame]:
    """Lay out every frame of the movie.

    Parameters
    ----------
    path : CameraPath
        The camera path through the simulation.
    domain : np.ndarray
        NZTM corners of the simulated domain.
    end_time : float
        End of the simulation (s).
    fps : int
        Frames per second.
    aspect : float
        Frame width over height.

    Returns
    -------
    list[Frame]
        The frames, in order. The simulation plays in real time.
    """
    wide = path.domain_shot
    wide_end = dataclasses.replace(wide, heading=wide.heading + 4.0)
    start = path.at(0.0)
    final = map_shot(domain, aspect)
    frames = []
    n = round(INTRO * fps)
    for i in range(n):
        s = i / n
        fade = float(np.clip(min(s / 0.2, (1 - s) / 0.2), 0.01, 1.0))
        frames.append(Frame("intro", 0.0, blend(wide, wide_end, s), title=fade))
    n = round(DIVE * fps)
    frames.extend(Frame("dive", 0.0, blend(wide_end, start, i / n)) for i in range(n))
    n = round(end_time * fps)
    frames.extend(Frame("simulation", i / fps, path.at(i / fps)) for i in range(n))
    last = path.at(end_time)
    n = round(PULL_UP * fps)
    for i in range(n):
        s = i / n
        frames.append(
            Frame("pull_up", end_time, blend(last, final, s), peak=s * s * (3 - 2 * s))
        )
    n = round(HOLD * fps)
    frames.extend(Frame("final", end_time, final, peak=1.0) for _ in range(n))
    return frames
