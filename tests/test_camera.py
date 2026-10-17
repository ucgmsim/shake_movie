"""Tests for the automatic camera."""

import numpy as np
import numpy.typing as npt
import pytest

from shake_movie import camera
from shake_movie.scene import Rupture, Town

DOMAIN = np.array(
    [[1.40e6, 5.10e6], [1.70e6, 5.10e6], [1.70e6, 5.35e6], [1.40e6, 5.35e6]]
)
FAULT = np.column_stack([np.linspace(1.50e6, 1.60e6, 50), np.full(50, 5.25e6)])


def rupture() -> Rupture:
    """An east-west fault rupturing eastward at 2.8 km/s."""
    start = np.hypot(*(FAULT - FAULT[0]).T) / 2800.0
    return Rupture([(FAULT, start)], FAULT[0].copy(), float(start[-1]))


def felt(time: float) -> npt.NDArray[np.float64]:
    """Shaking spreading from the epicentre at 3 km/s."""
    radius = 3000.0 * time
    angle = np.linspace(0, 2 * np.pi, 64)
    return FAULT[0] + radius * np.column_stack([np.cos(angle), np.sin(angle)])


@pytest.fixture
def path() -> camera.CameraPath:
    """The camera for the test rupture."""
    towns = [Town("South", 100_000, 1.55e6, 5.15e6)]
    return camera.plan_camera(rupture(), DOMAIN, towns, felt, 60.0, 16 / 9)


def test_camera_glides_frame_to_frame(path: camera.CameraPath) -> None:
    """At 30 fps the camera pulls back smoothly: no holding then jumping at
    the 0.5 s planning interval (the stutter in the first preview)."""
    times = np.arange(0.0, 60.0, 1 / 30)
    log_distance = np.log([path.at(t).distance for t in times])
    step = np.diff(log_distance)
    assert np.all(step >= -1e-12), "the camera only pulls back"
    assert np.abs(np.diff(step)).max() < 0.05 * step.max()
    for name in ("easting", "northing", "pitch"):
        values = np.array([getattr(path.at(t), name) for t in times])
        jumps = np.abs(np.diff(values, 2))
        assert jumps.max() < 0.05 * np.abs(np.diff(values)).max() + 1e-9


def test_camera_ends_on_the_whole_domain(path: camera.CameraPath) -> None:
    """By the end the camera has pulled back to take in the domain."""
    assert path.at(60.0).distance == pytest.approx(path.domain_shot.distance, rel=0.02)
    assert path.at(0.0).distance < 0.5 * path.domain_shot.distance
    assert (
        camera.PITCH_TIGHT
        <= path.at(0.0).pitch
        < path.at(60.0).pitch
        <= camera.PITCH_WIDE + 1e-9
    )


def test_camera_stands_where_people_live() -> None:
    """The camera looks across the fault from the populated side."""
    south = [Town("South", 100_000, 1.55e6, 5.15e6)]
    north = [Town("North", 100_000, 1.55e6, 5.30e6)]
    assert camera.choose_heading(rupture(), south, DOMAIN) == pytest.approx(
        0.0, abs=1e-6
    )
    # Facing south would put north at the bottom; north stays within 60 degrees of up.
    assert camera.choose_heading(rupture(), north, DOMAIN) in (
        pytest.approx(60.0),
        pytest.approx(300.0),
    )


def test_point_sources_face_north() -> None:
    """A rupture too short to draw faces north."""
    point = Rupture([(FAULT[:2] * 0 + FAULT[0], np.zeros(2))], FAULT[0].copy(), 0.0)
    assert camera.choose_heading(point, [], DOMAIN) == 0.0


def test_timeline_plays_in_real_time(path: camera.CameraPath) -> None:
    """The simulation plays at 1x between the title and the final map."""
    frames = camera.timeline(path, DOMAIN, 60.0, 30, 16 / 9)
    kinds = [frame.kind for frame in frames]
    assert kinds.count("simulation") == 60 * 30
    assert kinds.index("simulation") == (camera.INTRO + camera.DIVE) * 30
    assert kinds[-1] == "final" and frames[-1].shot.pitch == camera.PITCH_MAP
    assert all(frame.title > 0 for frame in frames if frame.kind == "intro")


def test_blend_takes_the_short_way_round() -> None:
    """Blending headings turns through the smaller angle."""
    a = camera.Shot(0.0, 0.0, 1e3, 40.0, 350.0)
    b = camera.Shot(10.0, 0.0, 1e5, 50.0, 10.0)
    middle = camera.blend(a, b, 0.5)
    assert middle.heading % 360 == pytest.approx(0.0)
    assert middle.distance == pytest.approx(1e4)
