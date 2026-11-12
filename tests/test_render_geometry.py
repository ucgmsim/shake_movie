"""Tests for the camera and sight-line geometry that places overlays.

These need no OpenGL: they check the maths that decides where the camera sits,
where world points land on screen, and which ground the terrain hides.
"""

import dataclasses
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from shake_movie import palette, render
from shake_movie.camera import Shot
from shake_movie.scene import Rupture, Scene


@dataclasses.dataclass
class _Terrain:
    """Just enough of a `Scene` for the geometry: a north-up height grid
    looked up with the scene's own bilinear `height_at`, and a rupture."""

    height: npt.NDArray[np.float32]
    height_bounds: tuple[float, float, float, float] = (0.0, 0.0, 10e3, 10e3)
    rupture: Rupture | None = None

    height_at = Scene.height_at


def _scene(
    height: npt.ArrayLike,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 10e3, 10e3),
    rupture: Rupture | None = None,
) -> Scene:
    """A scene whose terrain is `height`, rows from north to south."""
    return cast(Scene, _Terrain(np.asarray(height, np.float32), bounds, rupture))


def _grid(spacing: float = 50.0) -> tuple[npt.NDArray[np.float64], ...]:
    """Easting and northing of every node of a 10 km north-up grid."""
    axis = np.arange(0.0, 10e3 + spacing / 2, spacing)
    easting, northing = np.meshgrid(axis, axis[::-1])
    return easting, northing


# ------------------------------------------------------------- camera pose


def test_camera_looks_at_half_the_exaggerated_ground() -> None:
    """The focal point sits at half the exaggerated ground height, and the
    camera is `distance` back along the view, with an up vector square to it."""
    scene = _scene(np.full((11, 11), 400.0))
    shot = Shot(easting=3e3, northing=6e3, distance=5e3, pitch=40.0, heading=30.0)
    position, focal_point, up = render.camera_pose(scene, shot, 2.0)
    # Ground 400 m, exaggerated twice, halved: the focal point is at 400 m.
    np.testing.assert_allclose(focal_point, [3e3, 6e3, 400.0])
    h, p = np.radians(30.0), np.radians(40.0)
    forward = np.array([np.sin(h) * np.cos(p), np.cos(h) * np.cos(p), -np.sin(p)])
    np.testing.assert_allclose(position, focal_point - 5e3 * forward)
    np.testing.assert_allclose(
        up, [np.sin(h) * np.sin(p), np.cos(h) * np.sin(p), np.cos(p)]
    )
    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert up @ forward == pytest.approx(0.0, abs=1e-12)
    # Above the focal point, behind it along the heading.
    assert position[2] == pytest.approx(400.0 + 5e3 * np.sin(p))


def test_camera_height_scales_with_exaggeration() -> None:
    """Exaggerating the terrain raises the focal point in proportion."""
    scene = _scene(np.full((11, 11), 300.0))
    shot = Shot(easting=5e3, northing=5e3, distance=1e3, pitch=90.0, heading=0.0)
    for exaggeration, z in ((1.0, 150.0), (3.0, 450.0)):
        position, focal_point, up = render.camera_pose(scene, shot, exaggeration)
        assert focal_point[2] == pytest.approx(z)
        # Looking straight down, the camera is right above the focal point
        # and its up vector points north.
        np.testing.assert_allclose(position, [5e3, 5e3, z + 1e3], atol=1e-9)
        np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-12)


# --------------------------------------------------------------- projection


def test_projection_hides_points_within_a_metre_of_the_camera() -> None:
    """Points less than a metre in front of the camera, or behind it, project
    to NaN; points further out land on screen."""
    settings = render.Settings(width=640, height=360)
    projector = render.Projector(
        np.zeros(3), np.array([0.0, 10.0, 0.0]), np.array([0.0, 0.0, 1.0]), settings
    )
    screen = projector(
        np.array(
            [
                [0.0, 1.0, 0.0],  # exactly a metre ahead: too close
                [0.0, 0.5, 0.0],
                [0.0, -5.0, 0.0],  # behind
                [0.0, 1.5, 0.0],
                [0.0, 1000.0, 0.0],
            ]
        )
    )
    assert np.isnan(screen[:3]).all()
    # Straight ahead lands in the middle of the frame.
    np.testing.assert_allclose(screen[3:], [[320.0, 180.0], [320.0, 180.0]])


# ------------------------------------------------------------- sight lines


def test_sight_lines_follow_the_straight_path_to_each_point() -> None:
    """In a canyon cut along the line from the camera to a point, the point is
    in view; any step off that line meets the canyon wall."""
    easting, northing = _grid()
    camera = np.array([2e3, 3e3, 300.0])
    target = np.array([8e3, 7e3])
    along = target - camera[:2]
    # Distance of each node from the camera-to-point line, in plan.
    offset = np.abs(
        (easting - camera[0]) * along[1] - (northing - camera[1]) * along[0]
    ) / np.linalg.norm(along)
    height = np.where(offset < 300.0, 0.0, 1000.0)
    scene = _scene(height)
    assert render.visible(scene, camera, target[None, :], 1.0).tolist() == [True]
    # Point the same line elsewhere and the wall hides it.
    assert render.visible(scene, camera, np.array([[8e3, 4e3]]), 1.0).tolist() == [
        False
    ]
    # Swapping the camera's easting and northing moves the line out of the canyon.
    assert not render.visible(scene, camera[[1, 0, 2]], target, 1.0)[0]


def test_ridge_in_front_of_a_point_hides_it() -> None:
    """A ridge just in front of a point hides it from a camera high above."""
    easting, _ = _grid(100.0)
    camera = np.array([0.0, 5e3, 1000.0])
    target = np.array([[10e3, 5e3]])
    # The last step along the sight line passes over e = 9.7 km at 78.5 m.
    for ridge, seen in ((200.0, False), (70.0, True)):
        height = np.where((easting >= 9.6e3) & (easting <= 9.9e3), ridge, 0.0)
        assert render.visible(_scene(height), camera, target, 1.0)[0] == seen


def test_sight_lines_end_fifty_metres_above_the_ground() -> None:
    """Each sight line ends 50 m above the exaggerated ground at its point:
    a ridge just short of the point hides it only if it reaches that line."""
    easting, _ = _grid(100.0)
    camera = np.array([0.0, 5e3, 10.0])
    target = np.array([[10e3, 5e3]])
    # The line rises from 10 m to 50 m; at its last step (e = 9.7 km) it is
    # 10 + 40 * 0.97 = 48.8 m up.
    for ridge, seen in ((49.3, False), (48.3, True)):
        height = np.where((easting >= 9.6e3) & (easting <= 9.9e3), ridge, 0.0)
        assert render.visible(_scene(height), camera, target, 1.0)[0] == seen
    # On flat ground, a camera low down sees the point over the ground.
    flat = _scene(np.zeros_like(easting))
    assert render.visible(flat, camera, target, 1.0)[0]


def test_sight_lines_end_above_the_exaggerated_point() -> None:
    """A point on a plateau is seen from just above the exaggerated plateau:
    its own height, exaggerated, sets where the sight line ends."""
    easting, _ = _grid(100.0)
    # A plateau 1 km high east of e = 5 km, sea level to the west.
    scene = _scene(np.where(easting >= 5e3, 1000.0, 0.0))
    # 100 m above the plateau, exaggerated twice to 2 km.
    camera = np.array([9e3, 1e3, 2100.0])
    points = np.array([[6e3, 2e3], [6e3, 8e3]])
    assert render.visible(scene, camera, points, 2.0).tolist() == [True, True]
    # Unexaggerated, the same camera is 1.1 km above the plateau and the
    # sight line still clears it.
    assert render.visible(scene, camera - [0, 0, 1000.0], points, 1.0).all()
    # Below the exaggerated plateau, the camera sees nothing on it.
    assert not render.visible(scene, camera - [0, 0, 200.0], points, 2.0).any()


def test_sample_count_sets_what_a_thin_wall_can_hide() -> None:
    """With only the two end steps, a wall halfway along is stepped over;
    with the default steps, it hides the point."""
    easting, _ = _grid(100.0)
    scene = _scene(np.where(np.abs(easting - 5e3) <= 200.0, 1000.0, 0.0))
    camera = np.array([0.0, 5e3, 100.0])
    target = np.array([[10e3, 5e3]])
    assert render.visible(scene, camera, target, 1.0, samples=2)[0]
    assert not render.visible(scene, camera, target, 1.0)[0]


# ------------------------------------------------------------ point sources


def test_point_source_draws_no_fault() -> None:
    """A rupture too short to draw has no lines, planes or slipping cells."""
    line = np.array([[5e3, 5e3], [5e3 + 1.0, 5e3]])
    rupture = Rupture(
        segments=[(line, np.zeros(2))],
        hypocentre=np.array([5e3, 5e3]),
        end_time=0.0,
        planes=[np.vstack([line, line[::-1] + [0.0, 1.0]])],
    )
    assert rupture.is_point
    scene = _scene(np.zeros((11, 11)), rupture=rupture)
    settings = render.Settings(width=640, height=360)
    projector = render.Projector(
        np.array([5e3, 0.0, 5e3]),
        np.array([5e3, 5e3, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        settings,
    )
    assert render.fault_lines(scene, projector, 10.0, 1.0) == []
    assert render.fault_planes(scene, projector, 1.0) == render.FaultPlanes([], [])
    assert render.fault_slip(scene, projector, 10.0, 1.0) is None


# ----------------------------------------------------------------- overlays


def test_slip_off_screen_leaves_the_frame_alone() -> None:
    """Cells entirely off the frame blend nothing in; cells on it do."""
    image = np.full((60, 80, 3), 30, dtype=np.uint8)
    square = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    red = np.array([[1.0, 0.0, 0.0, 1.0]])
    off = render.SlipCells((square + [200.0, 10.0])[None], red)
    np.testing.assert_array_equal(render._blend_slip(image.copy(), off, 1.0), image)
    on = render.SlipCells((square + [30.0, 20.0])[None], red)
    blended = render._blend_slip(image.copy(), on, 1.0)
    assert blended[30, 40, 0] > 200 and blended[30, 40, 1] < 60
    np.testing.assert_array_equal(blended[:10], image[:10])


def test_velocity_ticks_keep_the_largest_when_thinned() -> None:
    """Thinned ticks step evenly from the smallest and end on the largest,
    dropping the evenly spaced tick that would crowd it."""
    scale = palette.VelocityScale(0.18)
    # Six boundaries, 0.002 to 0.1, room for four labels: every second one
    # (0.002, 0.01, 0.05) then the largest, without 0.05 crowding 0.1.
    assert render.velocity_ticks(scale, 200.0, 14.0) == [0.002, 0.01, 0.1]
    scale = palette.VelocityScale(1.0)
    # Seven boundaries, 0.01 to 1: every second one leaves 1 as the last.
    assert render.velocity_ticks(scale, 200.0, 14.0) == [0.01, 0.05, 0.2, 1.0]
