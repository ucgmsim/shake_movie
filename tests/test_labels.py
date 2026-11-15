"""Tests for `LabelPlacer`, which picks the towns named on each frame.

These need no OpenGL: a flat 10 km square of ground seen straight down from
above is enough to check the label limits, overlaps, fades and words.
"""

import dataclasses
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from shake_movie import render
from shake_movie.camera import Shot
from shake_movie.scene import Scene, Town

SETTINGS = render.Settings(width=1920, height=1080, fps=30)
STEP = 1.0 / (render.LABEL_FADE * SETTINGS.fps)
"""How much a label's opacity changes each frame."""


@dataclasses.dataclass
class _Ground:
    """Just enough of a `Scene` for labels: flat ground, a domain and towns."""

    towns: list[Town]
    height: npt.NDArray[np.float32] = dataclasses.field(
        default_factory=lambda: np.zeros((11, 11), np.float32)
    )
    height_bounds: tuple[float, float, float, float] = (0.0, 0.0, 10e3, 10e3)
    domain: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array(
            [[2e3, 2e3], [8e3, 2e3], [8e3, 8e3], [2e3, 8e3]]
        )
    )

    height_at = Scene.height_at


def _scene(towns: list[Town], height: npt.ArrayLike | None = None) -> Scene:
    """A flat 10 km scene holding `towns`, or one whose terrain is `height`."""
    ground = _Ground(towns)
    if height is not None:
        ground.height = np.asarray(height, np.float32)
    return cast(Scene, ground)


def _projector(scene: Scene, distance: float = 8e3) -> render.Projector:
    """Look straight down on the middle of the scene, north up."""
    shot = Shot(easting=5e3, northing=5e3, distance=distance, pitch=90.0, heading=0.0)
    return render.Projector(*render.camera_pose(scene, shot, 1.0), SETTINGS)


def _town(name: str, population: int, easting: float, northing: float) -> Town:
    """A town."""
    return Town(name, population, easting, northing)


def _names(labels: list[render.Label]) -> set[str]:
    """Names of the labelled towns."""
    return {label.town.name for label in labels}


def _settle(
    placer: render.LabelPlacer,
    projector: render.Projector,
    limit: int,
    words: dict[str, str | None] | None = None,
) -> list[render.Label]:
    """Place labels until every fade has finished."""
    labels: list[render.Label] = []
    for _ in range(int(np.ceil(1 / STEP)) + 2):
        labels = placer.place(projector, 1.0, limit, words)
    return labels


def test_towns_on_one_pixel_give_one_label_for_the_largest() -> None:
    """Three towns stacked on the same spot overlap: only the most populous
    is named."""
    towns = [
        _town("Small", 2_000, 5e3, 5e3),
        _town("Large", 150_000, 5e3, 5e3),
        _town("Middle", 20_000, 5e3, 5e3),
    ]
    scene = _scene(towns)
    labels = render.LabelPlacer(scene, SETTINGS).place(_projector(scene), 1.0, 8)
    assert _names(labels) == {"Large"}
    assert labels[0].position == pytest.approx((960.0, 540.0))


def test_at_most_limit_labels_are_shown() -> None:
    """Of many well spread towns, only the `limit` most populous are named."""
    towns = [
        _town(f"T{k}", 1_000 * (k + 1), 3e3 + 1e3 * (k % 5), 4e3 + 2e3 * (k // 5))
        for k in range(10)
    ]
    scene = _scene(towns)
    projector = _projector(scene)
    # All ten fit on screen without overlapping when the limit allows it.
    assert len(render.LabelPlacer(scene, SETTINGS).place(projector, 1.0, 20)) == 10
    for limit in (render.MAX_LABELS, 3):
        labels = render.LabelPlacer(scene, SETTINGS).place(projector, 1.0, limit)
        assert _names(labels) == {f"T{k}" for k in range(10 - limit, 10)}


def test_labels_fade_in_and_out() -> None:
    """A label's opacity rises a step a frame to 1, then falls back to 0 once
    its town leaves the screen, never leaving [0, 1]."""
    town = _town("Town", 5_000, 5e3, 5e3)
    scene = _scene([town])
    placer = render.LabelPlacer(scene, SETTINGS)
    projector = _projector(scene)
    rising = [placer.place(projector, 1.0, 8)[0].alpha for _ in range(15)]
    frames = round(1 / STEP)
    expected = [min(1.0, STEP * (k + 1)) for k in range(15)]
    np.testing.assert_allclose(rising, expected)
    assert rising[frames - 1] == pytest.approx(1.0)
    # Move the town behind the camera: its label fades rather than vanishing.
    town.easting, town.northing = 1e6, 1e6
    falling = []
    for _ in range(15):
        labels = placer.place(projector, 1.0, 8)
        falling.append(labels[0].alpha if labels else 0.0)
    assert all(0.0 <= alpha <= 1.0 for alpha in rising + falling)
    assert falling[0] == pytest.approx(1.0 - STEP)
    assert np.all(np.diff(falling) <= 0.0)
    assert falling[-1] == 0.0


def test_showing_labels_keep_their_place() -> None:
    """A town already named keeps its label when a larger town appears on
    top of it."""
    small = _town("Small", 2_000, 5e3, 5e3)
    large = _town("Large", 500_000, 1e6, 1e6)
    scene = _scene([large, small])
    placer = render.LabelPlacer(scene, SETTINGS)
    projector = _projector(scene)
    assert _names(_settle(placer, projector, 8)) == {"Small"}
    large.easting, large.northing = 5e3, 5e3
    labels = placer.place(projector, 1.0, 8)
    assert _names(labels) == {"Small"}
    assert labels[0].alpha == 1.0


def test_final_map_labels_towns_in_the_domain_with_their_words() -> None:
    """Given words, only towns inside the domain are named, each with its
    word."""
    towns = [
        _town("Inside", 3_000, 4e3, 4e3),
        _town("Quiet", 2_000, 6e3, 6e3),
        _town("Outside", 300_000, 8.5e3, 5e3),
    ]
    scene = _scene(towns)
    # Far enough back to see past the domain's edge.
    projector = _projector(scene, distance=12e3)
    words: dict[str, str | None] = {
        "Inside": "Strong",
        "Quiet": None,
        "Outside": "Severe",
    }
    labels = render.LabelPlacer(scene, SETTINGS).place(
        projector, 1.0, render.MAX_FINAL_LABELS, words
    )
    assert {label.town.name: label.word for label in labels} == {
        "Inside": "Strong",
        "Quiet": None,
    }
    # Without words the town outside the domain is named too, with no word.
    labels = render.LabelPlacer(scene, SETTINGS).place(projector, 1.0, 8)
    assert {label.town.name: label.word for label in labels} == {
        "Inside": None,
        "Quiet": None,
        "Outside": None,
    }


def test_towns_in_the_margins_are_not_named() -> None:
    """Towns in the strips along the frame's edges, kept clear for the
    on-screen text, get no label."""
    scene = _scene([])
    projector = _projector(scene)
    # Metres of ground per pixel, looking straight down.
    across = projector(np.array([[5e3, 5e3, 60.0], [6e3, 5e3, 60.0]]))
    metres = 1e3 / (across[1, 0] - across[0, 0])

    def at(x: float, y: float) -> tuple[float, float]:
        """Easting and northing of the ground at pixel `(x, y)`."""
        return 5e3 + (x - 960.0) * metres, 5e3 - (y - 540.0) * metres

    scene.towns = [
        _town("Middle", 1_000, *at(960.0, 540.0)),
        _town("Left", 1_000_000, *at(40.0, 540.0)),
        _town("Right", 1_000_000, *at(1700.0, 700.0)),
        _town("Top", 1_000_000, *at(500.0, 150.0)),
        _town("Bottom", 1_000_000, *at(500.0, 1000.0)),
    ]
    labels = render.LabelPlacer(scene, SETTINGS).place(projector, 1.0, 8)
    assert _names(labels) == {"Middle"}


def _pitched(scene: Scene) -> render.Projector:
    """Look north at the middle of the scene from 1.5 km up, 2.6 km south."""
    shot = Shot(easting=5e3, northing=5e3, distance=3e3, pitch=30.0, heading=0.0)
    return render.Projector(*render.camera_pose(scene, shot, 1.0), SETTINGS)


def test_towns_behind_the_camera_are_not_named() -> None:
    """A town behind the camera gets no label, even when nothing else
    competes for one."""
    towns = [_town("Ahead", 1_000, 5e3, 6e3), _town("Behind", 1_000_000, 5e3, 500.0)]
    scene = _scene(towns)
    labels = render.LabelPlacer(scene, SETTINGS).place(_pitched(scene), 1.0, 8)
    assert _names(labels) == {"Ahead"}


def test_towns_hidden_by_terrain_are_not_named() -> None:
    """A ridge between the camera and a town hides its label."""
    town = _town("Town", 1_000, 5e3, 6e3)
    flat = _scene([town])
    assert _names(render.LabelPlacer(flat, SETTINGS).place(_pitched(flat), 1.0, 8)) == {
        "Town"
    }
    # A 2 km high ridge running east-west at 4 km north, rows from the north.
    height = np.zeros((11, 11))
    height[6] = 2000.0
    ridged = _scene([town], height)
    assert render.LabelPlacer(ridged, SETTINGS).place(_pitched(ridged), 1.0, 8) == []
