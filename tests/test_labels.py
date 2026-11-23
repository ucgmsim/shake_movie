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


@pytest.mark.parametrize(
    ("population", "tier"),
    [(0, 0), (9_999, 0), (10_000, 1), (99_999, 1), (100_000, 2), (2_000_000, 2)],
)
def test_label_tiers_step_up_at_ten_and_a_hundred_thousand(
    population: int, tier: int
) -> None:
    """Towns rank 0 below 10 000 people, 1 from 10 000 and 2 from 100 000."""
    assert render._tier(population) == tier


@dataclasses.dataclass
class _FixedScreen:
    """A stand-in projection that puts every town at a chosen pixel, seen
    from high above the scene so the terrain hides none of them."""

    pixels: list[tuple[float, float]]
    position: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([5e3, 5e3, 50e3])
    )

    def __call__(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The chosen pixels, one per town."""
        assert len(points) == len(self.pixels)
        return np.array(self.pixels, dtype=np.float64)


def _named_at(settings: render.Settings, pixel: tuple[float, float]) -> bool:
    """Whether a lone town drawn at `pixel` gets a label."""
    scene = _scene([_town("Town", 1_000, 5e3, 5e3)])
    projector = cast(render.Projector, _FixedScreen([pixel]))
    labels = render.LabelPlacer(scene, settings).place(projector, 1.0, 8)
    return bool(labels)


def test_left_margin_is_sixty_pixels_at_full_hd() -> None:
    """At 1920 pixels wide, a town is named only strictly right of x = 60."""
    for x, named in ((59.99, False), (60.0, False), (60.01, True), (60.5, True)):
        assert _named_at(SETTINGS, (x, 540.0)) == named, x


def test_margins_scale_with_the_frame_width() -> None:
    """Margins shrink with the frame: at 960 x 540 the left margin is 30
    pixels, the right 130, the top 100 and the bottom 75."""
    half = render.Settings(width=960, height=540, fps=30)
    for pixel, named in (
        ((29.9, 270.0), False),
        ((30.1, 270.0), True),
        ((50.0, 270.0), True),
        ((829.9, 270.0), True),
        ((830.1, 270.0), False),
        ((400.0, 99.9), False),
        ((400.0, 100.1), True),
        ((400.0, 464.9), True),
        ((400.0, 465.1), False),
    ):
        assert _named_at(half, pixel) == named, pixel


def test_labels_hang_sixty_metres_above_the_exaggerated_ground() -> None:
    """A label's dot is where the ground 60 m above the town, after
    exaggeration, lands on screen."""
    # Ground rising 10 m per 100 m east, so it differs from point to point.
    height = np.tile(np.arange(11) * 10.0, (11, 1))
    town = _town("Town", 1_000, 5.5e3, 5e3)
    scene = _scene([town], height)
    shot = Shot(easting=5e3, northing=5e3, distance=3e3, pitch=50.0, heading=0.0)
    projector = render.Projector(*render.camera_pose(scene, shot, 2.0), SETTINGS)
    labels = render.LabelPlacer(scene, SETTINGS).place(projector, 2.0, 8)
    assert len(labels) == 1
    # The ground at the town is 55 m: exaggerated twice, plus 60 m.
    expected = projector(np.array([[5.5e3, 5e3, 55.0 * 2.0 + 60.0]]))[0]
    np.testing.assert_allclose(labels[0].position, expected, rtol=0, atol=1e-9)
    # A metre higher or lower moves the dot.
    for z in (169.0, 171.0):
        moved = projector(np.array([[5.5e3, 5e3, z]]))[0]
        assert np.abs(moved - expected).max() > 1e-3


def test_towns_shown_on_the_final_map_keep_their_place_after_it() -> None:
    """Towns outside the domain, never considered on the final map, are
    newcomers afterwards: a larger one overlapping a town already showing
    does not take its place."""
    inside = _town("Inside", 2_000, 7.95e3, 5e3)
    outside = _town("Outside", 500_000, 8.05e3, 5e3)
    scene = _scene([outside, inside])
    placer = render.LabelPlacer(scene, SETTINGS)
    projector = _projector(scene, distance=12e3)
    words: dict[str, str | None] = {"Inside": "Weak", "Outside": "Strong"}
    assert _names(_settle(placer, projector, 8, words)) == {"Inside"}
    labels = placer.place(projector, 1.0, 8)
    assert _names(labels) == {"Inside"}
    assert labels[0].alpha == 1.0
