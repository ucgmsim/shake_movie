"""Tests for the on-screen text drawn over each frame."""

import dataclasses
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from shake_movie import palette, render
from shake_movie.cli import plan_movie

LONG_TITLE = "Wednesday 30th September Mw 4.0 earthquake"
LONG_PLACE = "5 km south-west of Paraparaumu Beach, Kāpiti Coast District"


def test_fit_shrinks_only_text_that_is_too_wide() -> None:
    """Wide text shrinks to the width; text that fits keeps its size."""
    figure = Figure(figsize=(19.2, 10.8), dpi=100)
    FigureCanvasAgg(figure)
    wide = figure.text(0.5, 0.5, LONG_TITLE, fontsize=64, fontweight="bold")
    narrow = figure.text(0.5, 0.2, "Mw 4.0", fontsize=64)
    render._fit(wide, 1000.0)
    render._fit(narrow, 1000.0)
    assert wide.get_window_extent().width == pytest.approx(1000.0, rel=0.02)
    assert narrow.get_fontsize() == 64


def test_title_card_stays_on_screen(sw4_run: dict[str, Path]) -> None:
    """A long title and place stay inside the frame on the title card."""
    settings = render.Settings(width=640, height=360, fps=4)
    plan = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        settings,
        None,
        None,
        sw4_run["coastline"],
        online=False,
    )
    scene = dataclasses.replace(plan.scene, title=LONG_TITLE, location=LONG_PLACE)
    frame = dataclasses.replace(plan.frames[0], kind="intro", title=1.0)
    image = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
    hud = render.draw_hud(
        image,
        scene,
        frame,
        [],
        [],
        np.full(2, np.nan),
        settings,
        scale=plan.end_card.scale,
    )
    # Text is drawn in light colours over the black frame: no light pixel
    # may touch the frame's left or right edge.
    edges = hud[:, [0, 1, -2, -1]].max(axis=2)
    assert edges.max() < 120


def test_title_card_covers_the_epicentre(sw4_run: dict[str, Path]) -> None:
    """The epicentre star sits under the title card, not over it, and the
    map still shows through around the text."""
    settings = render.Settings(width=640, height=360, fps=4)
    plan = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        settings,
        None,
        None,
        sw4_run["coastline"],
        online=False,
    )
    frame = dataclasses.replace(plan.frames[0], kind="intro", title=1.0)
    image = np.full((settings.height, settings.width, 3), 128, dtype=np.uint8)

    bare, starred = (
        render.draw_hud(
            image,
            plan.scene,
            frame,
            [],
            [],
            epicentre,
            settings,
            scale=plan.end_card.scale,
        )
        for epicentre in (
            np.full(2, np.nan),
            np.array([settings.width / 2, settings.height * 0.45]),
        )
    )
    title = (bare.min(axis=2) > 240)[140:180, 280:360]
    assert title.any()
    assert np.array_equal(
        starred[140:180, 280:360][title], bare[140:180, 280:360][title]
    )
    assert np.array_equal(bare[settings.height // 2, 5], [128, 128, 128])


def test_velocity_ticks_are_round_and_fit_the_bar() -> None:
    """Ticks fall on 1, 2 and 5 of each decade inside the scale, and thin out
    on a short bar while keeping the largest."""
    scale = palette.VelocityScale(0.18)
    wide = render.velocity_ticks(scale, 620.0, 14.0)
    assert wide == [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    assert all(scale.low <= v <= scale.high for v in wide)
    narrow = render.velocity_ticks(scale, 120.0, 14.0)
    assert len(narrow) < len(wide) and narrow[-1] == wide[-1]
    assert set(narrow) <= set(wide)


def test_inset_map_sits_in_the_bottom_right_below_the_compass(
    sw4_run: dict[str, Path],
) -> None:
    """The inset shows the domain in red at the bottom right, and the north
    arrow sits above it."""
    settings = render.Settings(width=640, height=360, fps=4)
    plan = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        settings,
        None,
        None,
        sw4_run["coastline"],
        online=False,
    )
    # A domain large enough to see at the inset's scale, round the hypocentre.
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 80e3
    scene = dataclasses.replace(
        plan.scene, domain=plan.scene.rupture.hypocentre + corners
    )
    image = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
    hud = render.draw_hud(
        image,
        scene,
        plan.frames[len(plan.frames) // 2],
        [],
        [],
        np.full(2, np.nan),
        settings,
        scale=plan.end_card.scale,
    ).astype(int)
    s = settings.width / 1920
    left, top, width, height = render.inset_box(s, settings.width, settings.height)
    assert left + width < settings.width and top + height < settings.height
    # Red right of the legend (whose colour bar runs through red) is the domain.
    half = settings.width // 2
    right = hud[:, half:]
    red = (right[..., 0] > 150) & (right[..., 1] < 110) & (right[..., 2] < 110)
    rows, columns = np.nonzero(red)
    assert rows.size > 0
    assert rows.min() >= top and columns.min() + half >= left
    # The north arrow (white) is drawn above the inset, not over it.
    arrow_rows, _ = np.nonzero(hud[:, int(left) :].min(axis=2) > 230)
    assert arrow_rows.size > 0
    assert arrow_rows.max() < top


def _caption_row(hud: npt.NDArray[np.int_], columns: slice) -> float:
    """Middle row of the light caption text in some columns."""
    rows, _ = np.nonzero(hud[:, columns].min(axis=2) > 200)
    return float(np.median(rows))


@pytest.mark.parametrize("branding", [None, render.DEFAULT_BRANDING])
def test_branding_sits_above_the_caption(
    sw4_run: dict[str, Path], branding: Path | None
) -> None:
    """With branding, the header takes the top right and the caption drops
    beneath it; without, the caption stays at the top."""
    settings = render.Settings(width=1920, height=1080, fps=4, branding=branding)
    plan = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        settings,
        None,
        None,
        sw4_run["coastline"],
        online=False,
    )
    image = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
    hud = render.draw_hud(
        image,
        plan.scene,
        plan.frames[len(plan.frames) // 2],
        [],
        [],
        np.full(2, np.nan),
        settings,
        scale=plan.end_card.scale,
    ).astype(int)
    s = settings.width / 1920
    # The caption's left half, clear of the header's own light text.
    caption = slice(int(settings.width * 0.75), int(settings.width * 0.82))
    top = hud[: int(160 * s)]
    if branding is None:
        assert _caption_row(top, caption) == pytest.approx(46 * s, abs=3)
        return
    header = render.load_branding(branding)
    left, top_edge, width, height = render.branding_box(header, s, settings.width)
    assert left > render.HUD_TEXT_WIDTH * settings.width
    # The header is drawn over the map with no card behind it: the black frame
    # shows through at its top left corner, and its logo is red.
    box = hud[int(top_edge) : int(top_edge + height), int(left) : int(left + width)]
    assert box[1, 1].max() < 20
    assert (box[..., 0] - box[..., 1] > 80).any()
    bottom = int(np.ceil(top_edge + height))
    below = hud[bottom : bottom + int(40 * s), int(0.7 * settings.width) :]
    assert (below.min(axis=2) > 200).any()


def test_wide_branding_shrinks_beside_the_title() -> None:
    """An image too wide for the room beside the title is scaled down to fit."""
    s, w = 1.0, 1920.0
    left, _, width, height = render.branding_box(
        np.zeros((10, 400, 4), np.float32), s, w
    )
    assert left >= render.HUD_TEXT_WIDTH * w
    assert height < render.BRANDING_HEIGHT
    assert width / height == pytest.approx(40.0)


def test_slip_spreads_then_gives_way_to_the_planes(sw4_run: dict[str, Path]) -> None:
    """The fault's cells colour in as it slips, and fade from the final map."""
    settings = render.Settings(width=640, height=360, fps=4)
    plan = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        settings,
        None,
        None,
        sw4_run["coastline"],
        online=False,
    )
    scene = plan.scene
    frame = next(f for f in plan.frames if f.kind == "simulation")
    shot = dataclasses.replace(frame.shot, pitch=89.0)
    projector = render.Projector(
        *render.camera_pose(scene, shot, settings.exaggeration), settings
    )

    def _slip(time: float) -> render.SlipCells:
        cells = render.fault_slip(scene, projector, time, settings.exaggeration)
        assert cells is not None
        return cells

    assert len(_slip(-1.0).quads) == 0
    early, late = _slip(0.5), _slip(10.0)
    assert 0 < len(early.quads) < len(late.quads)
    assert np.isfinite(late.quads).all()
    # The most-slipped cell is the most opaque.
    assert late.colours[:, 3].max() == pytest.approx(render.SLIP_OPACITY)

    none = render.SlipCells(np.empty((0, 4, 2)), np.empty((0, 4)))

    def _hud(peak: float, slip: render.SlipCells) -> npt.NDArray[np.uint8]:
        """Draw the frame at `peak` of the way to the final map."""
        return render.draw_hud(
            np.zeros((settings.height, settings.width, 3), dtype=np.uint8),
            scene,
            dataclasses.replace(frame, shot=shot, peak=peak),
            [],
            [],
            np.full(2, np.nan),
            settings,
            scale=plan.end_card.scale,
            planes=render.fault_planes(scene, projector, settings.exaggeration),
            slip=slip,
        )

    def _changed(peak: float) -> int:
        """Count pixels the slip changes."""
        return int((_hud(peak, late) != _hud(peak, none)).any(axis=-1).sum())

    assert _changed(0.0) > 50
    assert _changed(1.0) == 0
