"""Tests for the on-screen text drawn over each frame."""

import dataclasses
from pathlib import Path

import numpy as np
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

    def draw(epicentre: np.ndarray) -> np.ndarray:
        return render.draw_hud(
            image,
            plan.scene,
            frame,
            [],
            [],
            epicentre,
            settings,
            scale=plan.end_card.scale,
        )

    bare = draw(np.full(2, np.nan))
    starred = draw(np.array([settings.width / 2, settings.height * 0.45]))
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
