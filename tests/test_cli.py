"""Tests for planning a movie from an SW4 run's output directory."""

from pathlib import Path

import fake_sw4
import numpy as np

from shake_movie import render
from shake_movie.cli import plan_movie


def test_images_from_another_run_are_skipped(sw4_run: dict[str, Path]) -> None:
    """Images left by an earlier SW4 attempt on another grid are ignored."""
    images = sw4_run["images"]
    stale = [
        fake_sw4.write_image(images, prefix, mode, 6, 1.5, np.ones((7, 5)), 1000.0)
        for prefix, mode in [("surf_mag", "mag"), ("surf_hmax", "hmax")]
    ]
    plan = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        render.Settings(width=320, height=180, fps=4),
        None,
        None,
        sw4_run["coastline"],
        online=False,
    )
    assert len(plan.velocity) == 3
    assert not set(stale) & set(plan.velocity)
