"""Tests for planning a movie from an SW4 run's output directory."""

import logging
from pathlib import Path

import fake_sw4
import numpy as np
import numpy.typing as npt
import psutil
import pytest
from conftest import write_measures
from matplotlib.axes import Axes
from matplotlib.text import Text

from shake_movie import cli, palette, render
from shake_movie.cli import plan_movie
from shake_movie.sw4_images import GridGeometry, ImageSequence, read_image

SETTINGS = render.Settings(width=320, height=180, fps=4)
"""The small frame the tests plan for."""


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


@pytest.mark.parametrize(
    ("affinity", "logical", "physical", "cores"),
    [(32, 192, 96, 16), (16, 16, 16, 16), (1, 192, 96, 1)],
)
def test_workers_default_to_physical_cores(
    monkeypatch: pytest.MonkeyPatch,
    affinity: int,
    logical: int,
    physical: int,
    cores: int,
) -> None:
    """A Slurm job's hyperthreads count once per core, and never as none."""
    counts = {True: logical, False: physical}
    monkeypatch.setattr(
        psutil.Process, "cpu_affinity", lambda self: list(range(affinity))
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: counts[logical])
    assert cli._cores() == cores


def _plan(sw4_run: dict[str, Path], measures: Path | None = None) -> cli.MoviePlan:
    """Plan the fixture run's movie, small and offline."""
    return plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        SETTINGS,
        None,
        None,
        sw4_run["coastline"],
        online=False,
        measures=measures,
    )


def _texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every text the HUD draws."""
    texts: list[str] = []
    draw = Axes.text

    def _record(self: Axes, x: float, y: float, s: str, *args, **kwargs) -> Text:  # noqa: ANN002, ANN003
        """Note the text, then draw it."""
        texts.append(s)
        return draw(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", _record)
    return texts


def test_waves_are_relative_to_the_strongest(sw4_run: dict[str, Path]) -> None:
    """The waves are scaled by the strongest ground velocity in the domain."""
    plan = _plan(sw4_run)
    stride = plan.scene.wave_stride
    moving = ImageSequence(plan.velocity).maximum()[::stride, ::stride]
    assert plan.reference == pytest.approx(float(moving[plan.scene.wave_inside].max()))


def test_waves_reach_the_broadband_peak(
    sw4_run: dict[str, Path], geometry: GridGeometry, tmp_path: Path
) -> None:
    """The waves' strongest moment is coloured as the strongest broadband
    shaking, however weak the low-frequency motion itself is."""
    measures = write_measures(
        tmp_path / "im.h5", geometry, lambda x, y: np.full_like(x, 3.0)
    )
    plan = _plan(sw4_run, measures)
    sequence = ImageSequence(plan.velocity)
    stride, scale = plan.scene.wave_stride, plan.end_card.scale
    levels = np.stack(
        [
            render.waves(sequence, t, stride, plan.reference, scale)[
                plan.scene.wave_inside
            ]
            for t in sequence.times
        ]
    )
    assert scale.high == pytest.approx(3.0, rel=1e-3)
    assert levels.max() == pytest.approx(1.0)
    assert levels.min() >= 0.0


def test_end_card_falls_back_to_low_frequency(
    sw4_run: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """Without intensity measures the final map is SW4's `hmax`, as before."""
    with caplog.at_level(logging.INFO, logger="shake_movie"):
        plan = _plan(sw4_run)
    stride = plan.scene.wave_stride
    (hmax,) = sw4_run["images"].glob("surf_hmax.*")
    expected = read_image(hmax).values[::stride, ::stride] * render.CMS_PER_MS
    land = plan.scene.wave_opacity >= 1.0
    np.testing.assert_allclose(plan.end_card.pgv[land], expected[land], rtol=1e-6)
    assert np.isnan(plan.end_card.pgv[~land]).all()
    assert plan.end_card.scale.high == pytest.approx(float(expected[land].max()))
    assert not plan.end_card.weak
    assert "low-frequency" in caplog.text


def test_end_card_uses_broadband_pgv(
    sw4_run: dict[str, Path], geometry: GridGeometry, tmp_path: Path
) -> None:
    """With intensity measures the final map is broadband PGV, the colour
    scale tops out at its strongest on land, and nothing shows beyond the
    stations."""

    def _pgv(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray:
        """Broadband PGV dying away from the source at (22, 17) km (cm/s)."""
        return 20.0 * 10.0 ** (-np.hypot(x - 22e3, y - 17e3) / 20e3)

    measures = write_measures(tmp_path / "im.h5", geometry, _pgv)
    plan = _plan(sw4_run, measures)
    pgv = plan.end_card.pgv
    stride = plan.scene.wave_stride
    near = (16 // stride, 17 // stride)
    assert pgv[near] == pytest.approx(
        float(_pgv(np.array(16e3), np.array(17e3))), rel=0.2
    )
    assert np.isnan(pgv[2 // stride, 25 // stride])
    sea = plan.scene.wave_opacity < 1.0
    assert sea.any() and np.isnan(pgv[sea]).all()
    assert plan.end_card.scale.high == pytest.approx(float(np.nanmax(pgv)))
    assert not plan.end_card.weak


def test_weak_event_end_card_says_so(
    sw4_run: dict[str, Path],
    geometry: GridGeometry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no shaking on land reaches "Weak", the final map says so; the
    legend is the same ground velocity scale throughout, without MMI words."""
    measures = write_measures(
        tmp_path / "im.h5", geometry, lambda x, y: np.full_like(x, 0.05)
    )
    plan = _plan(sw4_run, measures)
    assert plan.end_card.weak
    assert not any(plan.end_card.words.values())
    texts = _texts(monkeypatch)
    image = np.zeros((180, 320, 3), np.uint8)
    final = plan.frames[-1]
    middle = next(frame for frame in plan.frames if frame.kind == "simulation")
    words = {word for _, word in palette.SHAKING_WORDS}
    scale = plan.end_card.scale
    ticks = [
        f"{v:g}"
        for v in render.velocity_ticks(scale, 620.0 * 320 / 1920, 14.0 * 320 / 1920)
    ]
    render.draw_hud(
        image,
        plan.scene,
        middle,
        [],
        [],
        np.full(2, np.nan),
        SETTINGS,
        True,
        scale=scale,
    )
    during = list(texts)
    texts.clear()
    render.draw_hud(
        image,
        plan.scene,
        final,
        [],
        [],
        np.full(2, np.nan),
        SETTINGS,
        True,
        scale=scale,
    )
    assert render.WEAK_NOTE in texts and "Strongest shaking felt" not in texts
    for drawn in (during, texts):
        assert render.VELOCITY_TITLE in drawn and not words & set(drawn)
        assert set(ticks) <= set(drawn)
