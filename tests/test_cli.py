"""Tests for planning a movie from an SW4 run's output directory."""

import logging
import os
from pathlib import Path

import fake_sw4
import h5py
import numpy as np
import numpy.typing as npt
import psutil
import pytest
import typer
from conftest import write_measures
from matplotlib.axes import Axes
from matplotlib.text import Text

from shake_movie import cli, palette, render, sw4_images
from shake_movie.cli import plan_movie
from shake_movie.realisation import Realisation
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


@pytest.mark.parametrize(
    ("available", "logical", "physical", "cores"),
    [(192, 192, 96, 96), (8, 8, 8, 8), (None, 2, 1, 1)],
)
def test_workers_without_affinity_count_usable_cores(
    monkeypatch: pytest.MonkeyPatch,
    available: int | None,
    logical: int,
    physical: int,
    cores: int,
) -> None:
    """Without an affinity mask (macOS), the usable CPUs count once per core."""
    counts = {True: logical, False: physical}
    monkeypatch.delattr(psutil.Process, "cpu_affinity", raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: available)
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


def test_compressed_run_plans_as_uncompressed(
    sw4_run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose images SW4 wrote compressed (lossless ZFP here) plans the
    same movie as the same run uncompressed, with its peak taken in workers."""
    plain = _plan(sw4_run)
    monkeypatch.setattr(sw4_images, "MAXIMUM_IMAGES_PER_WORKER", 1)
    for path in sorted(sw4_run["images"].glob("*.sw4img.h5")):
        fake_sw4.compress_image(path, tmp_path / path.name)
        (tmp_path / path.name).replace(path)
    compressed = plan_movie(
        sw4_run["sw4_input"],
        sw4_run["realisation"],
        SETTINGS,
        None,
        None,
        sw4_run["coastline"],
        online=False,
        workers=2,
    )
    with h5py.File(compressed.velocity[0], "r") as handle:
        assert "patches" not in handle
    assert compressed.velocity == plain.velocity
    assert compressed.velocity_times == plain.velocity_times
    assert compressed.reference == plain.reference
    assert compressed.frames == plain.frames
    np.testing.assert_array_equal(compressed.end_card.pgv, plain.end_card.pgv)
    np.testing.assert_array_equal(
        compressed.scene.mesh_elevation, plain.scene.mesh_elevation
    )


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


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["a.in", "b.json", "--branding"], f"--branding={render.DEFAULT_BRANDING}"),
        (["--branding", "a.in", "b.json"], f"--branding={render.DEFAULT_BRANDING}"),
        (["--branding", "--offline"], f"--branding={render.DEFAULT_BRANDING}"),
        (["--branding", "logo.PNG"], "--branding"),
        (["--branding=logo.png"], "--branding=logo.png"),
    ],
)
def test_bare_branding_takes_the_default_header(args: list[str], expected: str) -> None:
    """A bare `--branding` names SeismicNow's header; a given image is kept."""
    expanded = cli.expand_branding(args)
    assert len(expanded) == len(args)
    assert expected in expanded
    assert [arg for arg in expanded if not arg.startswith("--branding")] == [
        arg for arg in args if not arg.startswith("--branding")
    ]


def test_default_branding_ships_with_the_package() -> None:
    """SeismicNow's header is installed with shake-movie and reads as RGBA."""
    image = render.load_branding(render.DEFAULT_BRANDING)
    assert image.ndim == 3 and image.shape[2] == 4
    assert 0.0 <= image.min() and image.max() <= 1.0


def test_branding_looks_at_the_argument_after_it() -> None:
    """Whether `--branding` takes a value depends on what follows it, not
    what comes before it."""
    assert cli.expand_branding(["logo.png", "--branding", "a.in"]) == [
        "logo.png",
        f"--branding={render.DEFAULT_BRANDING}",
        "a.in",
    ]
    assert cli.expand_branding(["a.in", "--branding", "logo.jpeg"]) == [
        "a.in",
        "--branding",
        "logo.jpeg",
    ]


def test_run_expands_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """`shake-movie` hands Typer its arguments, without the program name, with
    a bare `--branding` expanded."""
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "app", calls.append)
    monkeypatch.setattr(
        "sys.argv", ["shake-movie", "sw4.in", "realisation.json", "--branding"]
    )
    cli.run()
    assert calls == [
        ["sw4.in", "realisation.json", f"--branding={render.DEFAULT_BRANDING}"]
    ]


def test_image_prefix_picks_the_surface_command() -> None:
    """Only a mode's `z=0` command names its surface images; commands at depth
    or without a depth are skipped, and a run with no images has none."""
    commands = {
        "imagehdf5": [
            {"mode": "mag", "file": "no_depth"},
            {"mode": "mag", "z": "1000", "file": "deep_mag"},
            {"mode": "hmax", "z": "0", "file": "surf_hmax"},
            {"mode": "mag", "z": "0", "file": "surf_mag"},
        ]
    }
    assert cli._image_prefix(commands, "mag") == "surf_mag"
    assert cli._image_prefix(commands, "hmax") == "surf_hmax"
    assert cli._image_prefix(commands, "topo") is None
    assert cli._image_prefix({"imagehdf5": commands["imagehdf5"][:2]}, "mag") is None
    assert cli._image_prefix({}, "mag") is None


def test_matching_keeps_images_on_the_terrain_grid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Images on another grid are dropped with a warning counting them; when
    every image matches, nothing is said."""
    good = [
        fake_sw4.write_image(tmp_path, "surf_mag", "mag", c, c, np.ones((3, 4)), 1e3)
        for c in (1, 2, 3)
    ]
    stale = [
        fake_sw4.write_image(tmp_path, "surf_mag", "mag", c, c, np.ones((7, 5)), 1e3)
        for c in (4, 5)
    ]
    with caplog.at_level(logging.WARNING, logger="shake_movie"):
        assert cli._matching(good, (3, 4)) == good
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger="shake_movie"):
        assert cli._matching(good + stale, (3, 4)) == good
    assert [record.getMessage() for record in caplog.records] == [
        "Skipping 2 images that don't match the 3 x 4 terrain, left by another SW4 run"
    ]


def test_cores_round_down_to_whole_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A share of a core isn't a worker: the count is a whole number."""
    counts = {True: 4, False: 2}
    monkeypatch.setattr(psutil.Process, "cpu_affinity", lambda self: [0, 1, 2])
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: counts[logical])
    cores = cli._cores()
    assert cores == 1 and isinstance(cores, int)


def test_cores_with_nothing_known_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither psutil nor the OS can count the CPUs, one worker runs."""
    monkeypatch.delattr(psutil.Process, "cpu_affinity", raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: None)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: None)
    assert cli._cores() == 1


def _realisation(coastline: Path | None) -> Realisation:
    """A realisation naming the given velocity model coastline."""
    return Realisation("test_run", 6.0, np.zeros((4, 2)), coastline)


@pytest.mark.parametrize("name", [None, "missing.gmt"])
def test_coastline_downloads_linz_without_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str | None
) -> None:
    """Without a velocity model coastline on disk, LINZ's is downloaded."""
    linz = tmp_path / "linz.gpkg"
    monkeypatch.setattr(cli.scene, "download_coastline", lambda: linz)
    model = None if name is None else tmp_path / name
    assert cli._coastline(None, _realisation(model)) == linz


def test_coastline_prefers_given_then_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A given coastline wins, then the velocity model's, and nothing is
    downloaded."""

    def _no_download() -> Path:
        """Fail: the test has a coastline to hand."""
        raise AssertionError("downloaded LINZ's coastline")

    monkeypatch.setattr(cli.scene, "download_coastline", _no_download)
    model = tmp_path / "model.gmt"
    model.write_text("")
    given = tmp_path / "given.gmt"
    assert cli._coastline(given, _realisation(model)) == given
    assert cli._coastline(None, _realisation(model)) == model


def test_plan_needs_velocity_images(sw4_run: dict[str, Path], tmp_path: Path) -> None:
    """A run without its surface images can't make a movie."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(typer.BadParameter, match="surface `topo` image"):
        plan_movie(
            sw4_run["sw4_input"],
            sw4_run["realisation"],
            SETTINGS,
            empty,
            None,
            sw4_run["coastline"],
            online=False,
        )


def test_still_run_keeps_a_usable_scale(sw4_run: dict[str, Path]) -> None:
    """A run where the ground never moves still plans a movie, scaled to
    1 m/s and 1 cm/s rather than to zero."""
    for mode in ("mag", "hmax"):
        for path in sw4_run["images"].glob(f"surf_{mode}.*"):
            with h5py.File(path, "r+") as handle:
                handle["patches"][...] = 0.0
    plan = _plan(sw4_run)
    assert plan.reference == 1.0
    assert plan.end_card.scale.high == 1.0
    land = plan.scene.wave_opacity >= 1.0
    np.testing.assert_array_equal(plan.end_card.pgv[land], 0.0)
