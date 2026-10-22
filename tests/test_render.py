"""Rendering tests. These need ffmpeg and an OpenGL driver (EGL or OSMesa);
CI installs Mesa's software rasteriser, so no GPU is needed."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import write_measures
from typer.testing import CliRunner

from shake_movie import camera, render
from shake_movie.cli import app
from shake_movie.sw4_images import GridGeometry


def _can_render() -> bool:
    """Whether VTK can open an off-screen window here."""
    probe = "import pyvista as pv; p = pv.Plotter(off_screen=True, window_size=[64, 64]); p.add_mesh(pv.Sphere()); p.screenshot(return_img=True)"
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                timeout=120,
                check=False,
            ).returncode
            == 0
        )
    except subprocess.TimeoutExpired:
        return False


pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        not _can_render(), reason="no OpenGL driver for off-screen rendering"
    ),
]


def test_projection_matches_vtk() -> None:
    """Labels land where VTK draws the ground they name."""
    import pyvista as pv
    import vtk

    settings = render.Settings(width=640, height=360)
    position = np.array([1.55e6, 5.10e6, 80e3])
    focal_point = np.array([1.56e6, 5.20e6, 500.0])
    up = np.array([0.0, 0.0, 1.0])
    projector = render.Projector(position, focal_point, up, settings)
    plotter = pv.Plotter(off_screen=True, window_size=[settings.width, settings.height])
    plotter.add_mesh(pv.Sphere(center=focal_point, radius=10.0))
    plotter.camera.position, plotter.camera.focal_point, plotter.camera.up = (
        tuple(position),
        tuple(focal_point),
        tuple(up),
    )
    plotter.camera.view_angle = camera.VERTICAL_FOV
    plotter.render()
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    points = focal_point + np.array([[0, 0, 0], [20e3, 5e3, 0], [-15e3, -10e3, 1e3]])
    screen = projector(points)
    for k, point in enumerate(points):
        coordinate.SetValue(*map(float, point))
        vx, vy = coordinate.GetComputedDoubleDisplayValue(plotter.renderer)
        assert float(screen[k, 0]) == pytest.approx(vx, abs=0.5)
        assert float(screen[k, 1]) == pytest.approx(settings.height - vy, abs=0.5)
    plotter.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_cli_renders_a_movie(
    sw4_run: dict[str, Path], geometry: GridGeometry, tmp_path: Path
) -> None:
    """A small movie renders end to end from SW4's own output files and the
    broadband intensity measures."""
    output = tmp_path / "movie.mp4"
    measures = write_measures(
        tmp_path / "im.h5", geometry, lambda x, y: 10.0 ** (1 - x / 20e3)
    )
    result = CliRunner().invoke(
        app,
        [
            str(sw4_run["sw4_input"]),
            str(sw4_run["realisation"]),
            "--output",
            str(output),
            "--width",
            "320",
            "--height",
            "180",
            "--fps",
            "4",
            "--workers",
            "2",
            "--offline",
            "--intensity-measures",
            str(measures),
        ],
    )
    assert result.exit_code == 0, result.output
    frames = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(frames.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (320, 180)
    seconds = camera.INTRO + camera.DIVE + 4.0 + camera.PULL_UP + camera.HOLD
    assert int(stream["nb_read_frames"]) == pytest.approx(seconds * 4, abs=2)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["title"] == "Mw 6.0 earthquake"
    assert metadata["time"] is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_segment_renders_in_process(sw4_run: dict[str, Path], tmp_path: Path) -> None:
    """One worker's segment renders every kind of frame (in-process, so the
    renderer is measured by coverage)."""
    from shake_movie import cli

    settings = render.Settings(width=320, height=180, fps=4)
    plan = cli.plan_movie(
        sw4_run["sw4_input"], sw4_run["realisation"], settings, online=False
    )
    kinds = [frame.kind for frame in plan.frames]
    picks = [
        kinds.index("intro"),
        kinds.index("dive"),
        kinds.index("simulation") + 6,
        kinds.index("pull_up") + 1,
        len(kinds) - 1,
    ]
    frames = [plan.frames[i] for i in picks]
    job = render.Job(
        plan.scene,
        frames,
        0,
        len(frames),
        plan.velocity,
        plan.reference,
        plan.end_card,
        settings,
        tmp_path / "segment.mp4",
    )
    assert render.render_segment(job).stat().st_size > 0
    assert plan.metadata()["title"] == "Mw 6.0 earthquake"


def test_fault_planes_show_from_above(sw4_run: dict[str, Path]) -> None:
    """Seen from straight above, each plane's whole outline is in view."""
    from shake_movie import cli

    settings = render.Settings(width=320, height=180, fps=4)
    plan = cli.plan_movie(
        sw4_run["sw4_input"], sw4_run["realisation"], settings, online=False
    )
    shot = plan.frames[-1].shot
    projector = render.Projector(
        *render.camera_pose(plan.scene, shot, settings.exaggeration), settings
    )
    outlines = render.fault_planes(plan.scene, projector, settings.exaggeration)
    assert [len(outline) for outline in outlines] == [
        len(plane) for plane in plan.scene.rupture.planes
    ]
    assert all(np.isfinite(outline).all() for outline in outlines)
