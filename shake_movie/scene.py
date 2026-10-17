"""Everything drawn in a shake movie, prepared once before rendering.

The terrain is SW4's own `topo` image, on SW4's (rotated) grid. Land and sea
come from a coastline (the nzcvm one the velocity model was built with, or
LINZ's), not from elevation: NZCVM stores the sea as 0 m, the same height as much of the
Canterbury Plains. Beyond the SW4 grid the map continues as flat context, so
the coastline still frames the simulation.
"""

import csv
import dataclasses
import gzip
import os
import urllib.request
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import numpy.typing as npt
import shapely
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from pyproj import Transformer
from scipy import ndimage

from shake_movie import palette
from shake_movie.sw4_images import NZTM, GridGeometry

LINZ_COASTLINE_URL = "https://www.dropbox.com/scl/fi/zkohh794y0s2189t7b1hi/NZ.gmt?rlkey=02011f4morc4toutt9nzojrw1&st=vpz2ri8x&dl=1"
"""LINZ's NZ Coastlines and Islands Polygons (Topo 1:150k), CC BY 3.0 NZ, as
GMT rings. The same copy `ucgmsim/workflow` reads its coastline from."""

PLACES = "nz_places.csv"
"""The vendored town list (OpenStreetMap, see the file's header)."""

TEXTURE_MAX_PIXELS = 4096
"""Longest side of the terrain texture; GPUs and llvmpipe accept 8192+."""
WAVE_MAX_PIXELS = 1024
"""Longest side of the per-frame shaking texture."""
MESH_SPACING = 640.0
"""Target spacing (m) of the close-up terrain mesh; the far mesh is twice
this. Rendering cost on CPU is dominated by mesh size."""
CONTEXT_PAD = 200e3
"""Flat map shown around the SW4 grid (m)."""
CONTEXT_RESOLUTION = 500.0
"""Resolution of the flat context map (m)."""
LAKE_MAX_ELEVATION = 0.5
"""Water inside the coastline sits at 0 m in NZCVM's surface (m)."""
LAKE_MIN_AREA = 4e6
"""Smallest patch of 0 m ground inside the coastline drawn as a lake (m^2)."""
PixelMap = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]
"""Maps NZTM `(easting, northing)` arrays to raster `(column, row)` pixels,
with pixel centres at integer coordinates."""

HILLSHADE_EXAGGERATION = 2.0
"""Vertical exaggeration of the baked hillshade."""
HILLSHADE_STRENGTH = 0.55
"""How strongly the hillshade darkens and lightens the land tint."""
LIGHT_AZIMUTH, LIGHT_ALTITUDE = 315.0, 42.0
"""Sun position for the hillshade (degrees): low from the north-west."""
OUTSIDE_DIM = 0.45
"""How far ground outside the simulated domain is greyed and darkened."""
POINT_SOURCE_LENGTH = 3e3
"""Ruptures shorter than this (m) are drawn as an epicentre only."""


@dataclasses.dataclass
class Town:
    """A town to label."""

    name: str
    """Display name."""
    population: int
    """Population, used to rank labels."""
    easting: float
    """NZTM easting (m)."""
    northing: float
    """NZTM northing (m)."""


def load_towns() -> list[Town]:
    """Load the vendored New Zealand town list.

    Returns
    -------
    list[Town]
        Towns, largest first.
    """
    to_nztm = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    text = resources.files("shake_movie").joinpath(PLACES).read_text()
    rows = csv.DictReader(
        line for line in text.splitlines() if not line.startswith("#")
    )
    towns = []
    for row in rows:
        easting, northing = to_nztm.transform(
            float(row["longitude"]), float(row["latitude"])
        )
        towns.append(Town(row["name"], int(row["population"]), easting, northing))
    return sorted(towns, key=lambda town: -town.population)


def _read_gmt_rings(path: Path) -> shapely.Geometry:
    """Read a GMT multi-segment file of lon/lat rings as NZTM land polygons."""
    to_nztm = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    polygons, ring = [], []
    for line in [*path.read_text().splitlines(), ">"]:
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if len(ring) >= 4:
                lon, lat = np.asarray(ring).T
                polygons.append(
                    shapely.Polygon(np.column_stack(to_nztm.transform(lon, lat)))
                )
            ring = []
        elif line.strip():
            ring.append([float(value) for value in line.split()[:2]])
    return shapely.MultiPolygon(polygons)


def load_coastline(path: Path) -> shapely.Geometry:
    """Load land polygons for the map.

    Parameters
    ----------
    path : Path
        Either an nzcvm coastline (`*.wkb.gz`, gzipped WKB polygons in NZTM)
        or a GMT multi-segment file of lon/lat rings, such as LINZ's.

    Returns
    -------
    shapely.Geometry
        Land polygons in NZTM.
    """
    if path.suffix == ".gmt":
        return _read_gmt_rings(path)
    with gzip.open(path) as handle:
        return shapely.from_wkb(handle.read())


def cache_directory() -> Path:
    """Find where downloaded map data is kept.

    Returns
    -------
    Path
        `$XDG_CACHE_HOME/shake_movie`, or `~/.cache/shake_movie`.
    """
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "shake_movie"


def download_coastline() -> Path:
    """Fetch LINZ's New Zealand coastline, once.

    Returns
    -------
    Path
        The cached `NZ.gmt` file.
    """
    path = cache_directory() / "NZ.gmt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(LINZ_COASTLINE_URL, timeout=120) as source:
            partial.write_bytes(source.read())
        partial.replace(path)
    return path


def rasterise(
    polygons: list[list[npt.NDArray[np.float64]]], shape: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Rasterise polygons given in pixel coordinates, with antialiased edges.

    Parameters
    ----------
    polygons : list[list[np.ndarray]]
        Each polygon's rings as `(n, 2)` arrays of (column, row) pixel
        coordinates: the exterior, then any holes.
    shape : tuple[int, int]
        Raster shape `(rows, columns)`.

    Returns
    -------
    np.ndarray
        Coverage fraction in [0, 1] for each pixel.
    """
    rows, columns = shape
    figure = Figure(figsize=(columns / 100, rows / 100), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(-0.5, columns - 0.5)
    axes.set_ylim(rows - 0.5, -0.5)
    axes.axis("off")
    figure.patch.set_facecolor("black")
    for rings in polygons:
        vertices = np.concatenate(rings)
        codes = np.concatenate(
            [
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(ring) - 2)
                + [MplPath.CLOSEPOLY]
                for ring in rings
            ]
        )
        axes.add_patch(
            PathPatch(MplPath(vertices, codes), facecolor="white", edgecolor="none")
        )
    canvas.draw()
    return (np.asarray(canvas.buffer_rgba())[..., 0] / 255).astype(np.float32)


def land_fraction(
    coastline: shapely.Geometry,
    bounds: tuple[float, float, float, float],
    to_pixels: PixelMap,
    shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Rasterise the coastline's land onto a grid.

    Parameters
    ----------
    coastline : shapely.Geometry
        Land polygons in NZTM.
    bounds : tuple[float, float, float, float]
        NZTM bounds `(min_e, min_n, max_e, max_n)` covering the grid.
    to_pixels : PixelMap
        Maps NZTM `(easting, northing)` arrays to `(column, row)` pixels.
    shape : tuple[int, int]
        Grid shape `(rows, columns)`.

    Returns
    -------
    np.ndarray
        Land fraction in [0, 1] for each grid point.
    """
    clipped = shapely.clip_by_rect(coastline, *bounds)
    polygons = []
    for polygon in shapely.get_parts(clipped):
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        rings = []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords)
            column, row = to_pixels(coordinates[:, 0], coordinates[:, 1])
            rings.append(np.column_stack([column, row]))
        polygons.append(rings)
    return rasterise(polygons, shape)


def lakes(
    elevation: npt.NDArray[np.float32], land: npt.NDArray[np.float32], cell_area: float
) -> npt.NDArray[np.float32]:
    """Find lakes and lagoons: large patches of 0 m ground inside the coastline.

    NZCVM's coastline counts lagoons such as Te Waihora as land, but its
    surface puts their water at 0 m, like the sea.

    Parameters
    ----------
    elevation : np.ndarray
        Ground elevation (m).
    land : np.ndarray
        Land fraction from the coastline.
    cell_area : float
        Area of one cell (m^2).

    Returns
    -------
    np.ndarray
        Lake fraction in [0, 1] for each cell.
    """
    water = ndimage.binary_opening(
        (elevation <= LAKE_MAX_ELEVATION) & (land > 0.5), iterations=1
    )
    labels, count = ndimage.label(water)
    if count == 0:
        return np.zeros_like(elevation, dtype=np.float32)
    sizes = ndimage.sum(water, labels, range(1, count + 1)) * cell_area
    keep = np.isin(labels, 1 + np.flatnonzero(sizes >= LAKE_MIN_AREA))
    return ndimage.gaussian_filter(keep.astype(np.float32), 0.7)


def hillshade(
    elevation: npt.NDArray[np.float32], spacing: float, azimuth: float
) -> npt.NDArray[np.float32]:
    """Shade terrain on SW4's rotated grid, lit from the north-west on the map.

    Parameters
    ----------
    elevation : np.ndarray
        Elevation (m) on the grid, `(i, j)` along SW4's x and y.
    spacing : float
        Grid spacing (m).
    azimuth : float
        Bearing of SW4's x axis (degrees).

    Returns
    -------
    np.ndarray
        Lambertian shade in [0, 1].
    """
    dz_dx, dz_dy = np.gradient(elevation * HILLSHADE_EXAGGERATION, spacing)
    az = np.radians(azimuth)
    dz_de = dz_dx * np.sin(az) + dz_dy * np.cos(az)
    dz_dn = dz_dx * np.cos(az) - dz_dy * np.sin(az)
    light_az, light_alt = np.radians(LIGHT_AZIMUTH), np.radians(LIGHT_ALTITUDE)
    light = np.array(
        [
            np.sin(light_az) * np.cos(light_alt),
            np.cos(light_az) * np.cos(light_alt),
            np.sin(light_alt),
        ]
    )
    shade = (-dz_de * light[0] - dz_dn * light[1] + light[2]) / np.sqrt(
        dz_de**2 + dz_dn**2 + 1
    )
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def hex_rgb(colour: str) -> npt.NDArray[np.float32]:
    """Convert a hex colour to RGB floats.

    Parameters
    ----------
    colour : str
        Hex colour, e.g. `#5f8197`.

    Returns
    -------
    np.ndarray
        RGB in [0, 1].
    """
    return np.array(matplotlib.colors.to_rgb(colour), dtype=np.float32)


def tint(
    elevation: npt.NDArray[np.float32],
    land: npt.NDArray[np.float32],
    lake: npt.NDArray[np.float32],
    shade: npt.NDArray[np.float32] | None,
    spacing: float,
) -> npt.NDArray[np.float32]:
    """Colour terrain: hypsometric tint on land, sea darkening offshore.

    Parameters
    ----------
    elevation : np.ndarray
        Elevation (m).
    land : np.ndarray
        Land fraction.
    lake : np.ndarray
        Lake fraction.
    shade : np.ndarray | None
        Hillshade, or None for flat ground.
    spacing : float
        Cell size (m), for the offshore distance.

    Returns
    -------
    np.ndarray
        RGB in [0, 1], shape `elevation.shape + (3,)`.
    """
    top = palette.LAND_STOPS[-1][0]
    rgb_land = palette.land_cmap()(np.clip(elevation, 0, top) / top)[..., :3].astype(
        np.float32
    )
    if shade is not None:
        flat = np.sin(np.radians(LIGHT_ALTITUDE))
        rgb_land = np.clip(
            rgb_land * (1 + HILLSHADE_STRENGTH * (shade - flat))[..., None], 0, 1
        )
    offshore = ndimage.distance_transform_edt(land < 0.5) * spacing
    k = np.clip(offshore / palette.SEA_FADE_DISTANCE, 0, 1)[..., None]
    rgb_sea = hex_rgb(palette.SEA_NEAR) * (1 - k) + hex_rgb(palette.SEA_FAR) * k
    a = land[..., None]
    rgb = rgb_land * a + rgb_sea * (1 - a)
    w = (lake * land)[..., None]
    return rgb * (1 - w) + hex_rgb(palette.LAKE) * w


def dim(
    rgb: npt.NDArray[np.float32], amount: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Grey and darken ground outside the simulated domain.

    Parameters
    ----------
    rgb : np.ndarray
        RGB in [0, 1].
    amount : np.ndarray
        0 inside the domain, 1 outside.

    Returns
    -------
    np.ndarray
        The dimmed colours.
    """
    grey = rgb.mean(-1, keepdims=True)
    dimmed = (rgb * 0.35 + grey * 0.65) * (1 - OUTSIDE_DIM * 0.7)
    a = amount[..., None]
    return rgb * (1 - a) + dimmed * a


@dataclasses.dataclass
class Rupture:
    """The fault as it ruptures, from the SRF SW4 ran."""

    segments: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    """Per SRF plane: the NZTM `(n, 2)` top-edge polyline and when each point
    along it starts to rupture (s)."""
    hypocentre: npt.NDArray[np.float64]
    """NZTM position of the first point to rupture (the epicentre)."""
    end_time: float
    """When the last point starts to rupture (s)."""

    @property
    def is_point(self) -> bool:
        """Whether the rupture is too short to draw as a line.

        Returns
        -------
        bool
            True for ruptures shorter than `POINT_SOURCE_LENGTH`.
        """
        length = sum(
            float(np.hypot(*np.diff(line, axis=0).T).sum()) for line, _ in self.segments
        )
        return length < POINT_SOURCE_LENGTH

    def ruptured(self, time: float) -> list[npt.NDArray[np.float64]]:
        """Return the parts of the fault that have ruptured by `time`.

        Parameters
        ----------
        time : float
            Simulation time (s).

        Returns
        -------
        list[np.ndarray]
            Polylines of the ruptured parts.
        """
        parts = []
        for line, start in self.segments:
            indices = np.flatnonzero(start <= time)
            gaps = np.flatnonzero(np.diff(indices) > 1) + 1
            parts.extend(line[run] for run in np.split(indices, gaps) if run.size > 1)
        return parts

    @classmethod
    def from_srf(cls, path: Path) -> "Rupture":
        """Read the rupture from an SW4 `rupturehdf5` SRF file.

        Parameters
        ----------
        path : Path
            The SRF HDF5 file.

        Returns
        -------
        Rupture
            The rupture.
        """
        with h5py.File(path, "r") as handle:
            planes = np.atleast_1d(handle.attrs["PLANE"])
            points = handle["POINTS"][:]
        to_nztm = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
        segments = []
        offset = 0
        for plane in planes:
            nstk, ndip = int(plane["NSTK"]), int(plane["NDIP"])
            block = points[offset : offset + nstk * ndip]
            offset += nstk * ndip
            # SRF lists each plane's points with along-strike varying fastest,
            # so the first row is the shallowest.
            tinit = block["TINIT"].reshape(ndip, nstk).min(axis=0)
            easting, northing = to_nztm.transform(
                block["LON"][:nstk], block["LAT"][:nstk]
            )
            segments.append(
                (np.column_stack([easting, northing]), tinit.astype(np.float64))
            )
        first = int(np.argmin(points["TINIT"]))
        hypocentre = np.array(
            to_nztm.transform(points["LON"][first], points["LAT"][first])
        )
        return cls(segments, hypocentre, float(points["TINIT"].max()))


@dataclasses.dataclass
class Scene:
    """Static geometry and textures, prepared once and shared with workers."""

    mesh_easting: npt.NDArray[np.float64]
    """NZTM easting of close-up mesh nodes, `(i, j)` along SW4's axes."""
    mesh_northing: npt.NDArray[np.float64]
    """NZTM northing of close-up mesh nodes."""
    mesh_elevation: npt.NDArray[np.float32]
    """Ground elevation (m) at close-up mesh nodes (0 on the sea)."""
    mesh_texture_uv: npt.NDArray[np.float64]
    """Texture coordinates of close-up mesh nodes, `(i, j, 2)`."""
    mesh_index: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
    """SW4 image indices `(i, j)` of close-up mesh nodes."""
    terrain_texture: npt.NDArray[np.uint8]
    """Terrain colours on the SW4 grid, `(rows, columns, 3)`."""
    context_bounds: tuple[float, float, float, float]
    """NZTM bounds of the flat context map."""
    context_texture: npt.NDArray[np.uint8]
    """Flat context map, north up, `(rows, columns, 3)`."""
    wave_stride: int
    """Stride from SW4 image points to shaking texture pixels."""
    wave_opacity: npt.NDArray[np.float32]
    """Opacity multiplier for the shaking texture: faint at sea, none outside
    the domain."""
    wave_easting: npt.NDArray[np.float64]
    """NZTM easting of shaking texture pixels."""
    wave_northing: npt.NDArray[np.float64]
    """NZTM northing of shaking texture pixels."""
    wave_inside: npt.NDArray[np.bool_]
    """Whether each shaking texture pixel is inside the simulated domain."""
    height_bounds: tuple[float, float, float, float]
    """NZTM bounds of `height`."""
    height: npt.NDArray[np.float32]
    """Ground elevation (m) on a regular north-up NZTM grid, for draping
    labels and lines and testing what the terrain hides."""
    domain: npt.NDArray[np.float64]
    """NZTM corners of the simulated domain (inside SW4's sponge), `(4, 2)`."""
    towns: list[Town]
    """Towns inside the context map."""
    rupture: Rupture
    """The rupture."""
    title: str
    """The movie's title, e.g. "Thursday 26th March Mw 6.3 earthquake"."""
    location: str
    """Where the epicentre is, in words."""
    is_real: bool
    """Whether this is an earthquake that happened, rather than a scenario."""

    def height_at(
        self, easting: npt.ArrayLike, northing: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """Look up ground elevation, bilinearly.

        Parameters
        ----------
        easting : array-like
            NZTM easting (m).
        northing : array-like
            NZTM northing (m).

        Returns
        -------
        np.ndarray
            Elevation (m); 0 off the grid.
        """
        min_e, min_n, max_e, max_n = self.height_bounds
        rows, columns = self.height.shape
        column = (np.asarray(easting) - min_e) / (max_e - min_e) * (columns - 1)
        row = (max_n - np.asarray(northing)) / (max_n - min_n) * (rows - 1)
        return ndimage.map_coordinates(
            self.height, [np.atleast_1d(row), np.atleast_1d(column)], order=1, cval=0.0
        ).reshape(np.shape(easting))


def _stride(size: int, spacing: float, target: float, max_pixels: int) -> int:
    """Pick an integer stride that meets a spacing and a pixel budget."""
    return max(1, round(target / spacing), int(np.ceil(size / max_pixels)))


def build_scene(
    geometry: GridGeometry,
    topo: npt.NDArray[np.float32],
    spacing: float,
    domain: npt.NDArray[np.float64],
    coastline: shapely.Geometry,
    rupture: Rupture,
    title: str,
    location: str,
    is_real: bool,
) -> Scene:
    """Prepare the terrain, map and labels for a movie.

    Parameters
    ----------
    geometry : GridGeometry
        Where SW4's grid sits.
    topo : np.ndarray
        SW4's `topo` image (m, positive up), `(ni, nj)`.
    spacing : float
        SW4 image spacing (m).
    domain : np.ndarray
        NZTM corners of the simulated domain, `(4, 2)`.
    coastline : shapely.Geometry
        Land polygons in NZTM.
    rupture : Rupture
        The rupture.
    title : str
        The movie's title.
    location : str
        Where the epicentre is, in words.
    is_real : bool
        Whether this is an earthquake that happened.

    Returns
    -------
    Scene
        The prepared scene.
    """
    ni, nj = topo.shape
    topo = np.nan_to_num(topo, nan=0.0)

    def _grid_pixels(stride: int) -> PixelMap:
        """Map NZTM to pixels of the SW4 grid sampled every `stride` points."""

        # Raster rows run along SW4's x axis and columns along its y axis.
        def _to_pixels(
            easting: npt.NDArray[np.float64], northing: npt.NDArray[np.float64]
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Map NZTM to (column, row) pixels of the strided grid."""
            x, y = geometry.from_nztm(easting, northing)
            return y / (spacing * stride), x / (spacing * stride)

        return _to_pixels

    corners_i = np.array([0, ni - 1, ni - 1, 0]) * spacing
    corners_j = np.array([0, 0, nj - 1, nj - 1]) * spacing
    corner_e, corner_n = geometry.to_nztm(corners_i, corners_j)
    grid_bounds = (
        float(corner_e.min()) - 2e3,
        float(corner_n.min()) - 2e3,
        float(corner_e.max()) + 2e3,
        float(corner_n.max()) + 2e3,
    )

    # Inside the domain: a rectangle in SW4's own coordinates.
    domain_x, domain_y = geometry.from_nztm(domain[:, 0], domain[:, 1])

    def _inside_domain(
        i: npt.NDArray[np.float64], j: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """Whether SW4 image points `(i, j)` lie inside the simulated domain."""
        x, y = i * spacing, j * spacing
        return (
            (x >= domain_x.min())
            & (x <= domain_x.max())
            & (y >= domain_y.min())
            & (y <= domain_y.max())
        )

    # Terrain texture on SW4's grid.
    t = _stride(max(ni, nj), spacing, spacing, TEXTURE_MAX_PIXELS)
    elevation = topo[::t, ::t]
    land = land_fraction(coastline, grid_bounds, _grid_pixels(t), elevation.shape)
    lake = lakes(elevation, land, (spacing * t) ** 2)
    ground = np.where(land > 0.5, np.maximum(elevation, 0.0), 0.0).astype(np.float32)
    shade = hillshade(elevation, spacing * t, geometry.azimuth)
    rgb = tint(elevation, land, lake, shade, spacing * t)
    ti, tj = np.meshgrid(
        np.arange(elevation.shape[0]) * t,
        np.arange(elevation.shape[1]) * t,
        indexing="ij",
    )
    rgb = dim(rgb, (~_inside_domain(ti, tj)).astype(np.float32))
    terrain_texture = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # Close-up mesh.
    m = _stride(max(ni, nj), spacing, MESH_SPACING, 10**9)
    mi = np.arange(0, ni, m)
    mj = np.arange(0, nj, m)
    mesh_i, mesh_j = np.meshgrid(mi, mj, indexing="ij")
    mesh_easting, mesh_northing = geometry.to_nztm(mesh_i * spacing, mesh_j * spacing)
    mesh_elevation = ndimage.map_coordinates(
        ndimage.uniform_filter(ground, max(1, m // t)),
        [mesh_i / t, mesh_j / t],
        order=1,
    ).astype(np.float32)
    rows, columns = terrain_texture.shape[:2]
    mesh_texture_uv = np.stack(
        [(mesh_j / t + 0.5) / columns, 1.0 - (mesh_i / t + 0.5) / rows], axis=-1
    )

    # Shaking texture: opacity for sea and sponge.
    w = _stride(max(ni, nj), spacing, spacing, WAVE_MAX_PIXELS)
    wi, wj = np.meshgrid(np.arange(0, ni, w), np.arange(0, nj, w), indexing="ij")
    wave_land = ndimage.map_coordinates(land, [wi / t, wj / t], order=1)
    wave_inside = _inside_domain(wi, wj)
    wave_opacity = (
        np.where(wave_land > 0.5, 1.0, palette.SEA_SHAKING_OPACITY) * wave_inside
    ).astype(np.float32)
    wave_easting, wave_northing = geometry.to_nztm(wi * spacing, wj * spacing)

    # Flat context map, north up.
    min_e, min_n, max_e, max_n = grid_bounds
    context_bounds = (
        min_e - CONTEXT_PAD,
        min_n - CONTEXT_PAD,
        max_e + CONTEXT_PAD,
        max_n + CONTEXT_PAD,
    )
    columns = int((context_bounds[2] - context_bounds[0]) / CONTEXT_RESOLUTION)
    rows = int((context_bounds[3] - context_bounds[1]) / CONTEXT_RESOLUTION)

    def _context_pixels(
        easting: npt.NDArray[np.float64], northing: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map NZTM to (column, row) pixels of the context map."""
        return (
            (easting - context_bounds[0]) / CONTEXT_RESOLUTION - 0.5,
            (context_bounds[3] - northing) / CONTEXT_RESOLUTION - 0.5,
        )

    context_land = land_fraction(
        coastline, context_bounds, _context_pixels, (rows, columns)
    )
    context_rgb = tint(
        np.zeros((rows, columns), np.float32),
        context_land,
        np.zeros((rows, columns), np.float32),
        None,
        CONTEXT_RESOLUTION,
    )
    context_texture = (
        np.clip(dim(context_rgb, np.ones((rows, columns), np.float32)), 0, 1) * 255
    ).astype(np.uint8)

    # Elevation on a regular NZTM grid, for draping and occlusion.
    ce = context_bounds[0] + (np.arange(columns) + 0.5) * CONTEXT_RESOLUTION
    cn = context_bounds[3] - (np.arange(rows) + 0.5) * CONTEXT_RESOLUTION
    context_e, context_n = np.meshgrid(ce, cn)
    gx, gy = geometry.from_nztm(context_e, context_n)
    height = ndimage.map_coordinates(
        ground, [gx / (spacing * t), gy / (spacing * t)], order=1, cval=0.0
    ).astype(np.float32)
    height_bounds = (float(ce[0]), float(cn[-1]), float(ce[-1]), float(cn[0]))

    towns = [
        town
        for town in load_towns()
        if context_bounds[0] <= town.easting <= context_bounds[2]
        and context_bounds[1] <= town.northing <= context_bounds[3]
    ]
    return Scene(
        mesh_easting=mesh_easting,
        mesh_northing=mesh_northing,
        mesh_elevation=mesh_elevation,
        mesh_texture_uv=mesh_texture_uv,
        mesh_index=(mesh_i, mesh_j),
        terrain_texture=terrain_texture,
        context_bounds=context_bounds,
        context_texture=context_texture,
        wave_stride=w,
        wave_opacity=wave_opacity,
        wave_easting=wave_easting,
        wave_northing=wave_northing,
        wave_inside=wave_inside,
        height_bounds=height_bounds,
        height=height,
        domain=domain,
        towns=towns,
        rupture=rupture,
        title=title,
        location=location,
        is_real=is_real,
    )
