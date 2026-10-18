# shake_movie

Public-facing earthquake videos from SW4 ground-motion simulations, for
QuakeCoRE's outreach channel. One command turns a finished SW4 run into a
1080p MP4 with no per-event settings, so the realtime simulation workflow can
render, and later upload, a movie for every earthquake it simulates.

```
shake-movie SW4_INPUT REALISATION [--intensity-measures intensity_measures.h5] [--output MOVIE.mp4]
```

Next to the movie it writes `MOVIE.json` with the title, place, date,
magnitude and a description for the upload.

## What it reads

Everything comes from files a workflow run already has. The SW4 input file
says where the rest is:

| File | From | Used for |
|---|---|---|
| `topo` image (z = 0, cycle 0) | SW4 `imagehdf5 mode=topo` | Terrain mesh and hillshade |
| `surf_mag` images (z = 0, every 0.5 s) | SW4 `imagehdf5 mode=mag` | Shaking, interpolated in time to the frame rate |
| `surf_hmax` image (z = 0, end) | SW4 `imagehdf5 mode=hmax` | Strongest shaking map, without intensity measures |
| SRF (`rupturehdf5`) | the SRF SW4 ran | Fault trace, rupture timing, epicentre |
| Realisation JSON | the workflow | Magnitude, simulated domain, coastline |
| `intensity_measures.h5` (`--intensity-measures`) | the workflow's `im-calc` | Strongest shaking map (broadband PGV) |

The workflow's defaults already write these images. SW4 driven by an SRF slip
rate outputs velocity where its modes say displacement, so `mag` is ground
velocity and `hmax` peak horizontal ground velocity. Velocity becomes
intensity through Worden et al. (2012).

The final map uses the broadband intensity measures when given: RotD50 PGV
(`/PGV/rotd50`) at `im-calc`'s stations, the real ones plus its virtual grid,
leaving out stations inside SW4's absorbing layer (`supergrid_depth_gp > 0`).
PGV is interpolated linearly in its logarithm between stations, and nothing
is drawn beyond them (outside their hull, or over 10 km from any). Without
the file the map falls back to SW4's `hmax`, which is only the low-frequency
motion and underestimates how strongly smaller earthquakes are felt.

The grid is placed on the map exactly as SW4 places it: the PROJ string is
rebuilt the way SW4 assembles it from the `grid` command. That includes
passing `scale=0.9996` through as `+scale`, which PROJ ignores for `tmerc`, so
SW4's grid has an effective scale factor of 1. The test fixtures are real SW4
output, and the placement matches SW4's own `lat`/`lon` images to float32
rounding (under 1 m).

## Look

- **Terrain:** a muted hypsometric tint from a lowland green, with a baked
  hillshade and 2x vertical exaggeration. Land and sea come from the coastline,
  never from elevation: NZCVM stores the sea as 0 m, the same height as much of
  the Canterbury Plains. Lagoons such as Te Waihora are found as large 0 m
  patches inside the coastline.
- **Shaking:** ground velocity in cm/s on one colour scale for the whole
  movie, in magma, see-through at its low end and faint over the sea. The
  scale tops out at the strongest broadband shaking on land (`rotd50` PGV from
  `--intensity-measures`) and spans two decades below it. The legend reads
  "Ground velocity (cm/s)" throughout, ticked at round velocities (1, 2 and 5
  of each decade, thinned to fit the bar, as eqvis_workflow ticks its colour
  bars).
- **Waves:** SW4's low-frequency ground velocity scaled so its strongest (the
  largest `mag` value inside the domain) reaches the broadband peak. SW4's
  low-frequency motion from a Mw 4 peaks around 0.003 cm/s, far below what is
  felt, so drawn as it is it would show nothing; scaled, the waves fill the
  same scale the final map is drawn on.
- **Fault:** drawn grey, turning white as it ruptures ("Fault rupturing").
- **Ending:** a north-up map of the strongest broadband shaking on land
  (clipped to the coastline) on the same scale, with the word for each town's
  shaking (GeoNet's Weak, Light, Moderate, Strong, Severe, Extreme, from
  intensity by Worden et al., 2012) printed under its name. When nothing on
  land reaches "Weak", the map says "Strongest shaking: too weak for most
  people to notice". Without `--intensity-measures`, SW4's low-frequency peak
  (`hmax`) stands in for the broadband.
- **Domain edge:** only the simulated domain is drawn in relief; SW4's
  absorbing layer around it is the same flat, greyed map as everything
  outside, with a dark wall closing the gap under the domain's edge.

## Camera

Every shot is worked out from the rupture and the shaking (`camera.py`):

1. Look across the fault from the side where more people live, keeping north
   within 60 degrees of screen-up. Point sources look north.
2. Frame the ruptured part of the fault plus everywhere moving at a tenth or
   more of the event's strongest ground velocity, padded 30%, at least 70 km
   across.
3. Only ever pull back: the framed size is a running maximum.
4. Smooth the path with look-ahead, then evaluate it at every frame through
   monotone splines. The camera glides rather than stepping at SW4's image
   interval.
5. Ease the pitch from 40 to 55 degrees as the view widens, with a slow 12
   degree drift for parallax.

The movie runs: a 4 s title over the whole domain, a 2 s dive to the
epicentre, the simulation in real time, a 3 s tilt up to the map, and a 6 s
hold.

## Names

Titles read like "Thursday 26th March Mw 6.3 earthquake". For a realisation
named by a GeoNet public ID (as the realtime workflow names them), the date
(NZ time) and the place ("5 km west of Masterton") come from
[GeoNet's quake API](https://api.geonet.org.nz). Scenarios have no date, and
their place comes from [Nominatim](https://nominatim.org) reverse geocoding
("Hurunui District, Canterbury"). Neither needs a key; a movie makes at most
one request to each, identified by this tool's User-Agent. Offshore, or
without a network (`--offline`), the place is given relative to the nearest
sizeable town in the vendored town list.

## Rendering on CPUs

VTK renders off-screen through Mesa (EGL or OSMesa, llvmpipe) where there is
no GPU, as on HPC nodes. The terrain texture is uploaded once per worker, and
each frame only swaps a small shaking texture on a second copy of the terrain.
Mesh size dominates the cost, so a coarser mesh takes over once the camera is
more than 180 km out. Frames are split across worker processes (`--workers`,
by default the CPUs available) and piped straight into ffmpeg, one segment
each, then joined without re-encoding.

The container needs `ffmpeg` and Mesa (`libegl1 libegl-mesa0
libgl1-mesa-dri`, or `libosmesa6`).

## Data

- Coastline: the velocity model's nzcvm coastline when the realisation names
  one that exists. Otherwise [LINZ NZ Coastlines and Islands Polygons
  (Topo 1:150k)](https://data.linz.govt.nz/layer/1153), CC BY 3.0 NZ,
  downloaded once to `~/.cache/shake_movie`.
- Towns (`shake_movie/nz_places.csv`): cities, towns and villages with a
  population from OpenStreetMap, © OpenStreetMap contributors, ODbL 1.0.
- Places and dates: GeoNet and OpenStreetMap Nominatim, as above.

The final map credits SW4, LINZ and OpenStreetMap.

## Development

```
uv sync --all-extras --dev
uvx lefthook install
uv run pytest
```

The render tests need ffmpeg and an OpenGL driver (Mesa is enough), and skip
without them. `tests/fake_sw4.py` writes files in SW4's layouts for building
test runs.
