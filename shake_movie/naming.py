"""Name the movie: its title, where the earthquake was, and when.

Titles read like "Thursday 26th March Mw 6.3 earthquake". For a real
earthquake (a realisation named by its GeoNet public ID, as the realtime
workflow names them) the date and the place come from GeoNet's quake API,
whose locality ("5 km west of Masterton") is what the public already sees on
GeoNet. Other simulations have no date; their place comes from
OpenStreetMap's Nominatim reverse geocoder. Neither service needs a key, and
each movie makes at most one request to each. If a service cannot help, the
place falls back to the nearest town in the vendored town list.
"""

import dataclasses
import datetime
import importlib.metadata
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from shake_movie.scene import Town
from shake_movie.sw4_images import NZTM

GEONET_QUAKE_URL = "https://api.geonet.org.nz/quake/{}"
"""GeoNet's quake API (https://api.geonet.org.nz), one event by public ID."""
GEONET_PUBLIC_ID = re.compile(r"^\d{4}p\d{6}$")
"""GeoNet public IDs, e.g. 2026p711868."""
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
"""OpenStreetMap's reverse geocoder. Its usage policy asks for an identifying
User-Agent and at most one request per second; a movie makes one."""
NZ_TIME = ZoneInfo("Pacific/Auckland")
"""Dates are given in New Zealand time."""
TIMEOUT = 20.0
"""Seconds to wait for a service before falling back."""

logger = logging.getLogger(__name__)

COMPASS = [
    "north",
    "north-east",
    "east",
    "south-east",
    "south",
    "south-west",
    "west",
    "north-west",
]


@dataclasses.dataclass(frozen=True)
class Naming:
    """How the movie names the earthquake."""

    title: str
    """For example "Thursday 26th March Mw 6.3 earthquake"."""
    location: str
    """Where the epicentre was, in words."""
    time: datetime.datetime | None
    """When the earthquake happened (NZ time), for a real earthquake."""

    @property
    def is_real(self) -> bool:
        """Whether the movie shows an earthquake that happened.

        Returns
        -------
        bool
            True when the earthquake's date is known.
        """
        return self.time is not None


def ordinal(day: int) -> str:
    """Write a day of the month as an ordinal.

    Parameters
    ----------
    day : int
        Day of the month.

    Returns
    -------
    str
        For example "1st", "22nd", "26th".
    """
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def make_title(magnitude: float, time: datetime.datetime | None) -> str:
    """Write the movie's title.

    Parameters
    ----------
    magnitude : float
        Moment magnitude.
    time : datetime.datetime | None
        When the earthquake happened, or None for a scenario.

    Returns
    -------
    str
        For example "Thursday 26th March Mw 6.3 earthquake".
    """
    quake = f"Mw {magnitude:.1f} earthquake"
    if time is None:
        return quake
    return f"{time:%A} {ordinal(time.day)} {time:%B} {quake}"


def user_agent() -> str:
    """Identify this tool to web services.

    Returns
    -------
    str
        The User-Agent header.
    """
    try:
        version = importlib.metadata.version("shake-movie")
    except importlib.metadata.PackageNotFoundError:
        version = "0"
    return f"shake_movie/{version} (+https://github.com/ucgmsim/shake_movie)"


def _get_json(url: str, accept: str = "application/json") -> dict | None:
    """Fetch a JSON document, or None if the service cannot be reached."""
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent(), "Accept": accept}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        logger.warning("Could not reach %s: %s", url, error)
        return None


def geonet_event(public_id: str) -> tuple[datetime.datetime, str] | None:
    """Look up a real earthquake's time and locality on GeoNet.

    Parameters
    ----------
    public_id : str
        The GeoNet public ID.

    Returns
    -------
    tuple[datetime.datetime, str] | None
        The origin time in NZ time and GeoNet's locality, or None.
    """
    document = _get_json(
        GEONET_QUAKE_URL.format(public_id), "application/vnd.geo+json;version=2"
    )
    features = (document or {}).get("features") or []
    if not features:
        return None
    properties = features[0]["properties"]
    time = datetime.datetime.fromisoformat(properties["time"])
    return time.astimezone(NZ_TIME), properties.get("locality", "")


def reverse_geocode(longitude: float, latitude: float) -> str | None:
    """Name the district and region a point lies in, with Nominatim.

    Parameters
    ----------
    longitude : float
        Longitude.
    latitude : float
        Latitude.

    Returns
    -------
    str | None
        For example "Hurunui District, Canterbury", or None at sea.
    """
    query = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{latitude:.5f}",
            "lon": f"{longitude:.5f}",
            "zoom": 10,
            "accept-language": "en",
        }
    )
    document = _get_json(f"{NOMINATIM_URL}?{query}")
    address = (document or {}).get("address")
    if not address:
        return None
    parts = [
        address.get(key)
        for key in ("city", "town", "village", "county", "state")
        if address.get(key)
    ]
    return ", ".join(dict.fromkeys(parts[:2])) or None


def nearest_town(point: npt.NDArray[np.float64], towns: list[Town]) -> str:
    """Describe a point relative to the nearest sizeable town.

    Parameters
    ----------
    point : np.ndarray
        NZTM point.
    towns : list[Town]
        Candidate towns.

    Returns
    -------
    str
        For example "15 km north-west of Hanmer Springs".
    """
    sizeable = [town for town in towns if town.population >= 1000] or towns
    nearest = min(
        sizeable, key=lambda t: np.hypot(t.easting - point[0], t.northing - point[1])
    )
    de, dn = point[0] - nearest.easting, point[1] - nearest.northing
    distance = float(np.hypot(de, dn))
    if distance < 3e3:
        return f"Near {nearest.name}"
    bearing = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    direction = COMPASS[int((bearing + 22.5) // 45) % 8]
    km = distance / 1000.0
    km = round(km) if km < 20 else 5 * round(km / 5)
    return f"{km:.0f} km {direction} of {nearest.name}"


def name_earthquake(
    realisation_name: str,
    magnitude: float,
    epicentre: npt.NDArray[np.float64],
    towns: list[Town],
    online: bool = True,
) -> Naming:
    """Work out the movie's title and place name.

    Parameters
    ----------
    realisation_name : str
        The realisation's name; a GeoNet public ID for real earthquakes.
    magnitude : float
        Moment magnitude.
    epicentre : np.ndarray
        NZTM epicentre.
    towns : list[Town]
        Towns for the offline fallback.
    online : bool
        Whether to ask GeoNet and Nominatim.

    Returns
    -------
    Naming
        The title, place and time.
    """
    time = location = None
    if online and GEONET_PUBLIC_ID.match(realisation_name):
        event = geonet_event(realisation_name)
        if event is not None:
            time, location = event
    if online and not location:
        longitude, latitude = Transformer.from_crs(
            NZTM, "EPSG:4326", always_xy=True
        ).transform(*epicentre)
        location = reverse_geocode(longitude, latitude)
    if not location:
        location = nearest_town(epicentre, towns)
    location = location[0].upper() + location[1:]
    return Naming(make_title(magnitude, time), location, time)
