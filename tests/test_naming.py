"""Tests for naming the earthquake. Services are stubbed: no network."""

import datetime
import http.client
import importlib.metadata
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from shake_movie import naming
from shake_movie.scene import Town

GEONET = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "publicID": "2026p711868",
                "time": "2026-09-21T09:23:42.309Z",
                "locality": "5 km west of Masterton",
            },
        }
    ],
}
NOMINATIM = {
    "address": {
        "county": "Hurunui District",
        "state": "Canterbury",
        "country": "New Zealand",
    }
}
TOWNS = [
    Town("Masterton", 20900, 1824000.0, 5465000.0),
    Town("Tiny", 50, 1810000.0, 5465000.0),
]


@pytest.mark.parametrize(
    ("day", "text"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (26, "26th"),
        (31, "31st"),
    ],
)
def test_ordinal(day: int, text: str) -> None:
    """Days are written as ordinals, including the teens."""
    assert naming.ordinal(day) == text


def test_title_with_a_date() -> None:
    """Real earthquakes are titled by their day."""
    time = datetime.datetime(2026, 3, 26, 14, 0, tzinfo=naming.NZ_TIME)
    assert naming.make_title(6.31, time) == "Thursday 26th March Mw 6.3 earthquake"


def test_title_without_a_date() -> None:
    """Scenarios have no day."""
    assert naming.make_title(7.2, None) == "Mw 7.2 earthquake"


def test_real_earthquake_uses_geonet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GeoNet ID brings GeoNet's time, in NZ time, and its locality."""
    calls = []

    def fake(url: str, accept: str = "") -> dict:
        calls.append(url)
        return GEONET

    monkeypatch.setattr(naming, "_get_json", fake)
    name = naming.name_earthquake(
        "2026p711868", 3.96, np.array([1824000.0, 5465000.0]), TOWNS
    )
    assert name.title == "Monday 21st September Mw 4.0 earthquake"
    assert name.location == "5 km west of Masterton"
    assert name.is_real
    assert name.time is not None and name.time.utcoffset() == datetime.timedelta(
        hours=12
    )
    assert calls == [naming.GEONET_QUAKE_URL.format("2026p711868")]


def test_scenario_uses_nominatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other realisations ask Nominatim for the district and region, once."""
    calls = []

    def fake(url: str, accept: str = "") -> dict:
        calls.append(url)
        return NOMINATIM

    monkeypatch.setattr(naming, "_get_json", fake)
    name = naming.name_earthquake(
        "hope_scenario", 7.2, np.array([1600000.0, 5290000.0]), TOWNS
    )
    assert name.title == "Mw 7.2 earthquake"
    assert name.location == "Hurunui District, Canterbury"
    assert not name.is_real
    assert len(calls) == 1 and calls[0].startswith(naming.NOMINATIM_URL)


def test_unreachable_services_fall_back_to_towns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the services, the place is given from the nearest sizeable town."""
    monkeypatch.setattr(naming, "_get_json", lambda url, accept="": None)
    name = naming.name_earthquake(
        "2026p711868", 4.0, np.array([1824000.0, 5475000.0]), TOWNS
    )
    assert name.title == "Mw 4.0 earthquake"
    assert name.location == "10 km north of Masterton"


def test_offline_makes_no_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline naming never touches the network."""

    def fail(url: str, accept: str = "") -> dict:
        raise AssertionError("no requests offline")

    monkeypatch.setattr(naming, "_get_json", fail)
    name = naming.name_earthquake(
        "2026p711868", 4.0, np.array([1824500.0, 5465000.0]), TOWNS, online=False
    )
    assert name.location == "Near Masterton"


@pytest.mark.parametrize(
    ("offset", "text"),
    [
        ((-5000.0, 0.0), "5 km west of Masterton"),
        ((30000.0, -30000.0), "40 km south-east of Masterton"),
    ],
)
def test_nearest_town_wording(offset: tuple[float, float], text: str) -> None:
    """Distances round to the kilometre, or to 5 km when far, with a compass point."""
    point = np.array([1824000.0 + offset[0], 5465000.0 + offset[1]])
    assert naming.nearest_town(point, TOWNS) == text


def test_user_agent_identifies_the_tool() -> None:
    """Requests name the tool and its source, as Nominatim's policy asks."""
    assert naming.user_agent().startswith("shake_movie/")
    assert "github.com/ucgmsim/shake_movie" in naming.user_agent()


@pytest.mark.parametrize(
    ("day", "text"),
    [(14, "14th"), (101, "101st"), (111, "111th"), (112, "112th"), (121, "121st")],
)
def test_ordinal_beyond_a_month(day: int, text: str) -> None:
    """Only 11-13 within each hundred take "th"; 101 and 121 do not."""
    assert naming.ordinal(day) == text


def test_user_agent_gives_the_installed_version() -> None:
    """The User-Agent carries the installed package's version."""
    version = importlib.metadata.version("shake-movie")
    assert naming.user_agent() == (
        f"shake_movie/{version} (+https://github.com/ucgmsim/shake_movie)"
    )


def test_user_agent_without_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running from an uninstalled checkout reports version 0."""

    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert naming.user_agent() == (
        "shake_movie/0 (+https://github.com/ucgmsim/shake_movie)"
    )


class FakeUrlopen:
    """Stands in for urllib's urlopen, recording each request."""

    def __init__(self, body: bytes = b"{}", error: Exception | None = None) -> None:
        """Answer every request with `body`, or raise `error` if given."""
        self.body = body
        self.error = error
        self.requests: list[tuple[urllib.request.Request, object]] = []

    def __call__(self, request: urllib.request.Request, timeout: object) -> io.BytesIO:
        """Record the request and its timeout, then answer or fail."""
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.body)


def test_get_json_identifies_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests go to the URL with the tool's User-Agent, JSON Accept and timeout."""
    fake = FakeUrlopen(json.dumps(NOMINATIM).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert naming._get_json("https://example.org/a?b=1") == NOMINATIM
    [(request, timeout)] = fake.requests
    assert request.full_url == "https://example.org/a?b=1"
    assert request.get_header("User-agent") == naming.user_agent()
    assert request.get_header("Accept") == "application/json"
    assert timeout == naming.TIMEOUT


def test_geonet_asks_for_geojson(monkeypatch: pytest.MonkeyPatch) -> None:
    """GeoNet is asked for its versioned GeoJSON and its answer is parsed."""
    fake = FakeUrlopen(json.dumps(GEONET).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    event = naming.geonet_event("2026p711868")
    assert event is not None
    time, locality = event
    assert locality == "5 km west of Masterton"
    assert time == datetime.datetime(
        2026, 9, 21, 9, 23, 42, 309000, tzinfo=datetime.UTC
    )
    [(request, _)] = fake.requests
    assert request.full_url == naming.GEONET_QUAKE_URL.format("2026p711868")
    assert request.get_header("Accept") == "application/vnd.geo+json;version=2"


@pytest.mark.parametrize(
    "fake",
    [
        FakeUrlopen(error=urllib.error.URLError("no route to host")),
        FakeUrlopen(error=TimeoutError("timed out")),
        FakeUrlopen(b"<html>Service Unavailable</html>"),
        FakeUrlopen(error=http.client.RemoteDisconnected("closed")),
        FakeUrlopen(error=http.client.IncompleteRead(b"")),
        FakeUrlopen(error=ConnectionResetError("reset")),
        FakeUrlopen(b"\x80not utf-8"),
    ],
    ids=[
        "unreachable",
        "timeout",
        "not-json",
        "disconnected",
        "incomplete-read",
        "reset",
        "not-utf-8",
    ],
)
def test_get_json_failures_give_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fake: FakeUrlopen
) -> None:
    """An unreachable, slow or broken service gives None and a warning."""
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with caplog.at_level(logging.WARNING, logger=naming.__name__):
        assert naming._get_json("https://example.org/") is None
    assert "Could not reach https://example.org/" in caplog.text


def test_get_json_warning_names_the_url_and_the_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning says exactly which URL failed and why."""
    fake = FakeUrlopen(error=ConnectionResetError("connection reset by peer"))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with caplog.at_level(logging.WARNING, logger=naming.__name__):
        assert naming._get_json("https://example.org/") is None
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "Could not reach https://example.org/: connection reset by peer"
    )


def test_geonet_event_without_a_locality(monkeypatch: pytest.MonkeyPatch) -> None:
    """An event GeoNet has not yet placed gives an empty locality, not None."""
    feature = {"properties": {"time": "2026-09-21T09:23:42Z"}}
    monkeypatch.setattr(
        naming, "_get_json", lambda url, accept="": {"features": [feature]}
    )
    event = naming.geonet_event("2026p711868")
    assert event is not None
    time, locality = event
    assert locality == ""
    assert time.tzinfo == naming.NZ_TIME
    assert time == datetime.datetime(2026, 9, 21, 9, 23, 42, tzinfo=datetime.UTC)


@pytest.mark.parametrize(
    "document", [None, {}, {"features": []}, {"features": None}], ids=str
)
def test_geonet_event_without_features(
    monkeypatch: pytest.MonkeyPatch, document: dict | None
) -> None:
    """No answer, or an answer with no event, gives None."""
    monkeypatch.setattr(naming, "_get_json", lambda url, accept="": document)
    assert naming.geonet_event("2026p711868") is None


def test_reverse_geocode_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nominatim is asked for JSON v2 at district zoom, in English, at the point."""
    calls = []

    def fake(url: str, accept: str = "application/json") -> dict:
        calls.append(url)
        return NOMINATIM

    monkeypatch.setattr(naming, "_get_json", fake)
    assert (
        naming.reverse_geocode(172.6362254, -42.5263711)
        == "Hurunui District, Canterbury"
    )
    [url] = calls
    base, _, query = url.partition("?")
    assert base == naming.NOMINATIM_URL
    assert urllib.parse.parse_qs(query, strict_parsing=True) == {
        "format": ["jsonv2"],
        "lat": ["-42.52637"],
        "lon": ["172.63623"],
        "zoom": ["10"],
        "accept-language": ["en"],
    }


@pytest.mark.parametrize(
    ("document", "text"),
    [
        (
            {"address": {"town": "Hanmer Springs", "county": "Hurunui District"}},
            "Hanmer Springs, Hurunui District",
        ),
        (
            {
                "address": {
                    "city": "Christchurch",
                    "county": "Christchurch City",
                    "state": "Canterbury",
                }
            },
            "Christchurch, Christchurch City",
        ),
        (
            {"address": {"city": "Auckland", "state": "Auckland"}},
            "Auckland",
        ),
        ({"address": {"state": "Canterbury"}}, "Canterbury"),
        ({"address": {"country": "New Zealand"}}, None),
        ({"address": {}}, None),
        ({"error": "Unable to geocode"}, None),
        (None, None),
    ],
    ids=[
        "town-district",
        "first-two",
        "deduplicated",
        "region-only",
        "country-only",
        "empty-address",
        "at-sea",
        "unreachable",
    ],
)
def test_reverse_geocode_wording(
    monkeypatch: pytest.MonkeyPatch, document: dict | None, text: str | None
) -> None:
    """The two most local names are joined once each; nothing local gives None."""
    monkeypatch.setattr(naming, "_get_json", lambda url, accept="": document)
    assert naming.reverse_geocode(172.6, -42.5) == text


def test_reverse_geocode_names_a_village(monkeypatch: pytest.MonkeyPatch) -> None:
    """A village counts as the most local name, ahead of its district."""
    document = {
        "address": {
            "village": "Cheviot",
            "county": "Hurunui District",
            "state": "Canterbury",
        }
    }
    monkeypatch.setattr(naming, "_get_json", lambda url, accept="": document)
    assert naming.reverse_geocode(173.27, -42.81) == "Cheviot, Hurunui District"


SPREAD_TOWNS = [
    Town("Masterton", 20900, 1824000.0, 5465000.0),
    Town("Westville", 5000, 1810000.0, 5465000.0),
    Town("Southburn", 5000, 1824000.0, 5440000.0),
]


@pytest.mark.parametrize(
    ("point", "text"),
    [
        ((1824000.0, 5469000.0), "4 km north of Masterton"),
        ((1810000.0, 5469000.0), "4 km north of Westville"),
        ((1824000.0, 5436000.0), "4 km south of Southburn"),
        ((1806000.0, 5465000.0), "4 km west of Westville"),
    ],
)
def test_nearest_town_is_the_closest(point: tuple[float, float], text: str) -> None:
    """The place is given from whichever town is closest in both directions."""
    assert naming.nearest_town(np.array(point), SPREAD_TOWNS) == text


@pytest.mark.parametrize(
    ("population", "text"),
    [(1000, "10 km east of Ruralton"), (999, "Near Hamlet")],
)
def test_nearest_town_prefers_towns_of_a_thousand(population: int, text: str) -> None:
    """Towns of 1000 people or more are preferred over closer, smaller ones."""
    towns = [
        Town("Ruralton", population, 1814000.0, 5465000.0),
        Town("Hamlet", 50, 1824000.0, 5465000.0),
    ]
    assert naming.nearest_town(np.array([1824000.0, 5465000.0]), towns) == text


def test_nearest_town_among_only_small_towns() -> None:
    """With no sizeable town, the nearest small town is used."""
    towns = [
        Town("Hamlet", 50, 1824000.0, 5465000.0),
        Town("Farside", 200, 1900000.0, 5465000.0),
    ]
    point = np.array([1824000.0, 5459000.0])
    assert naming.nearest_town(point, towns) == "6 km south of Hamlet"


@pytest.mark.parametrize(
    ("distance", "text"),
    [
        (2999.0, "Near Masterton"),
        (3000.0, "3 km north of Masterton"),
        (12510.0, "13 km north of Masterton"),
        (12490.0, "12 km north of Masterton"),
        (19400.0, "19 km north of Masterton"),
        (21000.0, "20 km north of Masterton"),
        (23000.0, "25 km north of Masterton"),
    ],
)
def test_nearest_town_distance_rounding(distance: float, text: str) -> None:
    """Under 3 km is "near"; then whole km, and 5 km steps from 20 km."""
    point = np.array([1824000.0, 5465000.0 + distance])
    assert naming.nearest_town(point, TOWNS) == text


@pytest.mark.parametrize(
    ("bearing", "direction"),
    [
        (0.0, "north"),
        (22.0, "north"),
        (23.0, "north-east"),
        (67.0, "north-east"),
        (68.0, "east"),
        (157.0, "south-east"),
        (203.0, "south-west"),
        (292.0, "west"),
        (293.0, "north-west"),
        (337.0, "north-west"),
        (338.0, "north"),
        (340.0, "north"),
    ],
)
def test_nearest_town_compass_points(bearing: float, direction: str) -> None:
    """Bearings fall in 45-degree sectors centred on each compass point."""
    angle = np.radians(bearing)
    point = np.array(
        [1824000.0 + 10000.0 * np.sin(angle), 5465000.0 + 10000.0 * np.cos(angle)]
    )
    assert naming.nearest_town(point, TOWNS) == f"10 km {direction} of Masterton"
