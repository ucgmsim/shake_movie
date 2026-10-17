"""Tests for naming the earthquake. Services are stubbed: no network."""

import datetime

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
