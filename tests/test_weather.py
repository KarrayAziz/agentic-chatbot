"""Tests for the Open-Meteo weather tool."""

from typing import Any

from agentic_chatbot.tools import weather


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_weather_geocodes_location_and_returns_structured_conditions(
    monkeypatch,
) -> None:
    responses = iter(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "name": "Tunis",
                            "country": "Tunisia",
                            "latitude": 36.8065,
                            "longitude": 10.1815,
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "current": {
                        "time": "2026-08-24T12:00",
                        "temperature_2m": 31.2,
                        "apparent_temperature": 33.0,
                        "relative_humidity_2m": 54,
                        "weather_code": 1,
                        "wind_speed_10m": 14.5,
                    }
                }
            ),
        ]
    )
    requested_urls: list[str] = []

    def fake_get(url: str, **kwargs):
        requested_urls.append(url)
        return next(responses)

    monkeypatch.setattr(weather.httpx, "get", fake_get)

    result = weather.get_current_weather.invoke({"location": "Tunis"})

    assert requested_urls == [weather.GEOCODING_URL, weather.FORECAST_URL]
    assert result["location"]["country"] == "Tunisia"
    assert result["current"] == {
        "observed_at": "2026-08-24T12:00",
        "condition": "Mainly clear",
        "weather_code": 1,
        "temperature_c": 31.2,
        "apparent_temperature_c": 33.0,
        "relative_humidity_percent": 54,
        "wind_speed_kmh": 14.5,
    }
