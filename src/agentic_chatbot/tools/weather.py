"""Current weather lookup backed by Open-Meteo."""

from typing import Any

import httpx
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10.0

WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError("The weather service is currently unavailable.") from error


@tool
def get_current_weather(location: str) -> dict[str, Any]:
    """Get current weather for a city or location, such as 'Tunis' or 'Paris, France'."""

    location = location.strip()
    if len(location) < 2:
        raise ValueError("Provide a city or location with at least two characters.")
    if len(location) > 100:
        raise ValueError("Location is too long.")

    geocoding = _get_json(
        GEOCODING_URL,
        {"name": location, "count": 1, "language": "en", "format": "json"},
    )
    matches = geocoding.get("results", [])
    if not matches:
        raise ValueError(f"No location found for '{location}'.")

    place = matches[0]
    forecast = _get_json(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m"
            ),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        },
    )
    current = forecast["current"]
    weather_code = int(current["weather_code"])

    return {
        "location": {
            "name": place["name"],
            "country": place.get("country", ""),
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        },
        "current": {
            "observed_at": current["time"],
            "condition": WEATHER_DESCRIPTIONS.get(weather_code, "Unknown"),
            "weather_code": weather_code,
            "temperature_c": current["temperature_2m"],
            "apparent_temperature_c": current["apparent_temperature"],
            "relative_humidity_percent": current["relative_humidity_2m"],
            "wind_speed_kmh": current["wind_speed_10m"],
        },
    }
