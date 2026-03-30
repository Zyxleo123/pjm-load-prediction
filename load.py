"""
Load PJM hourly load from CSV and align Open-Meteo (ERA5 archive) weather.

``locations`` for ``create_dataset`` can be:
  * dict mapping station name -> (lat, lon) — columns get ``{name}_temp`` etc.
  * (lat, lon) tuple — single site, columns stay ``temp``, ``rh``, …
  * lat, lon as 3rd and 4th positional args (legacy) — same as a single tuple.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Tuple, Union

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

DATA_ROOT = "data"
RAW_SUBDIR = "raw"

LocationDict = Mapping[str, Tuple[float, float]]
LocationSpec = Union[LocationDict, Tuple[float, float]]

_openmeteo_client: openmeteo_requests.Client | None = None


def raw_csv_path(filename: str) -> str:
    return os.path.join(DATA_ROOT, RAW_SUBDIR, filename)


def _get_openmeteo_client() -> openmeteo_requests.Client:
    global _openmeteo_client
    if _openmeteo_client is None:
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        _openmeteo_client = openmeteo_requests.Client(session=retry_session)
    return _openmeteo_client


def _eastern_naive_index(utc_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC-aware index -> US/Eastern wall time, timezone-naive (matches PJM CSV)."""
    return utc_index.tz_convert("US/Eastern").tz_localize(None)


def _station_prefix(name: str) -> str:
    return name.lower().replace(" ", "_")


def _location_jobs(
    locations: LocationSpec | float, lon: float | None = None
) -> List[Tuple[str, float, float]]:
    """
    Return [(column_prefix, lat, lon), ...]. Empty prefix means do not add a column prefix.
    """
    if lon is not None:
        return [("", float(locations), float(lon))]

    if isinstance(locations, dict):
        jobs: List[Tuple[str, float, float]] = []
        for name, latlon in locations.items():
            lat, lo = latlon
            prefix = _station_prefix(name) if name else ""
            jobs.append((prefix, float(lat), float(lo)))
        return jobs

    if (
        isinstance(locations, tuple)
        and len(locations) == 2
        and all(isinstance(x, (int, float)) for x in locations)
    ):
        lat, lo = locations
        return [("", float(lat), float(lo))]

    raise TypeError(
        "locations must be dict[name -> (lat, lon)], a (lat, lon) tuple, "
        "or pass lat and lon as the third and fourth arguments"
    )


def _apply_station_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not prefix:
        return df
    return df.add_prefix(f"{prefix}_")


def join_weather_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("join_weather_frames: no frames")
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="inner")
    return out


def load_pjm_csv(filepath: str, target_zone: str = "PE") -> pd.DataFrame:
    """Read raw PJM CSV, filter zone, aggregate to hourly zone total load."""
    print(f"Loading PJM data from {filepath}...")

    df = pd.read_csv(raw_csv_path(filepath))

    if "zone" in df.columns:
        df = df[df["zone"] == target_zone].copy()
    else:
        print(f"Warning: 'zone' column not found. Available columns: {df.columns}")

    df["timestamp"] = pd.to_datetime(df["datetime_beginning_ept"])
    df_clean = df.groupby("timestamp")["mw"].sum().reset_index()
    df_clean.rename(columns={"mw": "load_mw"}, inplace=True)
    df_clean.set_index("timestamp", inplace=True)

    print(f"Loaded {len(df_clean)} rows for Zone: {target_zone}")
    return df_clean


def fetch_weather(
    start_date: pd.Timestamp, end_date: pd.Timestamp, lat: float, lon: float
) -> pd.DataFrame:
    """Hourly ERA5 archive from Open-Meteo; Eastern-naive index."""
    print("Fetching weather data via Open-Meteo (ERA5 Reanalysis)...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "wind_speed_10m",
        ],
        "timezone": "America/New_York",
    }

    client = _get_openmeteo_client()
    response = client.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    hourly_data = {
        "temp": hourly.Variables(0).ValuesAsNumpy(),
        "rh": hourly.Variables(1).ValuesAsNumpy(),
        "dwpt": hourly.Variables(2).ValuesAsNumpy(),
        "wspd": hourly.Variables(3).ValuesAsNumpy(),
    }
    weather_df = pd.DataFrame(hourly_data, index=date_range)
    weather_df.index = _eastern_naive_index(weather_df.index)

    print(f"Fetched {len(weather_df)} weather rows (Open-Meteo).")
    return weather_df


def create_dataset(
    pjm_filepath: str,
    target_zone: str,
    locations: LocationSpec | float,
    lon: float | None = None,
) -> pd.DataFrame:
    """
    Load PJM load, fetch Open-Meteo weather for each site, inner-join on timestamp.

    Parameters
    ----------
    locations
        Mapping ``station_name -> (lat, lon)`` for multi-station (prefixed columns),
        or ``(lat, lon)`` for a single site (unprefixed ``temp``, …).
    lon
        If given, ``locations`` is interpreted as *lat* (legacy 4-arg call).
    """
    df_load = load_pjm_csv(pjm_filepath, target_zone)
    if df_load.empty:
        raise ValueError(
            f"No data found for zone {target_zone}. Check your CSV or zone name."
        )

    start_date = df_load.index.min()
    end_date = df_load.index.max()

    jobs = _location_jobs(locations, lon)
    frames = []
    for prefix, lat, lo in jobs:
        df_w = fetch_weather(start_date, end_date, lat, lo)
        frames.append(_apply_station_prefix(df_w, prefix))

    df_weather = join_weather_frames(frames)
    return df_load.join(df_weather, how="inner")
