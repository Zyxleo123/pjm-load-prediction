import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import openmeteo_requests
import requests_cache
from retry_requests import retry

def load_pjm_csv(filepath, target_zone='PE'):
    """
    Reads the raw PJM CSV, filters for a specific zone, and cleans timestamps.
    """
    print(f"Loading PJM data from {filepath}...")
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # 1. Filter for the specific Zone (The "Pipe Owner")
    if 'zone' in df.columns:
        df = df[df['zone'] == target_zone].copy()
    else:
        print(f"Warning: 'zone' column not found. Available columns: {df.columns}")
    
    # 2. Parse Dates
    # We use 'datetime_beginning_ept' because it aligns with human behavior
    df['timestamp'] = pd.to_datetime(df['datetime_beginning_ept'])
    
    # 3. Aggregate
    # Sum sub-areas to get Zone total.
    df_clean = df.groupby('timestamp')['mw'].sum().reset_index()
    df_clean.rename(columns={'mw': 'load_mw'}, inplace=True)
    
    # Set index for merging later
    df_clean.set_index('timestamp', inplace=True)
    
    print(f"Loaded {len(df_clean)} rows for Zone: {target_zone}")
    return df_clean

def fetch_noaa_weather(start_date, end_date, lat, lon):
    """
    Original Meteostat implementation.
    """
    print("Fetching weather data via Meteostat (NOAA)...")
    
    location = Point(lat, lon)
    weather_df = Hourly(location, start_date, end_date)
    weather_df = weather_df.fetch()
    
    # Meteostat returns UTC index. Convert to Eastern Time.
    weather_df.index = weather_df.index.tz_localize('UTC')
    weather_df.index = weather_df.index.tz_convert('US/Eastern')
    weather_df.index = weather_df.index.tz_localize(None)
    
    # Select cols
    cols = ['temp', 'dwpt', 'wspd']
    available_cols = [c for c in cols if c in weather_df.columns]
    weather_df = weather_df[available_cols]
    
    # Fill gaps
    weather_df = weather_df.interpolate(method='linear')
    
    print(f"Fetched {len(weather_df)} weather rows (Meteostat).")
    return weather_df

def fetch_openmeteo_weather(start_date, end_date, lat, lon):
    """
    New Open-Meteo implementation using Reanalysis (ERA5) data.
    Guaranteed gap-free data for historical analysis.
    """
    print("Fetching weather data via Open-Meteo (ERA5 Reanalysis)...")

    # 1. Setup Client
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # 2. Prepare API Call
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Convert Timestamp objects to string "YYYY-MM-DD"
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        # Map variables to match Meteostat: Temp, Relative Humidity, Dew Point, Wind Speed
        "hourly": ["temperature_2m", "relative_humidity_2m",
                   "dew_point_2m", "wind_speed_10m"],
        "timezone": "America/New_York" # Handles DST automatically
    }

    # 3. Fetch Data
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # 4. Process into DataFrame
    hourly = response.Hourly()
    
    # Construct the time index
    date_range = pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )

    hourly_data = {
        "temp": hourly.Variables(0).ValuesAsNumpy(),       # temperature_2m
        "rh": hourly.Variables(1).ValuesAsNumpy(),         # relative_humidity_2m
        "dwpt": hourly.Variables(2).ValuesAsNumpy(),       # dew_point_2m
        "wspd": hourly.Variables(3).ValuesAsNumpy()        # wind_speed_10m
    }

    weather_df = pd.DataFrame(data = hourly_data, index=date_range)

    # 5. Timezone Alignment (Critical Step)
    # Open-Meteo returns UTC timestamps even if we asked for New_York timezone in the query.
    # However, the *values* align with the requested timezone logic, but the *index* needs conversion.
    
    # Convert UTC index -> Eastern Time
    weather_df.index = weather_df.index.tz_convert('US/Eastern')
    
    # Remove timezone info to match PJM's "naive" timestamp format
    weather_df.index = weather_df.index.tz_localize(None)

    print(f"Fetched {len(weather_df)} weather rows (Open-Meteo).")
    return weather_df

def create_dataset(pjm_filepath, target_zone, locations, weather_source='openmeteo'):
    """
    Master function to load PJM, load Weather for one or more locations, and join them.

    Args:
        locations: A list of (name, lat, lon) tuples identifying weather stations.
                   Example: [("Philadelphia", 39.95, -75.16), ("Harrisburg", 40.27, -76.88)]
                   Each location's weather columns are prefixed with its name, e.g. "philadelphia_temp".
        weather_source (str): 'openmeteo' (default) or 'meteostat'
    """
    # 1. Load Grid Data
    df_load = load_pjm_csv(pjm_filepath, target_zone)

    if df_load.empty:
        raise ValueError(f"No data found for zone {target_zone}. Check your CSV or zone name.")

    # 2. Determine date range
    start_date = df_load.index.min()
    end_date = df_load.index.max()

    # 3. Load Weather Data for each location and prefix columns
    fetch_fn = fetch_openmeteo_weather if weather_source == 'openmeteo' else fetch_noaa_weather

    weather_frames = []
    for name, lat, lon in locations:
        df_w = fetch_fn(start_date, end_date, lat, lon)
        prefix = name.lower().replace(" ", "_")
        df_w = df_w.add_prefix(f"{prefix}_")
        weather_frames.append(df_w)

    # 4. Merge all weather frames, then join with load
    df_weather_all = weather_frames[0].join(weather_frames[1:], how='inner') if len(weather_frames) > 1 else weather_frames[0]
    df_final = df_load.join(df_weather_all, how='inner')

    return df_final