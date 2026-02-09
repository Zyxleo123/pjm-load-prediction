import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from load import load_pjm_csv, create_dataset

TARGET_ZONE = 'PE'
LAT = 39.96
LON = -75.60

CURRENT_YEAR = None
CURRENT_DF = None
CURRENT_WEATHER_SOURCE = None

# --- Utility: Directory Setup & Save ---
def save_plot(fig, plot_name, year_label):
    """Saves the current figure to plots/{plot_name}/{year_label}.png"""
    base_dir = "plots"
    target_dir = os.path.join(base_dir, plot_name)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    filename = f"{year_label}.png"
    filepath = os.path.join(target_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig) # Close to free memory

def _load_pjm_data(year_label, zone='PE'):
    """Loads PJM data for a given year label (int/str)."""
    global CURRENT_YEAR
    global CURRENT_DF

    if not isinstance(year_label, (int, str)):
        raise TypeError("year_label must be an int or str")

    year_label = str(year_label)
    if CURRENT_YEAR == year_label and CURRENT_DF is not None:
        return CURRENT_DF, year_label

    if year_label == "combined":
        filepath = "hrl_load_metered_combined.csv"
    else:
        filepath = f"hrl_load_metered_{year_label}.csv"
    df = load_pjm_csv(filepath, target_zone=zone)
    CURRENT_YEAR = year_label
    CURRENT_DF = df
    return df, year_label

def _load_dataset_with_weather(year_label, zone='PE', lat=LAT, lon=LON, weather_source='openmeteo'):
    """Loads PJM + weather data for a given year label (int/str)."""
    global CURRENT_YEAR
    global CURRENT_DF
    global CURRENT_WEATHER_SOURCE

    if not isinstance(year_label, (int, str)):
        raise TypeError("year_label must be an int or str")

    year_label = str(year_label)
    if CURRENT_YEAR == year_label and CURRENT_DF is not None and CURRENT_WEATHER_SOURCE == weather_source:
        return CURRENT_DF, year_label

    if year_label == "combined":
        filepath = "hrl_load_metered_combined.csv"
    else:
        filepath = f"hrl_load_metered_{year_label}.csv"
    df = create_dataset(filepath, zone, lat, lon, weather_source=weather_source)
    CURRENT_YEAR = year_label
    CURRENT_DF = df
    CURRENT_WEATHER_SOURCE = weather_source
    return df, year_label

# --- Plot Function 1: Basic Time Series ---
def plot_time_series(year_label, target_col='load_mw', zone='PE'):
    df, year_label = _load_pjm_data(year_label, zone=zone)
    if df.empty:
        print(f"No data for {year_label}")
        return

    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[target_col], label=f'{zone} Load', color='navy', linewidth=0.5)
    plt.title(f'Hourly Energy Demand for {zone} ({year_label})')
    plt.ylabel('Megawatts (MW)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "time_series", year_label)

# --- Plot Function 2: Temperature vs Load ---
def plot_temp_vs_load(year_label, target_col='load_mw', temp_col='temp', zone='PE'):
    df, year_label = _load_dataset_with_weather(year_label, zone=zone)
    if df.empty or temp_col not in df.columns:
        print(f"No data or missing temp column for {year_label}")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[temp_col], y=df[target_col], alpha=0.1, s=10, color='darkorange')
    plt.title(f'Temperature vs. Load for {zone} ({year_label})')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Load (MW)')
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "temp_vs_load", year_label)

def plot_temp_and_load_dual_axis(year_label, target_col='load_mw', temp_col='temp', zone='PE'):
    """Plot temperature and load on two y-axes to compare trends."""
    df, year_label = _load_dataset_with_weather(year_label, zone=zone)
    if df.empty or temp_col not in df.columns:
        print(f"No data or missing temp column for {year_label}")
        return

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()

    ax1.plot(df.index, df[target_col], color='navy', linewidth=0.7, label='Load (MW)')
    ax2.plot(df.index, df[temp_col], color='darkorange', linewidth=0.7, label='Temperature (C)')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Load (MW)', color='navy')
    ax2.set_ylabel('Temperature (C)', color='darkorange')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    plt.title(f'Load and Temperature Trends ({zone}, {year_label})')
    ax1.grid(True, alpha=0.3)

    save_plot(fig, "temp_load_dual_axis", year_label)

# --- Plot Function 3: Daily Profile by Day of Week ---
def plot_daily_profile(year_label, target_col='load_mw'):
    df, year_label = _load_pjm_data(year_label)
    if df.empty:
        print(f"No data for {year_label}")
        return

    data = df.copy()
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek 

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x='hour', y=target_col, hue='day_of_week', palette='viridis')
    plt.title(f'Average Daily Load Profile by Day of Week ({year_label})')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Load (MW)')
    plt.grid(True)
    plt.legend(title='Day (0=Mon, 6=Sun)')
    
    save_plot(plt.gcf(), "daily_profile", year_label)

# --- Plot Function 4: Stationarity / Rolling Stats ---
def plot_rolling_stats(year_label, target_col='load_mw', window_size=168):
    df, year_label = _load_pjm_data(year_label)
    if df.empty: return

    target = df[target_col]
    rolling_mean = target.rolling(window=window_size).mean()
    rolling_std = target.rolling(window=window_size).std()

    plt.figure(figsize=(14, 6))
    plt.plot(target, color='blue', alpha=0.3, label='Original Hourly Load')
    plt.plot(rolling_mean, color='red', linewidth=2, label=f'Rolling Mean ({window_size}h)')
    plt.plot(rolling_std, color='black', linewidth=2, label=f'Rolling Std ({window_size}h)')
    plt.title(f'Stationarity Check: Rolling Mean & Std ({year_label})')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "rolling_stats", year_label)

# --- Plot Function 5: ACF Plot ---
def plot_acf_custom(year_label, target_col='load_mw', lags=504):
    df, year_label = _load_pjm_data(year_label)
    if df.empty: return

    plt.figure(figsize=(14, 6))
    plot_acf(df[target_col].dropna(), lags=lags, alpha=0.05, title=f'ACF ({year_label})', ax=plt.gca())
    plt.xlabel('Lag (Hours)')
    plt.ylabel('Autocorrelation')
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "acf", year_label)

# --- Plot Function 6: Seasonal Decomposition ---
def plot_decomposition(year_label, target_col='load_mw', period=24):
    df, year_label = _load_pjm_data(year_label)
    if df.empty: return
    
    # Decomposition requires continuous data without NaNs
    target = df[target_col].dropna()
    if len(target) < period * 2:
        print(f"Not enough data for decomposition for {year_label}")
        return

    decomposition = seasonal_decompose(target, model='additive', period=period)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(f'Classical Decomposition (Period={period}h) - {year_label}', y=1.02)
    
    save_plot(fig, "decomposition", year_label)

# --- Plot Function 7: Distribution Shift (Only for Combined usually, but adaptable) ---
def plot_distribution_density(year_label, target_col='load_mw'):
    df, year_label = _load_pjm_data(year_label)
    if df.empty: return

    data = df.copy()
    data['year_label'] = data.index.year
    
    plt.figure(figsize=(14, 6))
    # If year is singular, just show that year's density. If combined, hue by year.
    hue_param = 'year_label' if year_label == "combined" else None
    
    sns.kdeplot(data=data, x=target_col, hue=hue_param, palette='viridis', common_norm=False, linewidth=1.5)
    plt.title(f'Load Density Distribution ({year_label})')
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "distribution_density", year_label)


# --- MASTER RUN FUNCTION ---
def run_all_plots(years_to_plot=('combined', 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)):
    if not isinstance(years_to_plot, (list, tuple)):
        years_to_plot = [years_to_plot]

    for year_label in years_to_plot:
        print(f"\nGeneratings plots for: {year_label}")
        plot_time_series(year_label)
        plot_temp_vs_load(year_label)
        plot_daily_profile(year_label)
        plot_rolling_stats(year_label)
        plot_acf_custom(year_label)
        if str(year_label) != "combined":
             plot_decomposition(year_label)
        plot_distribution_density(year_label)

def plot_monthly_avg_by_year(years=range(2016, 2026), zone='PE', target_col='load_mw'):
    """Plot 12 subplots: each month shows average load across years as a 10-point curve."""
    monthly_by_year = {}

    for year in years:
        df, year_label = _load_pjm_data(year, zone=zone)
        if df.empty:
            continue

        monthly_mean = df[target_col].resample('M').mean()
        monthly_by_year[int(year_label)] = monthly_mean.groupby(monthly_mean.index.month).mean()

    if not monthly_by_year:
        print("No data available for monthly comparison.")
        return

    years_sorted = sorted(monthly_by_year.keys())

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for month in range(1, 13):
        ax = axes[month - 1]
        values = [monthly_by_year[year].get(month, float('nan')) for year in years_sorted]

        ax.plot(years_sorted, values, marker='o', linewidth=1)
        ax.set_title(f"Month {month}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Average Monthly Load by Year", y=1.02)
    fig.tight_layout()

    save_plot(fig, "monthly_avg_by_year", "combined")