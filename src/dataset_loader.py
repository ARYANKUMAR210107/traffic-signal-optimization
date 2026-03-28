import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_traffic_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the Metro Interstate traffic dataset, compute a normalized delay column,
    and add an hour-of-day column for peak-hour modelling.
    """
    df = pd.read_csv(csv_path, parse_dates=["date_time"])
    if "traffic_volume" not in df.columns:
        raise ValueError("Expected column 'traffic_volume' in traffic dataset")

    max_vol = df["traffic_volume"].max()
    df["delay"] = df["traffic_volume"] / max_vol
    df["hour"] = df["date_time"].dt.hour
    return df


def delay_lookup_by_hour(df: pd.DataFrame) -> Dict[int, float]:
    """
    Build a lookup table: hour -> mean delay factor (0..1).
    """
    grouped = df.groupby("hour")["delay"].mean()
    # Ensure every hour present
    lookup = {hour: float(grouped.get(hour, grouped.mean())) for hour in range(24)}
    return lookup


def load_ambulance_incident(csv_path: str | Path, seed: int = 42) -> Tuple[float, float]:
    """
    Load ambulance/911 dataset and return a random incident (lat, lon).
    Tries common column names: ('lat','lng') or ('latitude','longitude').
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    for lat_col, lon_col in (("lat", "lng"), ("latitude", "longitude"), ("LATITUDE", "LONGITUDE")):
        if lat_col in df.columns and lon_col in df.columns:
            latitudes = df[lat_col].dropna().to_numpy()
            longitudes = df[lon_col].dropna().to_numpy()
            if len(latitudes) == 0:
                continue
            idx = rng.integers(0, len(latitudes))
            return float(latitudes[idx]), float(longitudes[idx])

    raise ValueError("Could not find latitude/longitude columns in ambulance dataset")


__all__ = [
    "load_traffic_data",
    "delay_lookup_by_hour",
    "load_ambulance_incident",
]
