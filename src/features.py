from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES_PARQUET, JOINED_PARQUET


def _cyclical(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * series / period
    return pd.DataFrame({f"{prefix}_sin": np.sin(radians), f"{prefix}_cos": np.cos(radians)})


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["crash_datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["is_night"] = (df["hour"] < 6) | (df["hour"] >= 22)
    df["is_night"] = df["is_night"].astype(int)
    df = pd.concat(
        [
            df.reset_index(drop=True),
            _cyclical(df["hour"].astype(float), 24, "hour"),
            _cyclical(df["day_of_week"].astype(float), 7, "dow"),
            _cyclical(df["month"].astype(float), 12, "month"),
        ],
        axis=1,
    )
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["precipitation"] = df["precipitation"].fillna(0.0)
    df["rain_mm"] = df.get("rain_mm", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["temp_max"] = df["temp_max"].fillna(df["temp_max"].median())
    df["temp_min"] = df["temp_min"].fillna(df["temp_min"].median())
    df["temp_mean"] = df.get("temp_mean", (df["temp_max"] + df["temp_min"]) / 2).fillna(
        (df["temp_max"] + df["temp_min"]) / 2
    )
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    if "cloud_pct" in df.columns:
        df["cloud_pct"] = df["cloud_pct"].fillna(df["cloud_pct"].median())
    else:
        df["cloud_pct"] = 50.0
    if "wind_kmh_max" in df.columns:
        df["wind_kmh_max"] = df["wind_kmh_max"].fillna(df["wind_kmh_max"].median())
    else:
        df["wind_kmh_max"] = df.get("wind_kmh", pd.Series(0.0, index=df.index)).fillna(0.0)

    df["is_rainy"] = (df["precipitation"] > 1.0).astype(int)
    df["is_heavy_rain"] = (df["precipitation"] > 10.0).astype(int)
    df["is_freezing"] = (df["temp_min"] <= 32).astype(int)
    df["is_hot"] = (df["temp_max"] >= 85).astype(int)
    df["is_cloudy"] = (df["cloud_pct"] > 60).astype(int)
    df["is_windy"] = (df["wind_kmh_max"] > 30).astype(int)
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    times_square = (40.7580, -73.9855)
    df["dist_to_midtown_km"] = (
        np.sqrt(
            (111.0 * (df["latitude"] - times_square[0])) ** 2
            + (
                111.0
                * np.cos(np.radians(times_square[0]))
                * (df["longitude"] - times_square[1])
            )
            ** 2
        )
    )
    df["lat_bin"] = (df["latitude"] * 100).round().astype(int)
    df["lon_bin"] = (df["longitude"] * 100).round().astype(int)
    return df


def build_features(joined: pd.DataFrame | None = None) -> pd.DataFrame:
    if joined is None:
        joined = pd.read_parquet(JOINED_PARQUET)
    joined["crash_datetime"] = pd.to_datetime(joined["crash_datetime"])
    out = add_time_features(joined)
    out = add_weather_features(out)
    out = add_spatial_features(out)
    out["borough"] = out["borough"].fillna("UNKNOWN").astype(str)
    out["zip_code"] = out["zip_code"].fillna("UNKNOWN").astype(str)
    out["weather_description"] = out["weather_description"].fillna("Unknown").astype(str)
    return out.sort_values("crash_datetime").reset_index(drop=True)


def main() -> None:
    print("[features] reading joined parquet...")
    df = build_features()
    df.to_parquet(FEATURES_PARQUET, index=False)
    print(f"[features] wrote {len(df):,} rows -> {FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
