from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import (
    ACCIDENTS_CSV_PATH,
    ACCIDENTS_PARQUET,
    JOINED_PARQUET,
    MAX_ACCIDENT_ROWS,
    PROCESSED_DIR,
    SQLITE_PATH,
    WEATHER_CSV_PATH,
    WEATHER_PARQUET,
    YEAR_MAX,
    YEAR_MIN,
)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(WEATHER_CSV_PATH)
    df.columns = [c.split(" (")[0].strip().lower() for c in df.columns]
    time_col = _pick(df, ["time", "date", "datetime"]) or df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df["date"] = df[time_col].dt.date

    temp_col = _pick(df, ["temperature_2m", "temp", "tavg"])
    prcp_col = _pick(df, ["precipitation", "precip", "prcp"])
    rain_col = _pick(df, ["rain"])
    cloud_col = _pick(df, ["cloudcover"])
    wind_col = _pick(df, ["windspeed_10m", "wind", "windspeed"])

    df["temp_c"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan
    df["precip_mm"] = pd.to_numeric(df[prcp_col], errors="coerce") if prcp_col else 0.0
    df["rain_mm"] = pd.to_numeric(df[rain_col], errors="coerce") if rain_col else 0.0
    df["cloud_pct"] = pd.to_numeric(df[cloud_col], errors="coerce") if cloud_col else np.nan
    df["wind_kmh"] = pd.to_numeric(df[wind_col], errors="coerce") if wind_col else np.nan

    daily = df.groupby("date").agg(
        temp_max=("temp_c", "max"),
        temp_min=("temp_c", "min"),
        temp_mean=("temp_c", "mean"),
        precipitation=("precip_mm", "sum"),
        rain_mm=("rain_mm", "sum"),
        cloud_pct=("cloud_pct", "mean"),
        wind_kmh_max=("wind_kmh", "max"),
    ).reset_index()

    daily["temp_max"] = daily["temp_max"] * 9.0 / 5.0 + 32.0
    daily["temp_min"] = daily["temp_min"] * 9.0 / 5.0 + 32.0
    daily["temp_mean"] = daily["temp_mean"] * 9.0 / 5.0 + 32.0

    daily["weather_description"] = np.where(
        daily["precipitation"] > 1.0, "Rainy",
        np.where(daily["cloud_pct"] > 60, "Cloudy", "Clear"),
    )
    daily["precipitation_type"] = np.where(daily["temp_min"] <= 32, "Snow/Mix", "Rain")
    daily.insert(0, "weather_id", np.arange(1, len(daily) + 1))
    return daily


def load_accidents() -> pd.DataFrame:
    df = pd.read_csv(ACCIDENTS_CSV_PATH, low_memory=False)
    df.columns = [c.upper().replace(" ", "_") for c in df.columns]

    required = [
        "COLLISION_ID",
        "CRASH_DATE",
        "CRASH_TIME",
        "BOROUGH",
        "ZIP_CODE",
        "LATITUDE",
        "LONGITUDE",
        "ON_STREET_NAME",
        "OFF_STREET_NAME",
        "NUMBER_OF_PERSONS_INJURED",
        "NUMBER_OF_PERSONS_KILLED",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Crash CSV missing required columns: {missing}")

    df["CRASH_DATE_PARSED"] = pd.to_datetime(df["CRASH_DATE"], errors="coerce")
    df = df.dropna(subset=["CRASH_DATE_PARSED"])

    if YEAR_MIN is not None:
        df = df[df["CRASH_DATE_PARSED"].dt.year >= YEAR_MIN]
    if YEAR_MAX is not None:
        df = df[df["CRASH_DATE_PARSED"].dt.year <= YEAR_MAX]

    df["CRASH_TIME"] = df["CRASH_TIME"].astype(str).str.strip()
    df["CRASH_DATETIME"] = pd.to_datetime(
        df["CRASH_DATE_PARSED"].dt.strftime("%Y-%m-%d") + " " + df["CRASH_TIME"],
        errors="coerce",
    )
    df = df.dropna(subset=["CRASH_DATETIME"])

    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    df = df[(df["LATITUDE"].between(40.4, 41.0)) & (df["LONGITUDE"].between(-74.3, -73.6))]

    df["num_injuries"] = df["NUMBER_OF_PERSONS_INJURED"].fillna(0).astype(int)
    df["num_deaths"] = df["NUMBER_OF_PERSONS_KILLED"].fillna(0).astype(int)
    df["severity"] = np.where(
        (df["num_deaths"] > 0) | (df["num_injuries"] >= 3), 1, 0
    )

    df["ON_STREET_NAME"] = df["ON_STREET_NAME"].fillna("").astype(str)
    df["OFF_STREET_NAME"] = df["OFF_STREET_NAME"].fillna("").astype(str)
    df["street_name"] = df["ON_STREET_NAME"]
    empty = df["street_name"].str.strip().eq("")
    df.loc[empty, "street_name"] = df.loc[empty, "OFF_STREET_NAME"]
    df["street_name"] = df["street_name"].replace("", "Unknown")

    df["crash_date"] = df["CRASH_DATE_PARSED"].dt.date
    df = df.drop_duplicates(subset=["COLLISION_ID"])

    if MAX_ACCIDENT_ROWS is not None and len(df) > MAX_ACCIDENT_ROWS:
        df = df.sample(n=MAX_ACCIDENT_ROWS, random_state=0)

    return df.rename(
        columns={
            "COLLISION_ID": "accident_id",
            "CRASH_DATETIME": "crash_datetime",
            "BOROUGH": "borough",
            "ZIP_CODE": "zip_code",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )[
        [
            "accident_id",
            "crash_datetime",
            "crash_date",
            "borough",
            "zip_code",
            "latitude",
            "longitude",
            "street_name",
            "num_injuries",
            "num_deaths",
            "severity",
        ]
    ].reset_index(drop=True)


def join_weather(accidents: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    by_date = weather.set_index("date")
    joined = accidents.merge(
        by_date,
        left_on="crash_date",
        right_index=True,
        how="inner",
    )
    return joined.sort_values("crash_datetime").reset_index(drop=True)


def write_sqlite(accidents: pd.DataFrame, weather: pd.DataFrame, joined: pd.DataFrame) -> None:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    weather.to_sql("weather", engine, if_exists="replace", index=False)
    accidents.to_sql("accidents", engine, if_exists="replace", index=False)
    joined.to_sql("accidents_joined", engine, if_exists="replace", index=False)
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_acc_dt ON accidents(crash_datetime);
            CREATE INDEX IF NOT EXISTS idx_acc_sev ON accidents(severity);
            CREATE INDEX IF NOT EXISTS idx_acc_lat_lon ON accidents(latitude, longitude);
            CREATE INDEX IF NOT EXISTS idx_acc_borough_date ON accidents(borough, crash_date);
            """
        )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[etl] loading weather...")
    weather = load_weather()
    print(f"[etl]   {len(weather):,} weather rows")

    print("[etl] loading accidents...")
    accidents = load_accidents()
    print(f"[etl]   {len(accidents):,} accident rows")

    print("[etl] joining...")
    joined = join_weather(accidents, weather)
    print(f"[etl]   {len(joined):,} joined rows")

    weather.to_parquet(WEATHER_PARQUET, index=False)
    accidents.to_parquet(ACCIDENTS_PARQUET, index=False)
    joined.to_parquet(JOINED_PARQUET, index=False)

    print("[etl] writing sqlite cache...")
    write_sqlite(accidents, weather, joined)

    print(
        f"[etl] done. severity=1 rate: "
        f"{joined['severity'].mean():.4f} ({int(joined['severity'].sum()):,} of {len(joined):,})"
    )


if __name__ == "__main__":
    main()
