from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ACCIDENTS_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_crashes.csv"
WEATHER_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_weather_daily.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ACCIDENTS_PARQUET = PROCESSED_DIR / "accidents.parquet"
WEATHER_PARQUET = PROCESSED_DIR / "weather.parquet"
JOINED_PARQUET = PROCESSED_DIR / "accidents_joined.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
SQLITE_PATH = PROCESSED_DIR / "traffic.db"

METRICS_DIR = PROCESSED_DIR / "metrics"
MODELS_DIR = PROCESSED_DIR / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

YEAR_MIN = 2021
YEAR_MAX = 2022
MAX_ACCIDENT_ROWS = None

RANDOM_STATE = 42
TEST_FRACTION = 0.2
KMEANS_N_CLUSTERS = 4
DBSCAN_EPS_KM = 0.5
DBSCAN_MIN_SAMPLES = 50

EARTH_RADIUS_KM = 6371.0088
