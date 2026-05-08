from __future__ import annotations

import sqlite3

import pandas as pd

from config import METRICS_DIR, SQLITE_PATH


QUERIES = {
    "borough_severity": """
        SELECT borough,
               COUNT(*) AS total_accidents,
               SUM(severity) AS serious_accidents,
               ROUND(100.0 * SUM(severity) / COUNT(*), 2) AS serious_pct
        FROM accidents
        WHERE borough IS NOT NULL AND borough != ''
        GROUP BY borough
        ORDER BY serious_pct DESC
    """,
    "top_zip_codes": """
        SELECT zip_code,
               COUNT(*) AS total_accidents,
               SUM(severity) AS serious_accidents,
               ROUND(100.0 * SUM(severity) / COUNT(*), 2) AS serious_pct
        FROM accidents
        WHERE zip_code IS NOT NULL AND zip_code != 'UNKNOWN' AND zip_code != ''
        GROUP BY zip_code
        HAVING COUNT(*) >= 50
        ORDER BY serious_accidents DESC
        LIMIT 10
    """,
    "borough_hour": """
        SELECT borough,
               CAST(strftime('%H', crash_datetime) AS INTEGER) AS hour_of_day,
               COUNT(*) AS total_accidents,
               SUM(severity) AS serious_accidents
        FROM accidents
        WHERE borough IS NOT NULL AND borough != ''
        GROUP BY borough, hour_of_day
        ORDER BY serious_accidents DESC
        LIMIT 10
    """,
}


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SQLITE_PATH) as conn:
        for name, sql in QUERIES.items():
            df = pd.read_sql_query(sql, conn)
            out = METRICS_DIR / f"sql_{name}.csv"
            df.to_csv(out, index=False)
            print(f"[sql] wrote {out}")
            print(df.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
