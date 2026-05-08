-- 1. Borough x hour ranking of total and serious crashes.
SELECT
    borough,
    CAST(strftime('%H', crash_datetime) AS INTEGER) AS hour_of_day,
    COUNT(*) AS total_accidents,
    SUM(severity)  AS serious_accidents
FROM accidents
WHERE crash_date BETWEEN '2021-01-01' AND '2022-12-31'
GROUP BY borough, hour_of_day
ORDER BY serious_accidents DESC, total_accidents DESC
LIMIT 25;

-- 2. Severity distribution by borough.
SELECT
    borough,
    COUNT(*) AS total_accidents,
    SUM(severity) AS serious_accidents,
    ROUND(100.0 * SUM(severity) / COUNT(*), 2) AS serious_pct
FROM accidents
WHERE borough IS NOT NULL
GROUP BY borough
HAVING COUNT(*) >= 100
ORDER BY serious_pct DESC;

-- 3. Day-of-week pattern.
SELECT
    CAST(strftime('%w', crash_datetime) AS INTEGER) AS day_of_week,
    COUNT(*) AS total_accidents,
    SUM(severity) AS serious_accidents,
    ROUND(100.0 * SUM(severity) / COUNT(*), 2) AS serious_pct
FROM accidents
GROUP BY day_of_week
ORDER BY day_of_week;

-- 4. Top zip codes by serious crashes.
SELECT
    zip_code,
    COUNT(*) AS total_accidents,
    SUM(severity) AS serious_accidents,
    ROUND(100.0 * SUM(severity) / COUNT(*), 2) AS serious_pct
FROM accidents
WHERE zip_code IS NOT NULL AND zip_code != 'UNKNOWN'
GROUP BY zip_code
HAVING COUNT(*) >= 50
ORDER BY serious_accidents DESC
LIMIT 20;

-- 5. Monthly trend.
SELECT
    strftime('%Y-%m', crash_date) AS year_month,
    COUNT(*) AS total_accidents,
    SUM(severity) AS serious_accidents,
    ROUND(100.0 * SUM(severity) / COUNT(*), 2) AS serious_pct
FROM accidents
GROUP BY year_month
ORDER BY year_month;
