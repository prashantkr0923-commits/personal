-- ============================================================================
-- Week 1 - SQL tasks (UAV Propeller Performance)
-- ----------------------------------------------------------------------------
-- Tables (loaded by run_sql.py from the raw experiment CSVs):
--   experiment1  <- Experiment_vol1.csv
--   experiment2  <- Experiment_vol2.csv
--   experiment3  <- Experiment_vol3.csv
-- Column names are the original labels, exposed to SQL as safe aliases:
--   propeller_name, blade_name, propeller_brand, number_of_blades,
--   propeller_diameter, propeller_pitch, advance_ratio, rpm,
--   thrust_coeff, power_coeff, efficiency
-- Thrust coefficient is a fraction, so "more than 12%" means > 0.12.
-- ============================================================================


-- Q1. In experiment 1, total number of propellers whose thrust coefficient
--     output is more than 12% (i.e. > 0.12). A "propeller" is one distinct
--     Propeller's Name; we count the distinct propellers that reach CT > 0.12.
SELECT COUNT(DISTINCT propeller_name) AS propellers_ct_over_12pct
FROM   experiment1
WHERE  thrust_coeff > 0.12;

-- (companion) how many individual operating-point ROWS exceed 12% thrust:
SELECT COUNT(*) AS rows_ct_over_12pct
FROM   experiment1
WHERE  thrust_coeff > 0.12;


-- Q2. Reorder experiment 1 in descending order of efficiency output.
--     (Preview of the top of the reordered table.)
SELECT propeller_name, advance_ratio, rpm,
       thrust_coeff, power_coeff, efficiency
FROM   experiment1
ORDER BY efficiency DESC;


-- Q3. The 100 least-performing propellers based on power coefficient output
--     in experiment 1 (the 100 rows with the smallest power coefficient).
SELECT propeller_name, advance_ratio, rpm, power_coeff, efficiency
FROM   experiment1
ORDER BY power_coeff ASC
LIMIT 100;


-- Q4. Merge experiment 1, 2 and 3, then count the propellers whose efficiency
--     output is negative OR zero. UNION ALL stacks the three databases; we
--     count the distinct propellers that ever record efficiency <= 0.
WITH merged AS (
    SELECT * FROM experiment1
    UNION ALL
    SELECT * FROM experiment2
    UNION ALL
    SELECT * FROM experiment3
)
SELECT COUNT(DISTINCT propeller_name) AS propellers_eff_le_0
FROM   merged
WHERE  efficiency <= 0;

-- (companion) total ROWS across the merged data with efficiency <= 0:
WITH merged AS (
    SELECT * FROM experiment1
    UNION ALL
    SELECT * FROM experiment2
    UNION ALL
    SELECT * FROM experiment3
)
SELECT COUNT(*) AS rows_eff_le_0
FROM   merged
WHERE  efficiency <= 0;
