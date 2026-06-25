-- ============================================================
-- AttritionIQ SQL Analytics Queries
-- Load HR CSV into SQLite, then run these queries
-- ============================================================

-- 1. Overall Attrition Rate
SELECT
    Attrition,
    COUNT(*)                              AS employee_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM hr_data
GROUP BY Attrition;


-- 2. Attrition by Department
SELECT
    Department,
    COUNT(*)                                                      AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)           AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
          * 100.0 / COUNT(*), 2)                                  AS attrition_rate_pct
FROM hr_data
GROUP BY Department
ORDER BY attrition_rate_pct DESC;


-- 3. Attrition by Job Role
SELECT
    JobRole,
    COUNT(*)                                                      AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)           AS left_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                                    AS attrition_pct,
    ROUND(AVG(MonthlyIncome), 0)                                  AS avg_monthly_income,
    ROUND(AVG(JobSatisfaction), 2)                                AS avg_job_satisfaction
FROM hr_data
GROUP BY JobRole
ORDER BY attrition_pct DESC;


-- 4. Attrition by Overtime
SELECT
    OverTime,
    COUNT(*)                                                      AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)           AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                                    AS attrition_pct
FROM hr_data
GROUP BY OverTime;


-- 5. Salary Band Analysis
SELECT
    CASE
        WHEN MonthlyIncome < 3000  THEN '< $3K'
        WHEN MonthlyIncome < 6000  THEN '$3K–$6K'
        WHEN MonthlyIncome < 10000 THEN '$6K–$10K'
        ELSE '> $10K'
    END                                                           AS salary_band,
    COUNT(*)                                                      AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)           AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                                    AS attrition_pct,
    ROUND(AVG(JobSatisfaction), 2)                                AS avg_satisfaction
FROM hr_data
GROUP BY salary_band
ORDER BY attrition_pct DESC;


-- 6. Tenure vs Attrition (Years at Company)
SELECT
    CASE
        WHEN YearsAtCompany <= 1  THEN '0–1 Years'
        WHEN YearsAtCompany <= 3  THEN '2–3 Years'
        WHEN YearsAtCompany <= 5  THEN '4–5 Years'
        WHEN YearsAtCompany <= 10 THEN '6–10 Years'
        ELSE '10+ Years'
    END                                                           AS tenure_band,
    COUNT(*)                                                      AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)           AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                                    AS attrition_pct
FROM hr_data
GROUP BY tenure_band
ORDER BY attrition_pct DESC;


-- 7. Work-Life Balance vs Attrition
SELECT
    WorkLifeBalance,
    CASE WorkLifeBalance
        WHEN 1 THEN 'Bad'
        WHEN 2 THEN 'Good'
        WHEN 3 THEN 'Better'
        WHEN 4 THEN 'Best'
    END                                                           AS wlb_label,
    COUNT(*)                                                      AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)           AS attrition_count,
    ROUND(SUM(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                                    AS attrition_pct
FROM hr_data
GROUP BY WorkLifeBalance
ORDER BY WorkLifeBalance;


-- 8. High-Risk Employee Profile
-- Employees most likely to leave based on known risk factors
SELECT
    EmployeeNumber,
    Department,
    JobRole,
    MonthlyIncome,
    OverTime,
    YearsAtCompany,
    JobSatisfaction,
    WorkLifeBalance,
    Attrition
FROM hr_data
WHERE OverTime = 'Yes'
  AND JobSatisfaction <= 2
  AND WorkLifeBalance <= 2
  AND MonthlyIncome < 5000
ORDER BY MonthlyIncome ASC
LIMIT 20;
