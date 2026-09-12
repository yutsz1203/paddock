DROP TABLE IF EXISTS scratch.sf_variant;

-- Step 6: the daily variant. Positive = the track was slow, so figures go up.
-- One variant per meeting and surface. If the going changed during the meeting,
-- the card splits at the first change: 'early' and 'late', each with 3+ races.
CREATE TABLE scratch.sf_variant AS
WITH dev AS (
    SELECT rs.pass, rs.meeting_id, rs.track, ra.race_no, ra.going,
           e.expected_score - rs.score AS dev
    FROM scratch.sf_race_score rs
    JOIN scratch.sf_expected_score e
      ON e.pass = rs.pass
     AND e.score_class = rs.score_class
     AND e.distance_key = rs.distance_key
    JOIN races ra ON ra.id = rs.race_id
    WHERE rs.pass = 1
),
changed AS (
    SELECT dev.*,
           -- True from the first race whose going differs from the first race's going
           bool_or(going <> first_going) OVER (
               PARTITION BY meeting_id, track ORDER BY race_no
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS after_change
    FROM (
        SELECT dev.*,
               first_value(going) OVER (PARTITION BY meeting_id, track ORDER BY race_no) AS first_going
        FROM dev
    ) dev
),
parts AS (
    SELECT changed.*,
           CASE WHEN COUNT(*) FILTER (WHERE NOT after_change) OVER w >= 3
                 AND COUNT(*) FILTER (WHERE after_change)     OVER w >= 3
                THEN CASE WHEN after_change THEN 'late' ELSE 'early' END
                ELSE 'all' END AS part
    FROM changed
    WINDOW w AS (PARTITION BY meeting_id, track)
)
SELECT pass, meeting_id, track, part,
       MIN(race_no)     AS first_race_no,
       MAX(race_no)     AS last_race_no,
       AVG(dev)         AS variant_points,
       STDDEV_SAMP(dev) AS sd_dev,
       COUNT(*)         AS n_races
FROM parts
GROUP BY pass, meeting_id, track, part;

ALTER TABLE scratch.sf_variant ADD PRIMARY KEY (pass, meeting_id, track, part);