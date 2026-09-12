DROP TABLE IF EXISTS scratch.sf_race_score;

-- Step 5a: one score per race = median raw of its clean finishers,
-- plus the cell of the expected-score table that the race belongs to
CREATE TABLE scratch.sf_race_score AS
WITH per_race AS (
    SELECT race_id, meeting_id, track, race_class, distance_m,
           CASE WHEN race_class IN ('Class 1', 'Group One', 'Group Two',
                                    'Group Three', '4YO')
                AND COUNT(*) OVER (PARTITION BY race_class) < 30 THEN 'Open'
                WHEN race_class = 'Griffin'                      THEN 'Class 5'
                ELSE race_class END                              AS score_class,
           CASE WHEN distance_m <= 1200 THEN '1000-1200'
                WHEN distance_m <= 1650 THEN '1400-1650'
                ELSE '1800+' END                           AS band,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY raw) AS score,
           COUNT(*)                                        AS n_clean
    FROM scratch.sf_figure
    WHERE pass = 1 AND NOT dirty
    GROUP BY race_id, meeting_id, track, race_class, distance_m
),
keyed AS (
    SELECT per_race.*,
           -- Exact distance when the class ran 100+ races at it, else the band
           CASE WHEN COUNT(*) OVER (PARTITION BY score_class, distance_m) >= 100
                THEN distance_m::text
                ELSE band END AS cell_key
    FROM per_race
)
SELECT 1 AS pass,
       race_id, meeting_id, track, race_class, distance_m, score_class, band,
       score, n_clean,
       -- A cell with fewer than 30 races falls back to its class over all distances
       CASE WHEN COUNT(*) OVER (PARTITION BY score_class, cell_key) >= 30
            THEN cell_key
            ELSE 'all' END AS distance_key
FROM keyed;

ALTER TABLE scratch.sf_race_score ADD PRIMARY KEY (pass, race_id);