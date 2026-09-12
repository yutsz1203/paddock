CREATE SCHEMA IF NOT EXISTS scratch;
DROP TABLE IF EXISTS scratch.sf_par;

CREATE TABLE scratch.sf_par AS
WITH winners AS (
    SELECT ra.id AS race_id, m.racecourse, ra.track, ra.distance_m,
           MIN(r.finish_time_s) AS win_time_s    -- one row per race, dead heat or not
    FROM runners r
    JOIN races ra   ON ra.id = r.race_id
    JOIN meetings m ON m.id = ra.meeting_id
    WHERE ra.status = 'finished' AND NOT r.scratched
      AND r.finish_pos = 1 AND r.finish_time_s IS NOT NULL
    GROUP BY ra.id, m.racecourse, ra.track, ra.distance_m
)
SELECT 1                    AS pass,
       racecourse, track, distance_m,
       AVG(win_time_s)      AS par_s,
       1000 / AVG(win_time_s) AS points_per_second,
       COUNT(*)             AS n_races
FROM winners
GROUP BY racecourse, track, distance_m;

ALTER TABLE scratch.sf_par ADD PRIMARY KEY (pass, racecourse, track, distance_m);