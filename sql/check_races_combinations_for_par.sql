-- Query 2: races per racecourse x track x distance x rail.
-- Counts only races that can build a par: finished, with a timed winner.
WITH par_races AS (
    SELECT ra.id AS race_id, m.racecourse, ra.track, ra.distance_m,
           COALESCE(ra.course, '-') AS rail          -- all-weather races have no rail
    FROM races ra
    JOIN meetings m ON m.id = ra.meeting_id
    WHERE ra.status = 'finished'
      AND EXISTS (SELECT 1 FROM runners r
                  WHERE r.race_id = ra.id AND r.finish_pos = 1
                    AND NOT r.scratched AND r.finish_time_s IS NOT NULL)
)
SELECT racecourse, track, distance_m, rail,
       COUNT(*)       AS n_races,
       COUNT(*) >= 30 AS has_30
FROM par_races
GROUP BY racecourse, track, distance_m, rail
ORDER BY n_races DESC;