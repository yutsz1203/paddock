-- Races per class, distance, and race course.
-- Counts only races that can build a figure: finished, with a timed winner.
-- SELECT ra.race_class,
--        ra.distance_m,
--        m.racecourse,
--        COUNT(*)                                          AS n_races,
--        SUM(COUNT(*)) OVER (PARTITION BY ra.race_class)   AS class_total
-- FROM races ra
-- JOIN meetings m 
-- ON ra.meeting_id = m.id
-- WHERE ra.status = 'finished'
--   AND EXISTS (SELECT 1 FROM runners r
--               WHERE r.race_id = ra.id AND r.finish_pos = 1
--                 AND NOT r.scratched AND r.finish_time_s IS NOT NULL)
-- GROUP BY ra.race_class, ra.distance_m, m.racecourse
-- ORDER BY n_races DESC;

-- Races per class, diatance
SELECT ra.race_class,
       ra.distance_m,
       COUNT(*)                                          AS n_races,
       SUM(COUNT(*)) OVER (PARTITION BY ra.race_class)   AS class_total
FROM races ra
WHERE ra.status = 'finished'
  AND EXISTS (SELECT 1 FROM runners r
              WHERE r.race_id = ra.id AND r.finish_pos = 1
                AND NOT r.scratched AND r.finish_time_s IS NOT NULL)
GROUP BY ra.race_class, ra.distance_m
ORDER BY n_races DESC;