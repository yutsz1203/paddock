-- Every par time: the official par per racecourse, track and distance (pass 2),
-- then one row per class. Class rows use the same adjusted times as the official
-- par, so the class rows of a cell average (weighted by races) to its 'All classes' row.
WITH winners AS (
    SELECT ra.id AS race_id, ra.race_no, ra.meeting_id, m.racecourse, ra.track,
           ra.distance_m, ra.race_class,
           MIN(r.finish_time_s) AS win_time_s
    FROM runners r
    JOIN races ra   ON ra.id = r.race_id
    JOIN meetings m ON m.id = ra.meeting_id
    WHERE ra.status = 'finished' AND NOT r.scratched
      AND r.finish_pos = 1 AND r.finish_time_s IS NOT NULL
    GROUP BY ra.id, ra.race_no, ra.meeting_id, m.racecourse, ra.track, ra.distance_m, ra.race_class
),
adjusted AS (
    -- The same adjustment as step 8a: pass-1 par and pass-1 variant
    SELECT w.racecourse, w.track, w.distance_m, w.race_class,
           w.win_time_s - v.variant_points * p.par_s / 1000 AS adj_time_s
    FROM winners w
    JOIN scratch.sf_par p
      ON p.pass = 1 AND p.racecourse = w.racecourse
     AND p.track = w.track AND p.distance_m = w.distance_m
    JOIN scratch.sf_variant v
      ON v.pass = 1 AND v.meeting_id = w.meeting_id AND v.track = w.track
     AND w.race_no BETWEEN v.first_race_no AND v.last_race_no
),
rows_ AS (
    SELECT racecourse, track, distance_m, 'All classes' AS race_class, par_s, n_races
    FROM scratch.sf_par
    WHERE pass = 2
    UNION ALL
    SELECT racecourse, track, distance_m, race_class, AVG(adj_time_s), COUNT(*)
    FROM adjusted
    GROUP BY racecourse, track, distance_m, race_class
)
SELECT x.racecourse, x.track, x.distance_m, x.race_class,
       floor(x.par_s / 60)::int || ':' || to_char(x.par_s - 60 * floor(x.par_s /60), 'FM00.00')
                                                            AS par_time,
       ROUND(x.par_s::numeric, 2)                           AS par_s,
       -- Positive = this class runs faster than the official par
       ROUND(((a.par_s - x.par_s) * 1000 / a.par_s)::numeric, 1) AS pts_vs_par,
       x.n_races
FROM rows_ x
JOIN scratch.sf_par a
  ON a.pass = 2 AND a.racecourse = x.racecourse
 AND a.track = x.track AND a.distance_m = x.distance_m
ORDER BY x.racecourse, x.track, x.distance_m,
         x.race_class <> 'All classes', x.par_s;