-- Step 8: pass 2. Safe to run again: it first deletes any earlier pass-2 rows.
DELETE FROM scratch.sf_figure         WHERE pass = 2;
DELETE FROM scratch.sf_variant        WHERE pass = 2;
DELETE FROM scratch.sf_expected_score WHERE pass = 2;
DELETE FROM scratch.sf_race_score     WHERE pass = 2;
DELETE FROM scratch.sf_par            WHERE pass = 2;

-- 8a. New par from winning times adjusted to a neutral track:
--     adjusted_time = actual_time - variant_points * par / 1000
INSERT INTO scratch.sf_par (pass, racecourse, track, distance_m, par_s, points_per_second, n_races)
WITH winners AS (
    SELECT ra.id AS race_id, ra.race_no, ra.meeting_id, m.racecourse, ra.track, ra.distance_m,
           MIN(r.finish_time_s) AS win_time_s
    FROM runners r
    JOIN races ra   ON ra.id = r.race_id
    JOIN meetings m ON m.id = ra.meeting_id
    WHERE ra.status = 'finished' AND NOT r.scratched
      AND r.finish_pos = 1 AND r.finish_time_s IS NOT NULL
    GROUP BY ra.id, ra.race_no, ra.meeting_id, m.racecourse, ra.track, ra.distance_m
),
adjusted AS (
    SELECT w.racecourse, w.track, w.distance_m,
           w.win_time_s - v.variant_points * p.par_s / 1000 AS adj_time_s
    FROM winners w
    JOIN scratch.sf_par p
      ON p.pass = 1 AND p.racecourse = w.racecourse
     AND p.track = w.track AND p.distance_m = w.distance_m
    JOIN scratch.sf_variant v
      ON v.pass = 1 AND v.meeting_id = w.meeting_id AND v.track = w.track
     AND w.race_no BETWEEN v.first_race_no AND v.last_race_no
)
SELECT 2, racecourse, track, distance_m,
       AVG(adj_time_s), 1000 / AVG(adj_time_s), COUNT(*)
FROM adjusted
GROUP BY racecourse, track, distance_m;

-- 8b. Raw again, with the new par and the ACTUAL time (the variant is added in
INSERT INTO scratch.sf_figure
    (pass, runner_id, race_id, meeting_id, track, distance_m, race_class,
     finish_pos, raw, dirty, figure)
SELECT 2, f.runner_id, f.race_id, f.meeting_id, f.track, f.distance_m, f.race_class,
       f.finish_pos,
       100 + (p.par_s - r.finish_time_s) * p.points_per_second,
       f.dirty,
       NULL
FROM scratch.sf_figure f
JOIN runners r  ON r.id = f.runner_id
JOIN meetings m ON m.id = f.meeting_id
JOIN scratch.sf_par p
  ON p.pass = 2 AND p.racecourse = m.racecourse
 AND p.track = f.track AND p.distance_m = f.distance_m
WHERE f.pass = 1;

-- 8c. Race scores again. The cell of each race is the same as in pass 1.
INSERT INTO scratch.sf_race_score
    (pass, race_id, meeting_id, track, race_class, distance_m, score_class, band,
     score, n_clean, distance_key)
SELECT 2, rs.race_id, rs.meeting_id, rs.track, rs.race_class, rs.distance_m,
       rs.score_class, rs.band, s.score, s.n_clean, rs.distance_key
FROM scratch.sf_race_score rs
JOIN (
    SELECT race_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY raw) AS score,
           COUNT(*) AS n_clean
    FROM scratch.sf_figure
    WHERE pass = 2 AND NOT dirty
    GROUP BY race_id
) s ON s.race_id = rs.race_id
WHERE rs.pass = 1;

-- 8d. Expected scores again
INSERT INTO scratch.sf_expected_score
    (pass, score_class, distance_key, expected_score, sd_score, n_races)
SELECT pass, score_class, distance_key, AVG(score), STDDEV_SAMP(score), COUNT(*)
FROM scratch.sf_race_score
WHERE pass = 2 AND distance_key <> 'all'
GROUP BY pass, score_class, distance_key
UNION ALL
SELECT pass, score_class, 'all', AVG(score), STDDEV_SAMP(score), COUNT(*)
FROM scratch.sf_race_score
WHERE pass = 2
  AND score_class IN (SELECT score_class FROM scratch.sf_race_score
                      WHERE pass = 2 AND distance_key = 'all')
GROUP BY pass, score_class;

-- 8e. Variants again. The parts of each card are the same as in pass 1.
INSERT INTO scratch.sf_variant
    (pass, meeting_id, track, part, first_race_no, last_race_no,
     variant_points, sd_dev, n_races)
SELECT 2, v.meeting_id, v.track, v.part, v.first_race_no, v.last_race_no,
       AVG(e.expected_score - rs.score),
       STDDEV_SAMP(e.expected_score - rs.score),
       COUNT(*)
FROM scratch.sf_variant v
JOIN scratch.sf_race_score rs
  ON rs.pass = 2 AND rs.meeting_id = v.meeting_id AND rs.track = v.track
JOIN races ra ON ra.id = rs.race_id
 AND ra.race_no BETWEEN v.first_race_no AND v.last_race_no
JOIN scratch.sf_expected_score e
  ON e.pass = 2 AND e.score_class = rs.score_class AND e.distance_key = rs.distance_key
WHERE v.pass = 1
GROUP BY v.meeting_id, v.track, v.part, v.first_race_no, v.last_race_no;

-- 8f. Figures: raw + variant, as in step 7
UPDATE scratch.sf_figure f
SET figure = f.raw + v.variant_points
FROM races ra, scratch.sf_variant v
WHERE ra.id = f.race_id
  AND v.pass = f.pass
  AND v.meeting_id = f.meeting_id
  AND v.track = f.track
  AND ra.race_no BETWEEN v.first_race_no AND v.last_race_no
  AND f.pass = 2;