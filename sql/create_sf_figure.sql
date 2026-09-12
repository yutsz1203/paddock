DROP TABLE IF EXISTS scratch.sf_figure;

CREATE TABLE scratch.sf_figure AS
WITH timed AS (
    SELECT r.id AS runner_id, r.race_id, ra.meeting_id, m.racecourse, ra.track,
           ra.distance_m, ra.race_class, r.finish_pos, r.margin, r.finish_time_s,
           MIN(r.finish_time_s) OVER (PARTITION BY r.race_id) AS fastest_s
    FROM runners r
    JOIN races ra   ON ra.id = r.race_id
    JOIN meetings m ON m.id = ra.meeting_id
    WHERE ra.status = 'finished'
      AND NOT r.scratched
      AND r.finish_time_s IS NOT NULL
)
SELECT 1 AS pass,
       t.runner_id, t.race_id, t.meeting_id, t.track, t.distance_m, t.race_class,
       t.finish_pos,
       100 + (p.par_s - t.finish_time_s) * p.points_per_second AS raw,
       -- Official margin first. If it is missing, use time behind the fastest
       -- runner at 0.16 s per length, the rate measured on this data.
       COALESCE(t.margin, (t.finish_time_s - t.fastest_s) / 0.16) > 15 AS dirty,
       NULL::double precision AS figure
FROM timed t
JOIN scratch.sf_par p
  ON p.pass = 1
 AND p.racecourse = t.racecourse
 AND p.track = t.track
 AND p.distance_m = t.distance_m;

ALTER TABLE scratch.sf_figure ADD PRIMARY KEY (pass, runner_id);