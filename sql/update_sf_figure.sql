UPDATE scratch.sf_figure f
SET figure = f.raw + v.variant_points
FROM races ra, scratch.sf_variant v
WHERE ra.id = f.race_id
  AND v.pass = f.pass
  AND v.meeting_id = f.meeting_id
  AND v.track = f.track
  AND ra.race_no BETWEEN v.first_race_no AND v.last_race_no
  AND f.pass = 1;