-- Top 10 fastest speed figures
WITH top_10 AS (
    SELECT * FROM scratch.sf_figure sf
    JOIN meetings m ON sf.meeting_id=m.id
    WHERE pass=2 AND race_class NOT IN ('Group One', 'Group Two') AND race_date >= '2026-01-01'
    ORDER BY figure DESC
    LIMIT 10
)
SELECT h.name_zh, t.figure, m.race_date, m.source_url, m.racecourse, t.track, t.distance_m, t.race_class  
FROM top_10 t
JOIN meetings m 
ON t.meeting_id = m.id
JOIN runners r 
ON t.runner_id = r.id
JOIN horses h
ON r.horse_id = h.horse_id
ORDER BY figure DESC