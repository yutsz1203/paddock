--finding seconds per length 
-- it's actual fixed - one length is always 0.16s
WITH w AS (
    SELECT race_id, MIN(finish_time_s) AS t
    FROM runners
    WHERE finish_pos = 1
    GROUP BY race_id
)
SELECT ROUND((SUM(r.finish_time_s - w.t) / NULLIF(SUM(r.margin), 0))::numeric, 3)
       AS seconds_per_length
FROM runners r
JOIN w USING (race_id)
WHERE r.margin BETWEEN 1 AND 15;