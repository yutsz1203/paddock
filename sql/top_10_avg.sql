SELECT h.name_zh, AVG(sf.figure)
FROM scratch.sf_figure sf
JOIN runners r
ON sf.runner_id = r.id
JOIN horses h
ON r.horse_id = h.horse_id
GROUP BY h.name_zh
HAVING COUNT(*) >= 30
ORDER BY AVG(sf.figure) DESC
LIMIT 100