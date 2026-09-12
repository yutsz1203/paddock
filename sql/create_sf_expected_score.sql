DROP TABLE IF EXISTS scratch.sf_expected_score;
-- Step 5b: the expected score for each class and distance key.
-- An 'all' row averages every race of that class, not only the fallback races.
CREATE TABLE scratch.sf_expected_score AS
SELECT pass, score_class, distance_key,
       AVG(score) AS expected_score, STDDEV_SAMP(score) AS sd_score, COUNT(*) AS n_races
FROM scratch.sf_race_score
WHERE pass = 1 AND distance_key <> 'all'
GROUP BY pass, score_class, distance_key
UNION ALL
SELECT pass, score_class, 'all',
       AVG(score), STDDEV_SAMP(score), COUNT(*)
FROM scratch.sf_race_score
WHERE pass = 1
  AND score_class IN (SELECT score_class FROM scratch.sf_race_score
                      WHERE pass = 1 AND distance_key = 'all')
GROUP BY pass, score_class;

ALTER TABLE scratch.sf_expected_score ADD PRIMARY KEY (pass, score_class, distance_key);