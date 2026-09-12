"""Rebuild the speed-figure tables in the scratch schema, in order.

Run it after an ingest. Every step drops and rebuilds what it owns, so the script is
safe to run again and needs no clean-up between runs.

    uv run python scripts/build_speed_figures.py

The order is fixed, because each step reads the table that the step before it writes:

    sf_par          par time for each racecourse, track and distance
    sf_figure       raw for every timed runner, and the dirty flag
    sf_race_score   median raw of the clean finishers, and the cell of each race
    sf_expected_score  average race score for each score class and distance key
    sf_variant      expected score minus race score, per meeting, surface and part
    update          figure = raw + variant
    pass 2          the same chain again, from times adjusted to a neutral track

Everything runs in one transaction. If a step fails, the whole rebuild rolls back and
the tables from the last good run are still there.

The report at the end prints the run-to-run change: the mean absolute change in figure
between two consecutive clean runs of one horse. It is the measure that compares two
methods, so watch it when the corpus grows. Lower is better.
"""

from __future__ import annotations

import argparse
import pathlib
import time

from sqlalchemy import text
from sqlalchemy.orm import Session

from paddock.db.session import session_scope

STEPS = [
    ("par", "create_sf_par.sql"),
    ("raw figures", "create_sf_figure.sql"),
    ("race scores", "create_race_score.sql"),
    ("expected scores", "create_sf_expected_score.sql"),
    ("variants", "create_sf_variant.sql"),
    ("figures", "update_sf_figure.sql"),
    ("pass 2", "sf_pass_2.sql"),
]

COUNTS = """
SELECT 'sf_par' AS table_name, pass, COUNT(*) FROM scratch.sf_par GROUP BY pass
UNION ALL
SELECT 'sf_figure', pass, COUNT(*) FROM scratch.sf_figure GROUP BY pass
UNION ALL
SELECT 'sf_race_score', pass, COUNT(*) FROM scratch.sf_race_score GROUP BY pass
UNION ALL
SELECT 'sf_expected_score', pass, COUNT(*) FROM scratch.sf_expected_score GROUP BY pass
UNION ALL
SELECT 'sf_variant', pass, COUNT(*) FROM scratch.sf_variant GROUP BY pass
ORDER BY 1, 2;
"""

# A finished race with a timed winner that has no figure. The usual cause is a par cell
# that the race does not join to, so it must be empty after a good rebuild.
UNRATED = """
SELECT m.race_date, ra.race_no, m.racecourse, ra.track, ra.distance_m
FROM races ra
JOIN meetings m ON m.id = ra.meeting_id
WHERE ra.status = 'finished'
  AND EXISTS (SELECT 1 FROM runners r
              WHERE r.race_id = ra.id AND NOT r.scratched AND r.finish_time_s IS NOT NULL)
  AND NOT EXISTS (SELECT 1 FROM scratch.sf_figure f WHERE f.race_id = ra.id AND f.pass = 2)
ORDER BY m.race_date, ra.race_no;
"""

NO_FIGURE = """
SELECT COUNT(*) FROM scratch.sf_figure WHERE pass = 2 AND figure IS NULL;
"""

RUN_TO_RUN = """
WITH runs AS (
    SELECT r.horse_id, m.race_date, ra.race_no, f.figure
    FROM scratch.sf_figure f
    JOIN runners r  ON r.id = f.runner_id
    JOIN races ra   ON ra.id = f.race_id
    JOIN meetings m ON m.id = ra.meeting_id
    WHERE f.pass = :pass AND NOT f.dirty AND f.figure IS NOT NULL
),
pairs AS (
    SELECT figure - lag(figure) OVER (PARTITION BY horse_id ORDER BY race_date, race_no) AS d
    FROM runs
)
SELECT AVG(ABS(d)), COUNT(*) FROM pairs WHERE d IS NOT NULL;
"""

COVERAGE = """
SELECT MIN(m.race_date), MAX(m.race_date), COUNT(DISTINCT f.race_id)
FROM scratch.sf_figure f
JOIN races ra   ON ra.id = f.race_id
JOIN meetings m ON m.id = ra.meeting_id
WHERE f.pass = 2;
"""


def rebuild(sql_dir: pathlib.Path) -> None:
    with session_scope() as session:
        for name, filename in STEPS:
            path = sql_dir / filename
            started = time.monotonic()
            session.execute(text("SELECT 1"))  # keep the transaction open on a fresh session
            session.connection().exec_driver_sql(path.read_text(encoding="utf-8"))
            print(f"  {name:16} {filename:28} {time.monotonic() - started:5.1f}s", flush=True)

        report(session)


def report(session: Session) -> None:
    print("\nrows")
    for table_name, pass_no, rows in session.execute(text(COUNTS)):
        print(f"  {table_name:20} pass {pass_no}  {rows:7d}")

    first, last, races = session.execute(text(COVERAGE)).one()
    print(f"\nrated races: {races} from {first} to {last}")

    unrated = session.execute(text(UNRATED)).all()
    print(f"finished races with no figure: {len(unrated)}")
    for row in unrated[:10]:
        print("  ", tuple(row))

    missing = session.execute(text(NO_FIGURE)).scalar_one()
    if missing:
        print(f"WARNING: {missing} pass-2 rows have a raw but no figure (no variant joined)")

    print("\nrun-to-run change (lower is better)")
    for pass_no in (1, 2):
        change, pairs = session.execute(text(RUN_TO_RUN), {"pass": pass_no}).one()
        print(f"  pass {pass_no}  {change:.3f} points over {pairs} pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sql-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "sql",
        help="directory that holds the build files (default: the sql directory of the repo)",
    )
    args = parser.parse_args()

    print(f"rebuilding from {args.sql_dir}")
    started = time.monotonic()
    rebuild(args.sql_dir)
    print(f"\ndone in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
