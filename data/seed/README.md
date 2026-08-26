# The demo dataset

`paddock_demo.dump` is a compressed `pg_dump` of a slice of the corpus. It lets you
clone this repository and ask a question without a single request to HKJC.

## What it holds

Twenty meetings, 6 May 2026 to 15 July 2026, at both racecourses.

| Table | Rows |
|---|---|
| meetings | 20 |
| races | 202 |
| runners | 2,564 |
| incident_comments | 2,214 |
| chunks (embedded) | 4,625 |
| horses | 1,064 |
| jockeys | 27 |
| trainers | 24 |

The file is 22 MB. It carries the schema, the data, the `vector` extension and the
HNSW index, so a restore needs no migration step.

## Restore it

Run one command from a clean clone:

```bash
make demo
```

`make demo` restores this file into a database named after the configured one with
`_demo` appended. It never writes to the corpus database. Then it starts the API and
the Streamlit demo, and stops both on Ctrl-C.

To restore it by hand instead:

```bash
make up
docker compose exec -T postgres psql -U paddock -d postgres -c 'CREATE DATABASE paddock_demo'
docker compose exec -T postgres pg_restore -U paddock -d paddock_demo --no-owner < data/seed/paddock_demo.dump
DATABASE_URL=postgresql+psycopg://paddock:paddock@localhost:55432/paddock_demo uv run paddock demo report
```

## What it does not hold, and why

**The page archive.** `fetched_pages` is 117 MB of HKJC's own HTML. The spec forbids
publishing bulk HKJC-derived data beyond a small demo slice. One command depends on
it: `paddock check integrity` re-derives each meeting's date from the archived page,
so that command cannot run against the demo database.

**Meetings before 6 May 2026.** The corpus holds 176. All 36,417 vectors are about
200 MB compressed, which GitHub refuses. Twenty meetings are 22 MB.

**Horses, jockeys and trainers with no run in the window.** A horse row with no
starts answers "no runs found", which is a claim about form the slice cannot
support. An absent horse answers "I do not know that horse", which is true.

**Ingest bookkeeping.** `ingest_runs` and `watermarks` are empty. A restored snapshot
never ran an ingest. A watermark left behind tells a later `ingest since` that dates
the demo does not hold were already done.

## Rebuild it

You only need this after a backfill widens what should ship.

```bash
make seed              # 20 meetings, the default
MEETINGS=30 make seed  # more, if the size ceiling allows
```

`make seed` copies the corpus into a throwaway database, cuts the slice there, dumps
it, and drops the copy. The corpus is never written to. The script refuses to write a
file above 50 MB, because GitHub warns at 50 MB and refuses at 100 MB.

Each rebuild adds a new 22 MB object to the git history. Rebuild it when the data
must change, not to tidy it.

## The Oracle deployment is different

T24 restores the **whole** corpus onto the instance, page archive and watermarks
included, because that box runs the live pipeline from 6 September. That dump is not
committed here, and building it needs no code:

```bash
docker compose exec -T postgres pg_dump -U paddock -Fc -Z9 paddock > paddock_full.dump
```

Copy that file to the instance out of band. Do not commit it. It is bulk
HKJC-derived data.
