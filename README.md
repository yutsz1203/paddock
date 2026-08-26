# Paddock

A bilingual (English/Chinese) question-answering system over Hong Kong Jockey Club
racing data, combining vector retrieval with SQL over a structured corpus.

Race results, sectional times and stewards' reports are scraped from HKJC, stored in
Postgres (pgvector), and answered by a LangGraph agent that routes a question to
either structured queries or semantic search over race comments. No answer is
returned without a citation — if retrieval finds no evidence, the agent abstains
rather than guessing.

**Status:** work in progress. 

## Run the demo

One command from a clean clone. It makes no request to HKJC.

```bash
make demo              # restores data/seed/, then serves :8000 and :8501
```

`make demo` restores the committed dataset — 20 meetings, May to July 2026 — into a
database of its own. It never writes to your corpus database. Then it starts the API
and the Streamlit demo, and stops both on Ctrl-C.

You need Docker and an API key for one LLM provider. Copy `.env.example` to `.env`
and fill the key in. Without a key the demo still starts and still states its data
range, and every question reports the missing key.

Two downloads are not HKJC and are not avoidable: the Python dependencies, and
bge-m3 from Hugging Face (~2.2 GB) on the first question. Later questions are fast.

See [`data/seed/README.md`](data/seed/README.md) for what the dataset holds, what it
leaves out, and how to restore it by hand.

## Run it against your own corpus

The API and the demo are two processes. Start the API first.

```bash
make up                # Postgres
uv run paddock serve   # API  :8000
make ui                # Streamlit demo  :8501
```

The demo needs the `ui` extra. `make ui` installs nothing, so run
`uv sync --extra ui` once before the first start.

## What it answers

The demo states its data range and its scope on the first screen. It answers
questions about one horse at a time. It refuses a question about a jockey or about
a whole race, because the router reads keywords only. See ADR-004.
