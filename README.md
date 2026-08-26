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

The API and the demo are two processes. Start the API first.

```bash
make up                # Postgres
uv run paddock serve   # API  :8000
make ui                # Streamlit demo  :8501
```

The demo needs the `ui` extra. `make ui` installs nothing, so run
`uv sync --extra ui` once before the first start.

The demo states its data range and its scope on the first screen. It answers
questions about one horse at a time. It refuses a question about a jockey or about
a whole race, because the router reads keywords only. See ADR-004.
