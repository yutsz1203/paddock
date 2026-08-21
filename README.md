# Paddock

A bilingual (English/Chinese) question-answering system over Hong Kong Jockey Club
racing data, combining vector retrieval with SQL over a structured corpus.

Race results, sectional times and stewards' reports are scraped from HKJC, stored in
Postgres (pgvector), and answered by a LangGraph agent that routes a question to
either structured queries or semantic search over race comments. No answer is
returned without a citation — if retrieval finds no evidence, the agent abstains
rather than guessing.

**Status:** work in progress. 
