# Deploying paddock

This is the runbook for the public demo. It puts the stack on an Oracle Cloud Always
Free ARM instance, behind Caddy, with the corpus restored from a dump.

Read the whole page before you start. Two steps are hard to undo: the corpus
transfer takes about half an hour, and Let's Encrypt rate limits repeated
certificate requests for the same name.

## What runs where

```
        internet
           │  443 (and 80, for the certificate challenge)
           ▼
      ┌─────────┐
      │  caddy  │   terminates TLS, appends X-Forwarded-For
      └────┬────┘
           │  compose network — no published port faces the internet
      ┌────┴─────────────┬──────────────┐
      ▼                  ▼              ▼
  /ask /coverage      everything      postgres
  /health → api         else → ui      (pgvector)
```

The API and the demo are separate processes, so the demo talks to the API over
`http://api:8000`. Postgres, the API and the demo publish only to `127.0.0.1`, so
Caddy is the only way in from outside the box.

## Before you start

You need five things:

1. The instance, reachable as `ssh paddock`.
2. A public DNS name pointing at it. Let's Encrypt will not issue a certificate for
   a bare IP, so a name is required, not optional.
3. Ports 80 and 443 open, in the Oracle security list **and** in `iptables` on the
   instance. Both. The security list alone is not enough.
4. Docker and the Compose plugin on the instance.
5. A local corpus that `paddock check integrity` reports as clean.

## Step 1 — Open the ports on the instance

Read the chain before you insert. On the Ubuntu Minimal image the default `INPUT`
chain ends with a `REJECT` rule, and a rule appended below it is dead:

```bash
ssh paddock 'sudo iptables -L INPUT --line-numbers'
```

Insert above the `REJECT` line. Use the position that command reports, which is 5 on
this image:

```bash
ssh paddock 'sudo iptables -I INPUT 5 -p tcp --dport 80 -j ACCEPT'
ssh paddock 'sudo iptables -I INPUT 5 -p tcp --dport 443 -j ACCEPT'
ssh paddock 'sudo netfilter-persistent save'
```

## Step 2 — Set the spend cap at the provider

Do this before the URL is public, not after.

`API_DAILY_LLM_CALL_CAP` is the application half of the cap. It counts calls in
memory, so it resets if the container restarts. The half that cannot be evaded is a
billing limit on the key itself. Set it in the provider's console:

- A budget with a hard cap on the API key the demo uses.
- An alert below the cap, so you hear about it before it stops answering.

If you skip this step, a crash loop with traffic on it can spend past the number in
the Compose file.

## Step 3 — Put the code and the secrets on the instance

```bash
ssh paddock 'git clone https://github.com/<you>/paddock.git ~/paddock'
```

Write `~/paddock/.env` on the instance. Never commit it. It needs:

```
PADDOCK_DOMAIN=<your public name>
ACME_EMAIL=<where Let's Encrypt sends expiry warnings>

LLM_PROVIDER=gemini
LLM_MODEL=<model>
GEMINI_API_KEY=<key>
```

Leave `DATABASE_URL` out. The Compose file sets it, and a `environment:` key beats
an `env_file` entry, so a `DATABASE_URL` here is ignored rather than obeyed — which
is worse than either, because the file then states something that is not true.

## Step 4 — Start the stack

```bash
ssh paddock 'cd ~/paddock && docker compose -f docker-compose.yml -f docker-compose.deploy.yml up -d --build'
```

The build takes about ten minutes on two cores, most of it installing torch.

Watch Caddy get its certificate:

```bash
ssh paddock 'cd ~/paddock && docker compose logs -f caddy'
```

If the certificate fails, read the error before retrying. Let's Encrypt rate limits
repeated failures for the same name, and a retry loop can lock you out for a week.
The two usual causes are a DNS name that does not resolve to this box yet, and port
80 blocked in one of the two places Step 1 covers.

## Step 5 — Restore the corpus

**The box does not use `data/seed/`.** That file is 20 meetings and no page archive,
which is the right size for a clean clone and the wrong dataset for the live
pipeline. The box takes a full dump of the local corpus, page archive and watermarks
included, because it runs the pipeline.

The dump is transferred out of band and never committed. On your machine:

```bash
docker compose exec -T postgres pg_dump -U paddock -d paddock -Fc > /tmp/paddock_full.dump
scp /tmp/paddock_full.dump paddock:/tmp/
```

It is about 200 MB compressed and the transfer takes a while.

Stop the two services that hold connections first. `DROP DATABASE` fails while any
session is open, and the API opens one at startup — so a restore attempted with the
stack running fails on its first command:

```bash
ssh paddock 'cd ~/paddock && docker compose stop api ui'
```

Then drop, create and restore:

```bash
ssh paddock 'cd ~/paddock && docker compose exec -T postgres psql -U paddock -d postgres -c "DROP DATABASE IF EXISTS paddock"'
ssh paddock 'cd ~/paddock && docker compose exec -T postgres psql -U paddock -d postgres -c "CREATE DATABASE paddock"'
ssh paddock 'cd ~/paddock && docker compose exec -T postgres pg_restore -U paddock -d paddock --no-owner < /tmp/paddock_full.dump'
```

The dump carries the schema, the `vector` extension and the HNSW index, so no
migration step is needed. Start the two services again:

```bash
ssh paddock 'cd ~/paddock && docker compose start api ui'
```

## Step 6 — Warm the model before anyone visits

bge-m3 is 2.2 GB and is fetched on the first question, not at startup. On two cores
that first question takes minutes, and the acceptance criterion is 20 seconds. Ask
one question yourself, from the box, so the visitor is never the one paying:

```bash
ssh paddock 'curl -sS -m 900 -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Did SETANTA have any trouble in running last time?\"}" | tail -3'
```

The model lands in the `models` volume and stays there. A container restart reloads
it from disk rather than from Hugging Face, so this step is needed once per
deployment, not once per restart.

## Step 7 — Check it from outside

Do this from a phone on mobile data, not from the machine you deployed from. A
laptop on the same network can succeed for reasons a visitor does not have.

1. Open `https://<your domain>`. The data-range banner must state the real range.
2. Ask three questions, one of them in Chinese.
3. Open one source card and confirm that it holds the comment the answer cites.
4. Ask a question the current scope refuses. It must say so, not fail.
5. Confirm the rate limiter triggers. Eleven requests in a minute from one address:

   ```bash
   for i in $(seq 1 11); do
     curl -s -o /dev/null -w '%{http_code}\n' -X POST https://<your domain>/ask \
       -H 'Content-Type: application/json' -d '{"question":"Did SETANTA have trouble?"}'
   done
   ```

   The first ten answer 200 and the eleventh answers 429 with a `Retry-After`
   header. If every one of them answers 200, `API_TRUSTED_PROXY_HOPS` does not match
   the number of proxies in the path and the limiter is metering nothing.

## Updating it later

```bash
ssh paddock 'cd ~/paddock && git pull && docker compose -f docker-compose.yml -f docker-compose.deploy.yml up -d --build'
```

The corpus lives in the `pgdata` volume and survives. So does the model cache, in
`models`, and the certificate, in `caddy_data`.

## If something breaks

| Symptom | Look here first |
|---|---|
| No certificate | `docker compose logs caddy`. DNS, then port 80 in both places. |
| Every question says the model is not configured | `LLM_PROVIDER` and its key in `.env` on the instance. The API logs it at startup. |
| Every question says the budget is spent | The daily cap. It resets at midnight UTC. |
| Answers are slow, then stop | `docker stats`. bge-m3 is resident and the box has 12 GB. |
| The limiter never triggers | `API_TRUSTED_PROXY_HOPS` against the number of proxies. |
| The demo loads but shows no data range | The corpus restore, Step 5. The API answers, the database is empty. |
