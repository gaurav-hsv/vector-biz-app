# Incentives Backend — Step 1 (Bootstrap)

This is a minimal, production-quality Python backend skeleton for the vector-based incentives lookup.

## What’s included
- FastAPI app with health endpoints
- Typed settings from environment (.env supported)
- PostgreSQL connection pool using psycopg (pgvector-ready)
- Structured logging baseline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env to set PG_DSN and OPENAI_API_KEY

uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /healthz` — app liveness
- `GET /readyz` — checks DB connectivity (`SELECT 1`)
- `GET /version` — returns app build info

If `/readyz` reports DB down, verify your `PG_DSN` in `.env` and that PostgreSQL is running.

## Next Steps (Step 2)
- Add migrations & tables for `incentive_documents` and `incentive_chunks` (pgvector).
- Seed initial schema & run a sample query.
