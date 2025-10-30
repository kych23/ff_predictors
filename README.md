# Fantasy Football Predictors

Machine learning project all about PPR fantasy football.

Stack

- Python for ETL/ML
- FastAPI for an HTTP API
- PostgreSQL for storage

Data source: nflverse via `nflreadpy`

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Start Postgres (Docker or local service), then initialize tables:

```bash
python -c "from src.db.init_db import init_db; init_db(); print('DB ready')"
```

## Load data (from nflreadpy directly)

```bash
python -m src.ml.load_data
```

## Train a model

```bash
python -m src.ml.train
```

## Run the API

```bash
uvicorn src.api.main:app --reload
```

## Attribution

- NFL data via nflverse (`nflreadpy`).
