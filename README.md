# Fantasy Football Predictors

A comprehensive machine learning pipeline for predicting NFL player fantasy football performance and making lineup decisions.

---

End-to-end, free-data pipeline:

- Ingest nflverse datasets with `nflreadpy`
- Store as Parquet
- Build features with DuckDB SQL
- Train baseline models (notebook)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
