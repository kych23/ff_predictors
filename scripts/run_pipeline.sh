#!/usr/bin/env bash
# Full pipeline from empty Supabase DB to draft-ready projections.
# Usage: bash scripts/run_pipeline.sh [DRAFT_YEAR]
# DRAFT_YEAR defaults to the current year.

set -euo pipefail

DRAFT_YEAR="${1:-$(date +%Y)}"
START_YEAR=2012

cd "$(dirname "$0")/.."

echo "=========================================="
echo "FantasyForecast pipeline — ${DRAFT_YEAR} season"
echo "Start: ${START_YEAR}  End: ${DRAFT_YEAR}"
echo "=========================================="

echo ""
echo "[1/5] Seeding DB (NFL data + ADP)..."
python scripts/seed_db.py --start "${START_YEAR}" --end "${DRAFT_YEAR}"

echo ""
echo "[2/5] Building season labels..."
python scripts/build_labels.py

echo ""
echo "[3/5] Building features (${START_YEAR}–${DRAFT_YEAR})..."
python scripts/build_features.py --start "${START_YEAR}" --end "${DRAFT_YEAR}"

echo ""
echo "[4/5] Training model + generating ${DRAFT_YEAR} projections..."
python scripts/train_projection.py

echo ""
echo "[5/5] Running benchmarks (Tier-1 + Tier-2 + Tier-3)..."
python scripts/run_benchmark.py --with-sim

echo ""
echo "=========================================="
echo "Pipeline complete. Start your draft with:"
echo "  python scripts/draft.py --season ${DRAFT_YEAR} --position <draft_position>"
echo "=========================================="
