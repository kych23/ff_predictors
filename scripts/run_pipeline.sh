#!/usr/bin/env bash
# Full pipeline from empty Supabase DB to draft-ready projections.
# Usage: bash scripts/run_pipeline.sh [DRAFT_YEAR]
# DRAFT_YEAR defaults to the current year.

set -euo pipefail

DRAFT_YEAR="${1:-$(date +%Y)}"
START_YEAR=2012
WEEKLY_START=2017  # weekly grain needs rolling in-season stats; older seasons add noise

cd "$(dirname "$0")/.."

echo "=========================================="
echo "FantasyForecast pipeline — ${DRAFT_YEAR} season"
echo "Start: ${START_YEAR}  End: ${DRAFT_YEAR}"
echo "=========================================="

echo ""
echo "[1/8] Seeding DB (NFL data + ADP)..."
python scripts/seed_db.py --start "${START_YEAR}" --end "${DRAFT_YEAR}"

echo ""
echo "[2/8] Building season labels..."
python scripts/build_labels.py

echo ""
echo "[3/8] Building features (${START_YEAR}–${DRAFT_YEAR})..."
python scripts/build_features.py --start "${START_YEAR}" --end "${DRAFT_YEAR}"

echo ""
echo "[4/8] Training model + generating ${DRAFT_YEAR} projections..."
python scripts/train_projection.py

echo ""
echo "[5/8] Running benchmarks (Tier-1 + Tier-2 + Tier-3)..."
python scripts/run_benchmark.py --with-sim

echo ""
echo "[6/8] Building weekly labels + features (${WEEKLY_START}–${DRAFT_YEAR})..."
python scripts/build_weekly_data.py --start "${WEEKLY_START}" --end "${DRAFT_YEAR}"

echo ""
echo "[7/8] Training weekly model + OOF projections..."
python scripts/train_weekly_projection.py

echo ""
echo "[8/8] Running weekly lineup benchmark..."
python scripts/run_weekly_benchmark.py --start "${WEEKLY_START}"

echo ""
echo "=========================================="
echo "Pipeline complete."
echo "  Draft:     python scripts/draft.py --season ${DRAFT_YEAR} --position <draft_position>"
echo "  Start/sit: python scripts/project_week.py --season ${DRAFT_YEAR} --week <wk>"
echo "             python scripts/start.py --season ${DRAFT_YEAR} --week <wk> --roster <roster.yaml>"
echo "=========================================="
