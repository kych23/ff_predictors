#!/usr/bin/env bash
# In-season weekly update: refresh stats, rebuild the weekly grain, project the
# upcoming week. Run after the previous week's games finish (e.g. Tuesday).
#
# Usage: bash scripts/update_week.sh SEASON WEEK
#   WEEK = the upcoming week to project, e.g. after week 4's games:
#          bash scripts/update_week.sh 2026 5
#
# Then set your lineup:
#   python scripts/start.py --season SEASON --week WEEK --roster <roster.yaml>
#
# Takes a while (full re-seed + weekly feature rebuild): downstream loaders
# filter by snapshot_id, so a partial-season seed would orphan the training
# features under the old snapshot. One coherent snapshot per update.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: bash scripts/update_week.sh SEASON WEEK" >&2
    exit 1
fi

SEASON="$1"
WEEK="$2"
START_YEAR=2012
WEEKLY_START=2017

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Weekly update — ${SEASON} week ${WEEK}"
echo "=========================================="

echo ""
echo "[1/3] Re-seeding DB with latest stats (fresh snapshot, ADP skipped)..."
python scripts/seed_db.py --start "${START_YEAR}" --end "${SEASON}" --no-adp \
    --notes "in-season update: ${SEASON} week ${WEEK}"

echo ""
echo "[2/3] Rebuilding weekly labels + features (${WEEKLY_START}–${SEASON})..."
python scripts/build_weekly_data.py --start "${WEEKLY_START}" --end "${SEASON}"

echo ""
echo "[3/3] Projecting ${SEASON} week ${WEEK}..."
python scripts/project_week.py --season "${SEASON}" --week "${WEEK}"

echo ""
echo "=========================================="
echo "Done. Set your lineup:"
echo "  python scripts/start.py --season ${SEASON} --week ${WEEK} --roster <roster.yaml>"
echo "=========================================="
