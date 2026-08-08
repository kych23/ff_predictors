#!/usr/bin/env bash
#
# Scripted, non-interactive showcase of the draft cockpit.
#
#     bash scripts/demo.sh
#
# Drives `draft_night.py` through a real sequence — three opponent picks, the
# engine's recommendation at seat 4, an undo, and a ledger dump — using the
# actual bundle. Nothing is stubbed and nothing touches the network.
#
# Writes to a scratch session/ledger under data/demo/ so it never collides with
# a real draft in data/draft.json.
#
# To record a terminal GIF for the README:
#     asciinema rec demo.cast -c "bash scripts/demo.sh"
#     agg demo.cast docs/demo.gif
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found. Create the venv first:" >&2
    echo "    python -m venv venv && pip install -r requirements.txt" >&2
    exit 1
fi

BUNDLE="data/bundles/draft_night_bundle.parquet"
if [ ! -f "$BUNDLE" ]; then
    echo "error: $BUNDLE missing. Build it with:" >&2
    echo "    $PYTHON scripts/build_bundle.py --season 2026" >&2
    exit 1
fi

DEMO_DIR="data/demo"
mkdir -p "$DEMO_DIR"
rm -f "$DEMO_DIR/session.json" "$DEMO_DIR/ledger.sqlite"

if curl -sf -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "note: ollama detected — explanations will be model-generated."
else
    echo "note: ollama not running — explanations fall back to a plain table."
    echo "      start it with 'ollama serve && ollama pull qwen2.5:7b'."
fi
echo

# Three opponents pick, then seat 4 (us) is on the clock and the ladder fires
# automatically. We take the recommendation, inspect the resulting roster and
# board, print the hash-chained ledger, then `undo` to show that state is
# rebuilt by replaying the event log rather than by an inverse operation.
"$PYTHON" scripts/draft_night.py \
    --seat 4 \
    --session "$DEMO_DIR/session.json" \
    --ledger "$DEMO_DIR/ledger.sqlite" <<'COMMANDS'
Ja'Marr Chase
Bijan Robinson
Justin Jefferson
me Jahmyr Gibbs
roster
board 6
log
undo
quit
COMMANDS

echo
echo "demo complete — session and ledger left in $DEMO_DIR/ for inspection."
