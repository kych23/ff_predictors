"""Discover your Yahoo league and write the config the cockpit needs.

Run this AFTER `scripts/spikes/probe_yahoo.py` has cached a token. It fills the
two gaps that keep `source: yahoo` disabled:

* `yahoo.league_key` in `config/strategy.yaml`
* `yahoo.manager_map`, which the adapter reads as **team_key -> seat index**

The seat map is the part worth being careful about. `strategy.yaml` documents
`manager_map` as team_key -> manager id, and nothing in the repo maps a manager
to a seat — so `YahooSource._validate_seat_map` treats the values as seats and
refuses to start if they are not integers. Getting this wrong does not fail
loudly at draft time: it attributes every pick to the wrong team, which
corrupts the roster the recommender is optimising against.

So the seats come from Yahoo's own draft order (`team.draft_position`, 1-based)
converted to the cockpit's 0-based seats, and the script PRINTS the mapping for
you to eyeball against the Yahoo draft-order page before anything is written.

    venv/bin/python scripts/spikes/probe_yahoo.py     # once, browser login
    venv/bin/python scripts/setup_yahoo.py            # discover + preview
    venv/bin/python scripts/setup_yahoo.py --write    # commit to strategy.yaml

Nothing here touches the draft path. Manual entry keeps working throughout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
TOKEN_PATH = ROOT / ".yahoo_token.json"
STRATEGY = ROOT / "config" / "strategy.yaml"


def _client():
    try:
        from authlib.integrations.httpx_client import OAuth2Client
    except ImportError as exc:
        raise SystemExit("authlib missing: venv/bin/pip install authlib") from exc
    if not TOKEN_PATH.exists():
        raise SystemExit(
            f"no token at {TOKEN_PATH.name}. Run:\n"
            f"  venv/bin/python scripts/spikes/probe_yahoo.py")
    return OAuth2Client(token=json.loads(TOKEN_PATH.read_text()))


def _walk(payload: Any, key: str):
    """Yahoo nests everything under numerically-keyed objects and mixes dicts
    with lists of fragments. Walk rather than index a shape we cannot pin."""
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if key in node:
                inner = node[key]
                merged: dict[str, Any] = {}
                for part in (inner if isinstance(inner, list) else [inner]):
                    if isinstance(part, Mapping):
                        merged.update(part)
                    elif isinstance(part, list):
                        for sub in part:
                            if isinstance(sub, Mapping):
                                merged.update(sub)
                if merged:
                    yield merged
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _get(client, url: str) -> Any:
    response = client.get(url, timeout=20.0)
    if response.status_code == 401:
        raise SystemExit(
            "401 from Yahoo. The token is stale or the app lacks Fantasy "
            "Sports permission. Delete .yahoo_token.json and re-run "
            "scripts/spikes/probe_yahoo.py — permissions are baked in at grant.")
    response.raise_for_status()
    return response.json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="write to config/strategy.yaml (default: preview)")
    ap.add_argument("--league-key", help="skip discovery and use this key")
    args = ap.parse_args()

    client = _client()

    league_key = args.league_key
    if not league_key:
        print("[1/3] finding your NFL leagues...")
        payload = _get(client, f"{BASE}/users;use_login=1/games;game_keys=nfl/"
                               f"leagues?format=json")
        leagues = [lg for lg in _walk(payload, "league") if lg.get("league_key")]
        if not leagues:
            raise SystemExit(
                "no NFL leagues on this account. If you expected one, the app "
                "may lack Fantasy Sports permission — see probe_yahoo.py.")
        for i, lg in enumerate(leagues):
            print(f"   [{i}] {lg.get('name')!r}  key={lg['league_key']}  "
                  f"teams={lg.get('num_teams')}  season={lg.get('season')}")
        if len(leagues) == 1:
            league_key = leagues[0]["league_key"]
            print(f"   only one league; using {league_key}")
        else:
            choice = input("   which league? [index]: ").strip()
            league_key = leagues[int(choice)]["league_key"]

    print(f"\n[2/3] reading teams and draft order for {league_key}...")
    payload = _get(client, f"{BASE}/league/{league_key}/teams?format=json")
    teams = [t for t in _walk(payload, "team") if t.get("team_key")]
    if not teams:
        raise SystemExit("no teams returned; is the league key right?")

    rows = []
    for team in teams:
        position = team.get("draft_position")
        rows.append({
            "team_key": str(team["team_key"]),
            "name": str(team.get("name", "?")),
            "draft_position": None if position is None else int(position),
        })
    rows.sort(key=lambda r: (r["draft_position"] is None, r["draft_position"]))

    missing = [r for r in rows if r["draft_position"] is None]
    print(f"   {len(rows)} teams, {len(rows) - len(missing)} with a draft "
          f"position")
    print(f"\n   {'seat':>4}  {'draft_pos':>9}  {'team_key':<16} name")
    for r in rows:
        seat = "" if r["draft_position"] is None else r["draft_position"] - 1
        print(f"   {str(seat):>4}  {str(r['draft_position']):>9}  "
              f"{r['team_key']:<16} {r['name']}")

    if missing:
        print(f"\n   WARNING: {len(missing)} team(s) have no draft_position. "
              f"Yahoo publishes it only once the draft order is set. Until "
              f"then the seat map is incomplete and the adapter will refuse "
              f"to start — which is correct, because a guessed seat silently "
              f"attributes picks to the wrong roster.")

    manager_map = {r["team_key"]: r["draft_position"] - 1
                   for r in rows if r["draft_position"] is not None}

    print("\n[3/3] config to apply:")
    print(f"   yahoo.league_key: {league_key}")
    print(f"   yahoo.manager_map: {len(manager_map)} entries "
          f"(team_key -> 0-based seat)")
    print("\n   CHECK THIS AGAINST YOUR YAHOO DRAFT ORDER PAGE before "
          "--write.\n   A wrong seat does not error; it drafts to the wrong "
          "team all night.")

    if not args.write:
        print("\n   preview only — pass --write to apply")
        return

    import yaml

    raw = yaml.safe_load(STRATEGY.read_text())
    raw.setdefault("yahoo", {})
    raw["yahoo"]["league_key"] = league_key
    raw["yahoo"]["manager_map"] = manager_map
    STRATEGY.write_text(yaml.safe_dump(raw, sort_keys=False, width=100))
    print(f"\n   wrote {STRATEGY.relative_to(ROOT)}")
    print("   NOTE: strategy.yaml is hashed into decision_version, so now:")
    print("     venv/bin/python scripts/build_bundle.py --season 2026")
    print("     venv/bin/python scripts/regen_parity_golden.py --note 'yahoo'")


if __name__ == "__main__":
    main()
