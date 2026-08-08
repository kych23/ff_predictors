"""Probes P1 and P4 (DraftEngineDesign.md §11.5).

P1 — does Yahoo serve draft_analysis (ADP) for PRIOR game keys 2021-2025?
     Blocks the opponent model's reach calculation. Fallback if it fails: FFC
     ADP for those seasons, with tau calibrated only on the Yahoo season.
P4 — polling latency (p50/p99), 429 behavior, and whether SEASON-TOTAL player
     stats are reachable. The latter blocks the §21.1 reconciliation gate,
     which is declared blocking and sits on a non-cuttable day.

Usage:
    venv/bin/python scripts/spikes/probe_yahoo.py

Authorization uses a PASTE-BACK flow: you open a URL, Yahoo redirects to
localhost:8000 (which nothing is listening on, so the browser shows
ERR_CONNECTION_REFUSED -- expected), and you paste the resulting URL back in.
No local server, no TLS, no timeout race. The token caches to
.yahoo_token.json (gitignored, mode 600) so re-runs need no browser.

Game keys and league keys are discovered from the logged-in user -- nothing is
hardcoded.

Requires YAHOO_CLIENT_ID / YAHOO_CLIENT_SECRET / YAHOO_REDIRECT_URI in .env.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT / ".env")

from authlib.integrations.httpx_client import OAuth2Client  # noqa: E402

AUTHORIZE_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
TOKEN_PATH = ROOT / ".yahoo_token.json"

SEASONS = [2021, 2022, 2023, 2024, 2025]
def _extract_code(raw: str) -> str | None:
    """Accept a full redirect URL, a bare query string, or just the code."""
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return None
    if "code=" in raw:
        qs = urllib.parse.urlparse(raw).query or raw.split("?", 1)[-1]
        got = urllib.parse.parse_qs(qs).get("code")
        if got:
            return got[0]
    # a bare code: Yahoo's are short alphanumerics with no separators
    return raw if raw.isalnum() else None


def _client() -> OAuth2Client:
    cid = os.environ["YAHOO_CLIENT_ID"].strip()
    sec = os.environ["YAHOO_CLIENT_SECRET"].strip()
    uri = os.environ["YAHOO_REDIRECT_URI"].strip()

    if TOKEN_PATH.exists():
        tok = json.loads(TOKEN_PATH.read_text())
        c = OAuth2Client(cid, sec, token=tok, redirect_uri=uri, token_endpoint=TOKEN_URL)
        try:
            if tok.get("expires_at", 0) < time.time() + 60:
                tok = c.refresh_token(TOKEN_URL, refresh_token=tok["refresh_token"])
                TOKEN_PATH.write_text(json.dumps(tok))
                TOKEN_PATH.chmod(0o600)
            print(f"Using cached token from {TOKEN_PATH.name} (no browser needed).")
            return c
        except Exception as e:  # refresh rejected -> fall through to full flow
            print(f"  (cached token unusable: {e}; re-authorizing)")

    c = OAuth2Client(cid, sec, redirect_uri=uri)
    auth_url, _ = c.create_authorization_url(AUTHORIZE_URL)

    print("\n" + "=" * 68)
    print("AUTHORIZE (paste-back flow — no local server, no TLS, no timeout)")
    print("=" * 68)
    print("\n1. Open this URL and authorize:\n")
    print(auth_url)
    print("\n2. Your browser will land on a 'This site can't be reached' /")
    print("   ERR_CONNECTION_REFUSED page. THAT IS EXPECTED AND FINE — nothing")
    print("   is listening on localhost:8000, and nothing needs to be. The")
    print("   authorization code is in the address bar.")
    print("\n3. Copy the WHOLE URL from the address bar and paste it below.")
    print("   (Codes expire in about 60s, so paste promptly. If it expires,")
    print("    just re-run this script.)\n")

    for attempt in range(3):
        raw = input("Paste redirect URL (or bare code): ")
        code = _extract_code(raw)
        if not code:
            print("  Could not find a code in that. Try again.\n")
            continue
        try:
            tok = c.fetch_token(TOKEN_URL, code=code)
        except Exception as e:
            print(f"  Token exchange failed: {e}")
            if attempt < 2:
                print("  Most likely the code expired. Re-open the URL above")
                print("  for a fresh one, then paste again.\n")
            continue
        TOKEN_PATH.write_text(json.dumps(tok))
        TOKEN_PATH.chmod(0o600)
        print(f"\n  Token cached to {TOKEN_PATH.name} (gitignored, mode 600).")
        print("  Re-runs will not need the browser again.\n")
        return c

    print("Could not complete authorization after 3 attempts. Aborting.")
    sys.exit(1)


def get(c, path, *, timed=False):
    url = f"{BASE}/{path}{'&' if '?' in path else '?'}format=json"
    t0 = time.perf_counter()
    r = c.get(url)
    dt = (time.perf_counter() - t0) * 1000
    return (r, dt) if timed else r


def _preflight(c) -> None:
    """Separate 'bad token' from 'app lacks Fantasy Sports permission'.

    game/nfl is public game metadata and needs no league membership, so if the
    token authenticates but that 401s, the problem is app-level permission.
    """
    ident = c.get("https://api.login.yahoo.com/openid/v1/userinfo")
    fant = c.get(f"{BASE}/game/nfl?format=json")
    if fant.status_code == 200:
        return
    print("\n" + "!" * 68)
    print("PREFLIGHT FAILED")
    print("!" * 68)
    print(f"  identity endpoint : HTTP {ident.status_code}"
          f"  ({'token OK' if ident.status_code == 200 else 'token BAD'})")
    print(f"  fantasy game/nfl  : HTTP {fant.status_code}")
    print(f"  {fant.text[:200]}")
    if ident.status_code == 200 and "additional_authorization_required" in fant.text:
        print("\n  DIAGNOSIS: the token is valid, but your Yahoo APP lacks the")
        print("  Fantasy Sports API permission. game/nfl needs no league")
        print("  membership, so this is app-level, not account-level.")
        print("\n  FIX:")
        print("    1. https://developer.yahoo.com/apps/  -> open your app")
        print("    2. API Permissions -> tick 'Fantasy Sports' -> Read")
        print("    3. Save, then:  rm .yahoo_token.json  and re-run this script")
        print("       (permissions are baked into the token at grant time)")
        print("\n    If it still 401s, create a NEW app with Fantasy Sports")
        print("    ticked from the start and swap YAHOO_CLIENT_ID/SECRET.")
    elif ident.status_code != 200:
        print("\n  DIAGNOSIS: the token itself is rejected. Delete")
        print("  .yahoo_token.json and re-authorize.")
    sys.exit(1)


def main() -> None:
    c = _client()
    _preflight(c)

    # ---- discover game + league keys (nothing hardcoded) -------------------
    print("\n" + "=" * 68)
    print("DISCOVERY: your NFL game keys and league keys")
    print("=" * 68)
    r = get(c, "users;use_login=1/games;game_codes=nfl/leagues")
    if r.status_code != 200:
        print(f"FAIL HTTP {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    blob = r.json()
    found: dict[int, list[str]] = {}
    games: dict[int, str] = {}

    def walk(o):
        if isinstance(o, dict):
            if "season" in o and "game_key" in o:
                games[int(o["season"])] = str(o["game_key"])
            if "league_key" in o and "name" in o:
                s = int(o.get("season", 0))
                found.setdefault(s, []).append(f"{o['league_key']}  ({o['name']})")
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(blob)
    for s in sorted(games):
        print(f"  {s}: game_key={games[s]}")
    print("\n  leagues:")
    for s in sorted(found):
        for lk in found[s]:
            print(f"    {s}: {lk}")
    if not found:
        print("    (none found — is this the Yahoo account that owns the league?)")

    # ---- P1: draft_analysis on prior game keys ----------------------------
    print("\n" + "=" * 68)
    print("P1: draft_analysis (ADP) for prior seasons")
    print("=" * 68)
    p1: dict[int, str] = {}
    for s in SEASONS:
        gk = games.get(s)
        if not gk:
            p1[s] = "NO GAME KEY"
            print(f"  {s}: no game key on this account")
            continue
        r = get(c, f"game/{gk}/players;start=0;count=10/draft_analysis")
        if r.status_code != 200:
            p1[s] = f"HTTP {r.status_code}"
            print(f"  {s} (gk={gk}): HTTP {r.status_code} {r.text[:120]}")
            continue
        txt = r.text
        n = txt.count('"average_pick"')
        # a zeroed-out season returns the field but with "-" or 0.0 values
        vals = []
        for seg in txt.split('"average_pick":')[1:]:
            v = seg.split(",")[0].strip().strip('"')
            try:
                vals.append(float(v))
            except ValueError:
                pass
        live = [v for v in vals if v > 0]
        p1[s] = f"{len(live)}/{n} populated"
        print(f"  {s} (gk={gk}): {n} average_pick fields, {len(live)} populated"
              f"{'  <-- USABLE' if live else '  <-- ZEROED/EMPTY'}")
        if live:
            print(f"        sample values: {sorted(live)[:5]}")

    # ---- P4a: season-total stats reachability ----------------------------
    print("\n" + "=" * 68)
    print("P4a: season-total player stats (blocks the §21.1 gate)")
    print("=" * 68)
    gk25 = games.get(2025)
    if gk25:
        r = get(c, f"game/{gk25}/players;start=0;count=5/stats;type=season")
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            has = '"stats"' in r.text
            print(f"  stats block present: {has}  <-- {'REACHABLE' if has else 'NOT REACHABLE'}")
            print(f"  sample: {r.text[:300]}")
        else:
            print(f"  {r.text[:300]}")
    else:
        print("  skipped (no 2025 game key)")

    # ---- P4b: latency + 429 ----------------------------------------------
    print("\n" + "=" * 68)
    print("P4b: polling latency and 429 behavior")
    print("=" * 68)
    lk = None
    for s in sorted(found, reverse=True):
        lk = found[s][0].split()[0]
        break
    if not lk:
        print("  skipped (no league key discovered)")
    else:
        print(f"  polling {lk}/draftresults x40, no pacing...")
        lat, codes = [], {}
        for i in range(40):
            r, dt = get(c, f"league/{lk}/draftresults", timed=True)
            lat.append(dt)
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            if r.status_code == 429:
                print(f"    429 at request {i+1} after {sum(lat)/1000:.1f}s")
                print(f"    Retry-After: {r.headers.get('Retry-After')}")
                break
        lat.sort()
        print(f"  n={len(lat)}  status counts={codes}")
        print(f"  p50={statistics.median(lat):.0f}ms  "
              f"p95={lat[int(.95*len(lat))-1]:.0f}ms  max={lat[-1]:.0f}ms")
        budget = 500
        print(f"  verdict: {'OK' if lat[int(.95*len(lat))-1] < budget else 'TOO SLOW'} "
              f"against a {budget}ms poll budget "
              f"(manual entry is primary regardless — §19)")

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("SUMMARY — paste this back")
    print("=" * 68)
    print(f"  game keys found : {dict(sorted(games.items()))}")
    print(f"  P1 by season    : {p1}")
    usable = [s for s, v in p1.items() if "populated" in v and not v.startswith("0/")]
    print(f"  P1 verdict      : {'PASS' if len(usable) >= 4 else 'DEGRADED' if usable else 'FAIL'}"
          f"  ({len(usable)}/5 seasons usable)")
    if len(usable) < 4:
        print("    -> fallback per §11.5: FFC ADP for the missing seasons;")
        print("       tau calibrated only on seasons with Yahoo ADP.")


if __name__ == "__main__":
    main()
