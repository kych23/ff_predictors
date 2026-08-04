"""Task-0 feasibility spike (throwaway): does Yahoo's draftresults endpoint
update DURING a live draft, or only after it completes?

Usage:
    venv/bin/python scripts/spikes/yahoo_draft_probe.py <league_key>

<league_key> looks like "461.l.123456" — visible in the URL once you've
created a league and are on its page (football.fantasysports.yahoo.com/f1/<id>/...
or the "l/<id>" segment; the game part "461" for 2026 NFL is queried below).

Requires YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REDIRECT_URI in .env,
and a TLS-serving localhost matching YAHOO_REDIRECT_URI's port (this script
runs its own one-shot HTTPS listener for the OAuth callback — no need for
uvicorn to be running separately, but port 8000 must be free).
"""
from __future__ import annotations

import http.server
import os
import ssl
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from authlib.integrations.httpx_client import OAuth2Client  # noqa: E402

AUTHORIZE_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
FANTASY_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"

CLIENT_ID = os.environ["YAHOO_CLIENT_ID"]
CLIENT_SECRET = os.environ["YAHOO_CLIENT_SECRET"]
REDIRECT_URI = os.environ["YAHOO_REDIRECT_URI"]

_code_holder: dict = {}


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        qs = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(qs)
        _code_holder["code"] = params.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Authorized -- close this tab and check the terminal.")

    def log_message(self, *args):  # silence default request logging
        pass


def _wait_for_code(port: int, certfile: str, keyfile: str) -> None:
    server = http.server.HTTPServer(("localhost", port), _CallbackHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    server.handle_request()  # blocks for exactly one request, then returns
    server.server_close()


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: yahoo_draft_probe.py <league_key>")
        sys.exit(1)
    league_key = sys.argv[1]

    root = Path(__file__).resolve().parents[2]
    parsed_port = urllib.parse.urlparse(REDIRECT_URI).port or 8000

    client = OAuth2Client(CLIENT_ID, CLIENT_SECRET, redirect_uri=REDIRECT_URI)
    uri, _state = client.create_authorization_url(AUTHORIZE_URL)
    print("1. Open this URL, log in, and authorize:\n")
    print(uri)
    print("\n2. Waiting for the redirect (up to 180s)...")

    t = threading.Thread(
        target=_wait_for_code,
        args=(parsed_port, str(root / "localhost.pem"), str(root / "localhost-key.pem")),
        daemon=True,
    )
    t.start()
    t.join(timeout=180)
    code = _code_holder.get("code")
    if not code:
        print("No code received within 180s -- aborting.")
        sys.exit(1)

    token = client.fetch_token(TOKEN_URL, code=code)
    print(f"Got token (expires_in={token.get('expires_in')}s).")
    print(f"Polling {FANTASY_BASE}/league/{league_key}/draftresults every 10s.")
    print("Start (or resume) the live draft now -- watch the pick count below.\n")

    prev_count = -1
    for _ in range(90):  # 15 minutes
        resp = client.get(f"{FANTASY_BASE}/league/{league_key}/draftresults?format=json")
        ts = time.strftime("%H:%M:%S")
        if resp.status_code != 200:
            print(f"[{ts}] HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(10)
            continue
        count = resp.text.count('"player_key"')
        marker = "  <-- changed" if count != prev_count else ""
        print(f"[{ts}] picks so far: {count}{marker}")
        prev_count = count
        time.sleep(10)


if __name__ == "__main__":
    main()
