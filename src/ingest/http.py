"""Shared resilient HTTP client for CFBD and FFC (DrafterSpec.md §4.0/§4.2).

The one place all third-party HTTP resilience lives:
  * exponential-backoff retries,
  * per-source rate-limit throttling (CFBD free tier = 60 req/min),
  * on-disk JSON cache keyed by URL+params,
  * graceful degradation: on failure/rate-limit serve the last-good cached
    response with a loud staleness warning rather than crashing.

Draft day is a hard deadline — stale data beats no tool. ``nflreadpy`` manages
its own fetching and does not use this client.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Cache lives outside the source tree, under the repo data dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "http_cache"


class ResilientClient:
    """A throttled, cached, retrying HTTP GET client for one logical source."""

    def __init__(
        self,
        source: str,
        *,
        rate_limit_per_min: int = 60,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        timeout: float = 30.0,
        cache_ttl_seconds: Optional[float] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.source = source
        self.min_interval = 60.0 / rate_limit_per_min if rate_limit_per_min else 0.0
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_dir = (cache_dir or CACHE_DIR) / source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_ts = 0.0

    # -- cache helpers ----------------------------------------------------
    def _cache_path(self, url: str, params: Optional[Dict[str, Any]]) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path, *, respect_ttl: bool) -> Optional[Any]:
        if not path.exists():
            return None
        if respect_ttl and self.cache_ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > self.cache_ttl_seconds:
                return None
        try:
            return json.loads(path.read_text())["payload"]
        except Exception:
            return None

    def _write_cache(self, path: Path, payload: Any) -> None:
        try:
            path.write_text(json.dumps({"fetched_at": time.time(), "payload": payload}))
        except Exception as exc:  # cache write failure must not break ingest
            logger.warning("%s: cache write failed: %s", self.source, exc)

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    # -- public API -------------------------------------------------------
    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        *,
        immutable: bool = False,
    ) -> Any:
        """GET JSON with caching, throttling, retries, and stale-cache fallback.

        ``immutable=True`` (e.g. a past season) ignores TTL and serves cache
        forever once fetched.
        """
        cache_path = self._cache_path(url, params)
        cached = self._read_cache(cache_path, respect_ttl=not immutable)
        if cached is not None:
            return cached

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._throttle()
            self._last_request_ts = time.time()
            try:
                resp = httpx.get(url, params=params, headers=headers, timeout=self.timeout)
                if resp.status_code == 429:  # rate limited
                    raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
                resp.raise_for_status()
                payload = resp.json()
                self._write_cache(cache_path, payload)
                return payload
            except Exception as exc:  # noqa: BLE001 - retry on any transport error
                last_exc = exc
                wait = self.backoff_base ** attempt
                logger.warning(
                    "%s GET %s failed (attempt %d/%d): %s; retrying in %.1fs",
                    self.source, url, attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        # Graceful degradation: serve last-good cache regardless of TTL.
        stale = self._read_cache(cache_path, respect_ttl=False)
        if stale is not None:
            logger.error(
                "%s UNREACHABLE for %s — serving STALE cache (%s). Last error: %s",
                self.source, url, cache_path.name, last_exc,
            )
            return stale
        raise RuntimeError(f"{self.source} GET {url} failed and no cache exists") from last_exc
