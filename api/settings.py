"""API runtime settings from environment. Kept tiny on purpose — full
pydantic-settings is not warranted for two values."""
from __future__ import annotations

import os


def allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]
