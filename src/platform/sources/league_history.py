"""Parse the hand-transcribed league draft history (§9, §14.2).

``notes/league-history.md`` holds one section per manager, one table per season:
``Round | Overall Pick | Player | Position``, with the draft slot in the
subheading. It is the only record of what this specific room actually did, and
Yahoo's API is blocked, so it is parsed rather than fetched.

Every pick is re-checked against the snake sequence implied by its stated slot;
a transcription error in the overall-pick column would otherwise corrupt the
choice sets silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

DEFAULT_PATH: Final = Path("notes/league-history.md")

_MANAGER = re.compile(r"^##\s+Manager:\s*(.+?)\s*$")
_SEASON = re.compile(r"^###\s+(\d{4})\s*—\s*[\"“](.*?)[\"”]\s*\(Slot\s+(\d+)\)")
_PICK = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*([A-Z/]+)\s*\|"
)


@dataclass(frozen=True)
class HistoryLoad:
    picks: pd.DataFrame
    snake_errors: list[str]

    @property
    def managers(self) -> list[str]:
        return sorted(self.picks["manager"].unique())


def _expected_overall(rnd: int, slot: int, teams: int) -> int:
    return (rnd - 1) * teams + slot if rnd % 2 == 1 else rnd * teams - slot + 1


def load_history(path: Path | None = None, *, teams: int = 12) -> HistoryLoad:
    text = (path or DEFAULT_PATH).read_text(encoding="utf-8")
    rows: list[dict] = []
    errors: list[str] = []
    manager = season = slot = None

    for line in text.splitlines():
        m = _MANAGER.match(line)
        if m:
            manager, season, slot = m.group(1), None, None
            continue
        m = _SEASON.match(line)
        if m:
            season, slot = int(m.group(1)), int(m.group(3))
            continue
        m = _PICK.match(line)
        if m and manager and season:
            rnd, overall = int(m.group(1)), int(m.group(2))
            expected = _expected_overall(rnd, slot, teams)
            if overall != expected:
                errors.append(
                    f"{manager} {season} R{rnd}: overall {overall}, "
                    f"snake for slot {slot} says {expected}"
                )
            rows.append({
                "manager": manager, "season": season, "slot": slot,
                "round": rnd, "overall": overall,
                "player_name": m.group(3).strip(), "position": m.group(4).strip(),
            })

    return HistoryLoad(picks=pd.DataFrame(rows), snake_errors=errors)
