"""Custom tier rankings — "My Board".

The user's own research, stored server-side as one JSON document per board.
Deliberately **standalone**: nothing here reaches a recommendation. The engine
prices picks in expected dollars from its own objective, and a hand-built board
is a different kind of claim — an opinion about players rather than a valuation
of picks. Mixing them would make "the engine beats the market" circular.

What this layer owns:

* `schema`  — the document shape, its validation, and its one migration seam
* `store`   — atomic reads and writes, revision conflicts, path containment
* `data`    — the warm read-only frames every request reads from
* `seed`    — building a starting tier list from the engine or the market
* `detail`  — one player's full card
"""
