#!/usr/bin/env python
"""CLI: train the projection engine and write out-of-fold projections (§4.6)."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.projection.train import train_and_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Train projection engine (M3).")
    parser.add_argument("--snapshot-id", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = train_and_write(snapshot_id=args.snapshot_id)
    print(f"projections written: {n}")


if __name__ == "__main__":
    main()
