"""Sync the canonical dashboard into the portfolio's GitHub Pages directory.

Run after regenerating research artifacts, then commit both directories.
Use --check in CI to prevent a stale published dashboard.
"""
import argparse
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
SOURCE = PROJECT / "docs"
TARGET = REPOSITORY / "docs/market-direction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not (REPOSITORY / ".git").exists():
        raise SystemExit("Run this script from the QuantDS-Projects checkout containing market-direction-prediction/.")
    files = [p for p in SOURCE.rglob("*") if p.is_file()]
    if args.check:
        stale = [str(p.relative_to(SOURCE)) for p in files if not (TARGET / p.relative_to(SOURCE)).exists() or p.read_bytes() != (TARGET / p.relative_to(SOURCE)).read_bytes()]
        if stale: raise SystemExit("Stale Pages files: " + ", ".join(stale))
        print(f"Verified {len(files)} canonical dashboard files match the Pages directory")
    else:
        shutil.copytree(SOURCE, TARGET, dirs_exist_ok=True)
        print(f"Synced {len(files)} dashboard files to {TARGET}")


if __name__ == "__main__": main()
