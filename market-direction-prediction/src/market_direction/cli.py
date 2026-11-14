"""Run with `python -m market_direction.cli fetch|verify|run`."""
import argparse
import json

from .data import fetch, verify


def main():
    parser = argparse.ArgumentParser(description="Market direction research pipeline")
    parser.add_argument("command", choices=["fetch", "verify", "run", "extension-fetch", "extension-run", "profitability"])
    args = parser.parse_args()
    if args.command == "fetch": print(json.dumps(fetch(), indent=2))
    elif args.command == "verify": print(json.dumps(verify(), indent=2))
    elif args.command == "profitability":
        from .profitability import run_profitability
        run_profitability()
    elif args.command == "extension-fetch":
        from .extension import fetch_extension
        fetch_extension()
    elif args.command == "extension-run":
        from .extension import run_extension
        run_extension()
    else:
        from .experiment import run
        run()


if __name__ == "__main__": main()
