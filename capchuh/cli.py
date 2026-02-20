"""CLI entry point for capchuh.

Usage:
    python -m capchuh "Some text to analyze"
    capchuh "Some text to analyze"
"""

from __future__ import annotations

import argparse
import json

from capchuh import analyze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the generic-ness of text.",
    )
    parser.add_argument("text", help="The text to analyze.")
    args = parser.parse_args()

    result = analyze(args.text)
    print(json.dumps(result.raw_output, indent=2))


if __name__ == "__main__":
    main()
