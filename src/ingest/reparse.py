"""Re-run section extraction on every filing already downloaded. No network needed.

Usage:  python -m src.ingest.reparse
Use it after changing the parser: it rewrites data/parsed/*.json from data/raw/*/*.html.
With --tables structured the tables are written value by value (row + column) and the result
goes to data/parsed_tables/, so the plain-text version in data/parsed/ is left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest.parse import extract_sections, html_to_text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--tickers", nargs="+", help="only these tickers (default: all)")
    parser.add_argument("--tables", choices=["flat", "structured", "structured2"], default="flat")
    args = parser.parse_args(argv)
    out_dir = (
        args.out
        / {"flat": "parsed", "structured": "parsed_tables", "structured2": "parsed_tables_v2"}[
            args.tables
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {t.upper() for t in args.tickers} if args.tickers else None

    for parsed_path in sorted((args.out / "parsed").glob("*.json")):
        record = json.loads(parsed_path.read_text(encoding="utf-8"))
        if wanted and record["ticker"].upper() not in wanted:
            continue
        raw_path = args.out / "raw" / record["ticker"] / f"{record['accession']}.html"
        if not raw_path.exists():
            print(f"skip {parsed_path.name}: raw file not found")
            continue
        html = raw_path.read_text(encoding="utf-8")
        sections, warnings = extract_sections(html_to_text(html, tables=args.tables))
        record["sections"], record["warnings"] = sections, warnings
        (out_dir / parsed_path.name).write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        print(parsed_path.name, {k: len(v) for k, v in sections.items()}, warnings or "")


if __name__ == "__main__":
    main()
