"""Re-run section extraction on every filing already downloaded. No network needed.

Usage:  python -m src.ingest.reparse
Use it after changing the parser: it rewrites data/parsed/*.json from data/raw/*/*.html.
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
    args = parser.parse_args(argv)
    wanted = {t.upper() for t in args.tickers} if args.tickers else None

    for parsed_path in sorted((args.out / "parsed").glob("*.json")):
        record = json.loads(parsed_path.read_text(encoding="utf-8"))
        if wanted and record["ticker"].upper() not in wanted:
            continue
        raw_path = args.out / "raw" / record["ticker"] / f"{record['accession']}.html"
        if not raw_path.exists():
            print(f"skip {parsed_path.name}: raw file not found")
            continue
        sections, warnings = extract_sections(html_to_text(raw_path.read_text(encoding="utf-8")))
        record["sections"], record["warnings"] = sections, warnings
        parsed_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        print(parsed_path.name, {k: len(v) for k, v in sections.items()}, warnings or "")


if __name__ == "__main__":
    main()
