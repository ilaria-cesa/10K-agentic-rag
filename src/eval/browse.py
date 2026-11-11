"""Find passages in the corpus to base golden-set questions on (keyword search, no model).

Usage:
  python -m src.eval.browse "catastrophe losses" --ticker TRV --date 2025-12-31 --top 5
  python -m src.eval.browse --show TRV_2025-12-31_mdna_0073     # print one chunk in full
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(text: str, terms: list[str]) -> float:
    """Number of distinct query terms present, then total hits, normalised by length."""
    lowered = text.lower()
    present = sum(term in lowered for term in terms)
    hits = sum(lowered.count(term) for term in terms)
    return present + hits / (1 + len(text) / 1000)


def search(chunks: list[dict], query: str, ticker=None, date=None, section=None, top=5):
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    pool = [
        c
        for c in chunks
        if (not ticker or c["ticker"] == ticker.upper())
        and (not date or c["report_date"] == date)
        and (not section or c["section"] == section)
    ]
    ranked = sorted(pool, key=lambda c: score(c["text"], terms), reverse=True)
    return [c for c in ranked[:top] if score(c["text"], terms) > 0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks_1600.jsonl"))
    parser.add_argument("--ticker")
    parser.add_argument("--date")
    parser.add_argument("--section", choices=["business", "risk_factors", "mdna"])
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--show", help="print the full text of this chunk_id")
    args = parser.parse_args(argv)

    chunks = load_jsonl(args.chunks)
    if args.show:
        match = next((c for c in chunks if c["chunk_id"] == args.show), None)
        print(match["embed_text"] if match else f"chunk not found: {args.show}")
        return
    for chunk in search(chunks, args.query, args.ticker, args.date, args.section, args.top):
        print(f"\n[{chunk['chunk_id']}]")
        print(chunk["text"][:500].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
