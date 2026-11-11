"""Write one plain-text file per filing, with every chunk id visible, for reading by hand.

Usage:  python -m src.eval.export_readable      ->  data/readable/<TICKER>_<report_date>.txt

Chunks overlap by about 200 characters, so a few lines appear twice around each marker.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.eval.browse import load_jsonl

_SECTION_ORDER = ["business", "risk_factors", "mdna"]
_TITLES = {
    "business": "ITEM 1 BUSINESS",
    "risk_factors": "ITEM 1A RISK FACTORS",
    "mdna": "ITEM 7 MD&A",
}


def render(filing_chunks: list[dict]) -> str:
    first = filing_chunks[0]
    lines = [f"{first['company']} | Form 10-K | period ended {first['report_date']}", ""]
    for section in _SECTION_ORDER:
        part = sorted(
            (c for c in filing_chunks if c["section"] == section), key=lambda c: c["position"]
        )
        if not part:
            continue
        lines += ["#" * 100, f"# {_TITLES[section]}", "#" * 100, ""]
        for chunk in part:
            lines += [f"=== [{chunk['chunk_id']}] ===", chunk["text"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks_1600.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/readable"))
    args = parser.parse_args(argv)

    by_filing = defaultdict(list)
    for chunk in load_jsonl(args.chunks):
        by_filing[(chunk["ticker"], chunk["report_date"])].append(chunk)
    args.out.mkdir(parents=True, exist_ok=True)
    for (ticker, date), filing_chunks in sorted(by_filing.items()):
        (args.out / f"{ticker}_{date}.txt").write_text(render(filing_chunks), encoding="utf-8")
    print(f"wrote {len(by_filing)} files to {args.out}")


if __name__ == "__main__":
    main()
