"""Summarise every parsed filing in data/parsed (no network needed).

Usage:  python -m src.ingest.summary
Writes data/summary.csv and prints one row per filing plus a short health report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.ingest.parse import MIN_SECTION_CHARS, WANTED_ITEMS


def build_rows(parsed_dir: Path) -> list[dict]:
    rows = []
    for path in parsed_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        row = {"ticker": record["ticker"], "report_date": record["report_date"]}
        row.update({n: len(record["sections"].get(n, "")) for n in WANTED_ITEMS.values()})
        row["ok"] = all(row[n] >= MIN_SECTION_CHARS for n in WANTED_ITEMS.values())
        row["warnings"] = "; ".join(record["warnings"])
        rows.append(row)
    rows.sort(key=lambda r: (r["ticker"], r["report_date"]), reverse=False)
    return rows


def health_report(rows: list[dict]) -> list[str]:
    per_ticker = Counter(r["ticker"] for r in rows)
    bad = sorted({r["ticker"] for r in rows if not r["ok"]})
    lines = [f"filings: {len(rows)}, companies: {len(per_ticker)}"]
    lines.append(
        f"companies with fewer than 2 filings: {sorted(t for t, c in per_ticker.items() if c < 2)}"
    )
    lines.append(f"companies with a missing or short section: {bad}")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    rows = build_rows(args.out / "parsed")
    if not rows:
        print("No parsed filings found. Run `python -m src.ingest.run` first.")
        return
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(",".join(str(v) for v in row.values()))
    print()
    print("\n".join(health_report(rows)))


if __name__ == "__main__":
    main()
