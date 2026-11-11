"""Build data/chunks_<max_chars>.jsonl from data/parsed/*.json (CPU only, no models).

Usage:  python -m src.chunking.run [--max-chars 1600] [--overlap 200] [--tables]
With --tables the input is data/parsed_tables/ and the output data/chunks_tables_<max_chars>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.chunking.chunker import ChunkConfig, build_chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--tables", action="store_true", help="use the table-aware parsed text")
    args = parser.parse_args(argv)
    source, prefix = ("parsed_tables", "chunks_tables") if args.tables else ("parsed", "chunks")

    cfg = ChunkConfig(max_chars=args.max_chars, overlap_chars=args.overlap)
    rows = []
    for path in sorted((args.out / source).glob("*.json")):
        rows.extend(build_chunks(json.loads(path.read_text(encoding="utf-8")), cfg))

    target = args.out / f"{prefix}_{cfg.max_chars}.jsonl"
    with open(target, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    sizes = sorted(r["n_chars"] for r in rows)
    print(f"wrote {len(rows)} chunks to {target}")
    print("per section:", dict(Counter(r["section"] for r in rows)))
    print(
        f"chars  min={sizes[0]}  median={int(statistics.median(sizes))}  "
        f"p95={sizes[int(len(sizes) * 0.95)]}  max={sizes[-1]}"
    )
    n_tables = sum(r["text"].startswith("Table") for r in rows)
    print(f"table chunks: {n_tables} ({n_tables / len(rows):.1%})")
    table_like = sum(r["digit_ratio"] > 0.3 for r in rows)
    print(f"table-like chunks (digit ratio > 0.3): {table_like} ({table_like / len(rows):.1%})")


if __name__ == "__main__":
    main()
