"""Schema and validator for the hand-checked question set (questions/questions.jsonl).

One JSON object per line:
  id                str, unique and numbered in order, e.g. "Q001"
  split             "dev" | "test" | "agentic": development questions (used to build and tune the
                    systems), held-out test questions, and questions about several companies
  question          str
  language          "en" | "it"
  type              "fact" | "numeric" | "comparison" | "multi_hop" | "unanswerable" | "false_premise"
  tickers           list[str], the companies the question is about
  report_dates      list[str], filing periods, e.g. ["2025-12-31"]
  expected          "answer" | "refuse"
  answer            str, the correct answer written by hand ("" when expected == "refuse")
  evidence_chunk_ids list[str], chunks that contain the answer ([] when expected == "refuse")
  category          str or null, only for the "agentic" split (multi_company, description, ...)
  notes             str, optional

Rules that keep the evaluation honest:
  * "unanswerable" questions must be about topics the documents really do not cover, and must
    use expected == "refuse" with no evidence;
  * every other question needs a written answer and at least one evidence chunk that exists;
  * multi_hop questions need evidence from at least two chunks; a comparison should have two,
    but one chunk is accepted (with a warning) when a single table holds both periods;
  * false_premise questions state something untrue and expect the answer to correct it;
  * name the filing year in the question: the 2024 and 2025 filings repeat many passages, so a
    question without a year can have several valid answers.

Usage:  python -m src.eval.golden [--golden questions/questions.jsonl] [--chunks data/chunks_1600.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SPLITS = {"dev", "test", "agentic"}
TYPES = {"fact", "numeric", "comparison", "multi_hop", "unanswerable", "false_premise"}
LANGUAGES = {"en", "it"}
_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d\d(?!\d)")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_golden(path: Path, split: str | None = None) -> list[dict]:
    """All questions in the file, or only those of one split ("dev", "test" or "agentic")."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if split is None or r.get("split") == split]


def validate(rows: list[dict], chunks: list[dict]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). An empty `errors` list means the file is usable."""
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    periods = {(c["ticker"], c["report_date"]) for c in chunks}
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for row in rows:
        qid = row.get("id", "<missing id>")

        def err(message: str, qid: str = qid) -> None:
            errors.append(f"{qid}: {message}")

        if qid in seen:
            err("duplicate id")
        seen.add(qid)
        if "split" in row and row["split"] not in SPLITS:
            err(f"split must be one of {sorted(SPLITS)}")
        if not str(row.get("question", "")).strip():
            err("empty question")
        if row.get("language") not in LANGUAGES:
            err(f"language must be one of {sorted(LANGUAGES)}")
        if row.get("type") not in TYPES:
            err(f"type must be one of {sorted(TYPES)}")
        tickers, dates = row.get("tickers", []), row.get("report_dates", [])
        # Companies close their fiscal year at different dates, so with several companies a date
        # only has to exist for one of them (and every company needs at least one listed date).
        for date in dates:
            if not any((t, date) in periods for t in tickers):
                err(f"no filing for {' / '.join(tickers)} {date} in the corpus")
        for ticker in tickers:
            if dates and not any((ticker, d) in periods for d in dates):
                err(f"no filing for {ticker} on any listed date in the corpus")
        if not row.get("tickers") or not row.get("report_dates"):
            err("tickers and report_dates must not be empty")

        expected = row.get("expected")
        evidence = row.get("evidence_chunk_ids", [])
        if expected not in {"answer", "refuse"}:
            err("expected must be 'answer' or 'refuse'")
        elif expected == "refuse":
            if evidence or str(row.get("answer", "")).strip():
                err("a 'refuse' question must have no answer and no evidence")
            if row.get("type") != "unanswerable":
                err("expected 'refuse' requires type 'unanswerable'")
        else:
            if row.get("type") == "unanswerable":
                err("type 'unanswerable' requires expected 'refuse'")
            if not str(row.get("answer", "")).strip():
                err("missing written answer")
            if not evidence:
                err("missing evidence_chunk_ids")
            if len(set(evidence)) < 2:
                if row.get("type") == "multi_hop":
                    err("type 'multi_hop' needs evidence from at least two chunks")
                elif row.get("type") == "comparison":
                    warnings.append(f"{qid}: comparison backed by a single chunk")
        for chunk_id in evidence:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                err(f"evidence chunk does not exist: {chunk_id}")
            elif chunk["ticker"] not in row.get("tickers", []):
                err(f"evidence chunk {chunk_id} belongs to a company not listed in tickers")
        if row.get("language") == "en" and not _YEAR_RE.search(str(row.get("question", ""))):
            warnings.append(f"{qid}: no year in the question (2024/2025 passages repeat)")

    counts = Counter(r.get("type") for r in rows)
    if rows and counts["unanswerable"] / len(rows) < 0.1:
        warnings.append("fewer than 10% unanswerable questions: refusal will be barely measurable")
    return errors, warnings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("questions/questions.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks_1600.jsonl"))
    args = parser.parse_args(argv)

    rows = load_golden(args.golden)
    chunks = load_jsonl(args.chunks)
    errors, warnings = validate(rows, chunks)
    print(f"{len(rows)} questions | by type: {dict(Counter(r.get('type') for r in rows))}")
    print(f"by language: {dict(Counter(r.get('language') for r in rows))}")
    for line in errors:
        print("ERROR  ", line)
    for line in warnings:
        print("WARNING", line)
    print("OK" if not errors else f"{len(errors)} error(s)")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
