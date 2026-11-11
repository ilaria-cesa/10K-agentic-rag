"""Metrics for the RAG baseline: retrieval quality and cheap, rule-based answer checks.

Nothing here uses an LLM judge. The checks are proxies:
  * retrieval: are the golden evidence chunks among the top-k retrieved chunks?
  * refusal: did the system refuse exactly when the golden set says it should?
  * citations: are the cited ids real (among the provided chunks) and do they hit the evidence?
  * number match: do the key numbers of the golden answer appear in the generated answer?
A "number match" only says the figures agree; a human (or a judge, later) must still read
the answers, in particular for false_premise questions that need a correction.
"""

from __future__ import annotations

import re
from collections import defaultdict

KS = (1, 3, 5, 10)
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def retrieval_metrics(evidence: list[str], retrieved: list[str], ks: tuple[int, ...] = KS) -> dict:
    """Recall@k, hit@k (any evidence chunk), all@k (every evidence chunk) and reciprocal rank."""
    gold = set(evidence)
    out: dict[str, float] = {}
    for k in ks:
        found = gold & set(retrieved[:k])
        out[f"recall@{k}"] = len(found) / len(gold)
        out[f"hit@{k}"] = float(bool(found))
        out[f"all@{k}"] = float(found == gold)
    ranks = [i + 1 for i, chunk_id in enumerate(retrieved) if chunk_id in gold]
    out["mrr"] = 1 / ranks[0] if ranks else 0.0
    return out


_AMOUNT = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(trillion|billion|million|thousand|%|percent)?", re.IGNORECASE
)
_SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}


def _is_year(raw: str, value: float) -> bool:
    return "," not in raw and "." not in raw and 1900 <= value <= 2100


def key_numbers(text: str) -> list[tuple[float, str]]:
    """(value, kind) pairs. Amounts with a scale word are converted to units ($38,219 million
    equals 38.219 billion); percentages keep their own kind; years and one-digit numbers are
    skipped because they would match by accident."""
    found = []
    for raw, unit in _AMOUNT.findall(text):
        raw = raw.rstrip(".")  # a trailing dot is sentence punctuation, not a decimal point
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        unit = unit.lower()
        if unit in _SCALE:
            found.append((value * _SCALE[unit], "amount"))
        elif unit in {"%", "percent"}:
            found.append((value, "percent"))
        elif value >= 10 and not _is_year(raw, value):
            found.append((value, "plain"))
    return found


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b))


def numbers_match(gold_answer: str, answer: str, tolerance: float = 0.01) -> bool | None:
    """True when every key number of the gold answer appears in the answer (within 1% for
    scaled amounts, exact otherwise); None if the gold answer has no key number. Rounded or
    re-scaled figures count as a match; a figure that is merely close in kind does not."""
    wanted = key_numbers(gold_answer)
    if not wanted:
        return None
    given = key_numbers(answer)
    for value, kind in wanted:
        tol = tolerance if kind == "amount" else 1e-9
        if not any(k == kind and _close(value, v, tol) for v, k in given):
            return False
    return True


def score_row(question: dict, parsed: dict, retrieved: list[str]) -> dict:
    """Combine the golden row, the parsed model answer and the ranked retrieved ids."""
    should_refuse = question["expected"] == "refuse"
    row = {
        "id": question["id"],
        "type": question["type"],
        "refused": parsed["refused"],
        "behaviour_ok": should_refuse == parsed["refused"],
        "invalid_citation": bool(parsed["invalid_citations"]),
    }
    if should_refuse:
        return row
    cited = set(parsed["cited_ids"])
    row["cites_evidence"] = bool(cited & set(question["evidence_chunk_ids"]))
    row["uncited_answer"] = not parsed["refused"] and not cited
    row["numbers_match"] = (
        None if parsed["refused"] else numbers_match(question["answer"], parsed["answer"])
    )
    row.update(retrieval_metrics(question["evidence_chunk_ids"], retrieved))
    return row


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _numeric_columns(items: list[dict]) -> dict[str, list]:
    """Columns whose values are all numbers, booleans or None (text and lists are skipped)."""
    columns: dict[str, list] = {}
    for key in sorted({key for item in items for key in item}):
        values = [item.get(key) for item in items]
        if all(v is None or isinstance(v, (bool, int, float)) for v in values):
            columns[key] = values
    return columns


def summarise(rows: list[dict]) -> dict[str, dict]:
    """Mean of every numeric/bool column, overall and per question type."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["type"]].append(row)
    summary = {}
    for name, items in groups.items():
        summary[name] = {"n": len(items)} | {
            col: _mean(values) for col, values in _numeric_columns(items).items()
        }
    return summary
