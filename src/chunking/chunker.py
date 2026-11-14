"""Split extracted 10-K sections into overlapping, paragraph-aligned chunks.

Size is measured in characters (about 4 characters per token for English prose, so the
default of 1600 characters is roughly 400 tokens). The estimate is checked against the real
tokenizer on Colab; no model is needed to build chunks.

Design choices:
  * lines (paragraphs) are never cut in the middle unless a single line exceeds the maximum;
  * a chunk with less than `min_chars` of new content keeps absorbing lines, so it can exceed
    `max_chars` by up to `min_chars` (this avoids chunks made of a lone heading);
  * consecutive chunks share a small tail of whole lines, so an answer that straddles a
    boundary still appears fully in at least one chunk;
  * every chunk gets a context header (company, period, section) that is prepended only for
    embedding (`embed_text`), so the stored `text` stays clean for citation display;
  * tables written by `html_to_text(tables="structured")` (between [TABLE] markers) are never
    mixed with prose: each table is its own chunk, or several chunks cut between rows, each
    starting with the table title line. Text without markers is chunked exactly as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_LABELS = {
    "business": "Item 1 Business",
    "risk_factors": "Item 1A Risk Factors",
    "mdna": "Item 7 Management's Discussion and Analysis",
}
_TABLE_RE = re.compile(r"\[TABLE\]\n(.*?)\n\[/TABLE\]", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Running page footers such as "Apple Inc. | 2025 Form 10-K | 7" carry no information.
_FOOTER_RE = re.compile(r"\b(19|20)\d\d\s+form\s+10-k\b|\bform\s+10-k\s*\|", re.IGNORECASE)
_FOOTER_MAX_CHARS = 100


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 1600
    overlap_chars: int = 200
    min_chars: int = 300  # a shorter trailing chunk is merged into the previous one
    small_table_chars: int = 0  # >0: a table shorter than this joins the chunk before it

    def __post_init__(self):
        if self.overlap_chars >= self.max_chars // 2:
            raise ValueError("overlap_chars must be smaller than half of max_chars")


DEFAULT_CONFIG = ChunkConfig()


def split_long_line(line: str, max_chars: int) -> list[str]:
    """Split one very long line at sentence ends, then hard-wrap whatever is still too long."""
    pieces, current = [], ""
    for sentence in _SENTENCE_RE.split(line):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(units: list[str], limit: int) -> list[str]:
    """Longest run of whole lines at the end of `units` that fits in `limit` characters."""
    tail, size = [], 0
    for unit in reversed(units):
        size += len(unit) + (1 if tail else 0)
        if size > limit:
            break
        tail.insert(0, unit)
    return tail


def chunk_table(block: str, cfg: ChunkConfig = DEFAULT_CONFIG) -> list[str]:
    """Cut one serialized table between rows; every piece repeats the title line."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    title = lines[0] if lines[0].startswith("Table") else None
    body = lines[1:] if title else lines
    pieces: list[list[str]] = []
    size = 0
    for line in body:
        for part in split_long_line(line, cfg.max_chars - len(title or "") - 1):
            if not pieces or size + 1 + len(part) > cfg.max_chars:
                pieces.append([])
                size = len(title or "")
            pieces[-1].append(part)
            size += 1 + len(part)
    return [
        "\n".join(([title] if title else []) + piece) for piece in pieces or [[]] if piece or title
    ]


def chunk_text(text: str, cfg: ChunkConfig = DEFAULT_CONFIG) -> list[str]:
    """Prose chunks and table chunks, in document order."""
    if "[TABLE]" not in text:
        return _chunk_prose(text, cfg)
    chunks, last = [], 0
    for match in _TABLE_RE.finditer(text):
        before = _chunk_prose(text[last : match.start()], cfg)
        chunks.extend(before)
        pieces = chunk_table(match.group(1), cfg)
        small = len(pieces) == 1 and len(pieces[0]) < cfg.small_table_chars
        # a small table next to its introducing paragraph is kept in the same chunk
        if small and chunks and before and len(chunks[-1]) + 1 + len(pieces[0]) <= cfg.max_chars:
            chunks[-1] += "\n" + pieces[0]
        else:
            chunks.extend(pieces)
        last = match.end()
    return chunks + _chunk_prose(text[last:], cfg)


def _chunk_prose(text: str, cfg: ChunkConfig) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or (len(line) < _FOOTER_MAX_CHARS and _FOOTER_RE.search(line)):
            continue
        units.extend(split_long_line(line, cfg.max_chars) if len(line) > cfg.max_chars else [line])

    chunks: list[tuple[list[str], int]] = []  # (lines, number of leading overlap lines)
    current: list[str] = []
    n_overlap = 0
    for unit in units:
        new_chars = len("\n".join(current[n_overlap:]))
        too_big = bool(current) and len("\n".join(current)) + 1 + len(unit) > cfg.max_chars
        # With little new content (e.g. a lone heading) keep going, even past max_chars,
        # instead of emitting a tiny chunk that is useless for retrieval.
        if too_big and new_chars >= cfg.min_chars:
            chunks.append((current, n_overlap))
            current = _tail(current, cfg.overlap_chars)
            if current and len("\n".join(current)) + 1 + len(unit) > cfg.max_chars:
                current = []
            n_overlap = len(current)
        current.append(unit)
    if len(current) > n_overlap:  # skip a final chunk made only of overlap
        chunks.append((current, n_overlap))

    if len(chunks) > 1 and len("\n".join(chunks[-1][0][chunks[-1][1] :])) < cfg.min_chars:
        last_lines, last_overlap = chunks.pop()
        previous_lines, previous_overlap = chunks.pop()
        chunks.append((previous_lines + last_lines[last_overlap:], previous_overlap))
    return ["\n".join(lines) for lines, _ in chunks]


def build_chunks(record: dict, cfg: ChunkConfig = DEFAULT_CONFIG) -> list[dict]:
    """Chunks for one parsed filing, with metadata and the context header for embedding."""
    rows = []
    for section, text in record["sections"].items():
        header = (
            f"{record['company']} | Form 10-K | period ended {record['report_date']} | "
            f"{SECTION_LABELS[section]}"
        )
        for position, chunk in enumerate(chunk_text(text, cfg)):
            non_space = [c for c in chunk if not c.isspace()]
            rows.append(
                {
                    "chunk_id": f"{record['ticker']}_{record['report_date']}_{section}_{position:04d}",
                    "ticker": record["ticker"],
                    "company": record["company"],
                    "report_date": record["report_date"],
                    "section": section,
                    "position": position,
                    "text": chunk,
                    "embed_text": f"{header}\n{chunk}",
                    "n_chars": len(chunk),
                    "digit_ratio": round(sum(c.isdigit() for c in non_space) / len(non_space), 3),
                }
            )
    return rows
