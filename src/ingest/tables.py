"""Turn an HTML table of a 10-K into text where every value carries its row and its column.

Why: a flattened table is a row of loose numbers ("iPhone $ 254,098 21 % ..."). A reader, human
or model, cannot tell which number is which year. Here each row becomes one line:

    iPhone: 2026 = $254,098; Change 2026 vs 2025 = 21%; 2025 = $209,586; ...

Steps (no library beyond BeautifulSoup, so every step can be tested):
  1. read the cells of every row and drop the empty ones (10-K tables are full of them);
  2. glue the pieces of one value back together ("$" + "254,098", "21" + "%", "(4" + ")");
  3. rows that hold numbers are data rows; the rows above them are headers;
  4. the headers are matched to the values by position; if a row does not match, it is written
     without headers instead of guessing;
  5. a header such as "Change" is ambiguous when it appears twice, so the years on its sides are
     added ("Change 2026 vs 2025").

Tables without any data row (layout boxes, text in a table) return None and stay ordinary text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_YEAR = re.compile(r"^(19|20)\d\d$")
_VALUE = re.compile(r"^[(\-–—]?\$?\(?\d[\d,]*(\.\d+)?\)?\s?(%|x|pts|bps)?\)?$|^[—–-]+\s?%?$")
_PREFIX = {"$", "(", "($", "$(", "US$", "£", "€"}
_SUFFIX = {"%", ")", "%)", ")%", "pts", "bps", "x"}
UNIT = re.compile(r"in\s+(millions|billions|thousands)(?:[^)]{0,40})?", re.IGNORECASE)
_VAGUE_HEADER = re.compile(r"^(%\s*)?(change|variance|chg\.?)$", re.IGNORECASE)


@dataclass
class Table:
    header: list[str]  # one name per value column ([] when no usable header was found)
    rows: list[tuple[str, list[str]]]  # (row label, values); values == [] for a heading row
    context: list[str] = field(default_factory=list)  # e.g. "Year ended December 31,"
    unit: str | None = None  # e.g. "in millions"


def _is_year(cell: str) -> bool:
    return bool(_YEAR.match(cell))


def _is_number(cell: str) -> bool:
    """A value cell. Plain years do not count: they are column headers."""
    return bool(_VALUE.match(cell)) and not _is_year(cell)


def merge_fragments(cells: list[str]) -> list[str]:
    """Glue the pieces of one value back together and drop empty cells."""
    merged: list[str] = []
    prefix = ""
    for cell in cells:
        cell = cell.replace("\xa0", " ").strip()
        if not cell:
            continue
        if cell in _PREFIX:
            prefix += cell
        elif cell in _SUFFIX and merged:
            merged[-1] += cell
        else:
            merged.append(prefix + cell)
            prefix = ""
    return merged


def _split_row(cells: list[str]) -> tuple[str, list[str]]:
    """(label, values). A row that starts with a number has no label (a year can be a label)."""
    if cells and not _is_number(cells[0]):
        return cells[0], cells[1:]
    return "", cells


def _spread(group_titles: list[str], n_cols: int) -> list[str]:
    """["Credit exposure", "Nonperforming"] over 4 columns -> each title repeated twice."""
    return [title for title in group_titles for _ in range(n_cols // len(group_titles))]


def _name_vague_headers(header: list[str]) -> list[str]:
    """ "Change" next to two years becomes "Change 2026 vs 2025"."""
    named = list(header)
    for i, name in enumerate(header):
        if not _VAGUE_HEADER.match(name):
            continue
        left = next((header[j] for j in range(i - 1, -1, -1) if _is_year(header[j])), None)
        right = next((header[j] for j in range(i + 1, len(header)) if _is_year(header[j])), None)
        if left and right:
            named[i] = f"{name} {left} vs {right}"
        elif left:
            named[i] = f"{name} {left}"
    return named


def parse_table(table) -> Table | None:
    """`table` is a BeautifulSoup <table> tag. Returns None when it has no numeric rows."""
    rows = []
    for tr in table.find_all("tr"):
        cells = merge_fragments(
            [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"], recursive=False)]
        )
        if cells:
            rows.append(cells)

    data_at = [i for i, cells in enumerate(rows) if any(_is_number(c) for c in cells[1:] or cells)]
    if not data_at:
        return None
    first = data_at[0]
    split = {i: _split_row(rows[i]) for i in data_at}
    n_cols = Counter(len(split[i][1]) for i in data_at).most_common(1)[0][0]

    context: list[str] = []
    unit = None
    columns: list[list[str]] = []
    for cells in rows[:first]:
        text = " ".join(cells)
        match = UNIT.search(text)
        if match and unit is None:
            unit = match.group(0).strip().rstrip(",;")
        if len(cells) == n_cols:
            columns.append(cells)
        elif len(cells) == n_cols + 1 and not _is_year(cells[0]):
            columns.append(cells[1:])  # first cell names the label column, e.g. "(in millions)"
        elif 1 < len(cells) < n_cols and n_cols % len(cells) == 0:
            columns.append(_spread(cells, n_cols))  # group titles over several columns
        elif len(cells) > 2 and n_cols % (len(cells) - 1) == 0 and not _is_year(cells[0]):
            columns.append(_spread(cells[1:], n_cols))  # same, after a caption cell
        elif not match:
            context.append(text)
    header = [" ".join(parts) for parts in zip(*columns, strict=True)] if columns else []

    out_rows: list[tuple[str, list[str]]] = []
    for i, cells in enumerate(rows[first:], start=first):
        if i in split:
            out_rows.append(split[i])
        elif len(cells) == 1:
            out_rows.append((cells[0], []))  # a heading row such as "Assets"
        else:
            out_rows.append((" | ".join(cells), []))
    return Table(_name_vague_headers(header), out_rows, context, unit)


_DELTA_HEADER = re.compile(r"^(%\s*)?(change|variance|chg\.?)\b", re.IGNORECASE)


def _base_columns(header: list[str]) -> list[str]:
    """The header without its "Change ..." columns: the value columns themselves."""
    return [h for h in header if not _DELTA_HEADER.match(h)]


def serialize(table: Table, title: str | None = None, *, align_to_base: bool = False) -> str:
    """One line per row; every value is written as "column = value".

    With `align_to_base`, a row that has fewer values than the header (typically "Percentage of
    net sales: 8% | 8% | 8%" under a header that also has "Change" columns) is matched to the
    value columns when their number is exactly the number of values."""
    head = f"Table: {title}" if title else "Table"
    if table.unit:
        head += f" ({table.unit})"
    lines = [head + ":", *table.context]
    base = _base_columns(table.header) if align_to_base else []
    for label, values in table.rows:
        columns = None
        if table.header and len(values) == len(table.header):
            columns = table.header
        elif base and values and len(values) == len(base):
            columns = base
        if not values:
            lines.append(label)
        elif columns:
            pairs = "; ".join(f"{h} = {v}" for h, v in zip(columns, values, strict=True))
            lines.append(f"{label or '(no label)'}: {pairs}")
        else:  # not aligned with the header: keep the values, do not guess the columns
            lines.append(f"{label or '(no label)'}: " + " | ".join(values))
    return "\n".join(lines)


def mostly_percent(table: Table, share: float = 0.8) -> bool:
    """True when most values are percentages (a money unit such as "in millions" is then wrong)."""
    values = [v for _, vals in table.rows for v in vals]
    return bool(values) and sum("%" in v for v in values) / len(values) >= share


def inherit_header(table: Table, previous: Table) -> bool:
    """Give a table without header the header of the table right above it, when every data row has
    exactly as many values as that header has columns. Returns True when it was applied."""
    if table.header or not previous.header:
        return False
    counts = {len(values) for _, values in table.rows if values}
    if counts != {len(previous.header)}:
        return False
    table.header = list(previous.header)
    return True
