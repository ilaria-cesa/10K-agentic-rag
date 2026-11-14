"""Turn a 10-K HTML document into plain text and extract the sections we index.

Limitations (documented on purpose):
  * tables are flattened to text, so numeric tables lose their layout;
  * section detection relies on "Item X." headings, so unusual filings may need
    manual checks. `extract_sections` returns warnings instead of failing silently.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

from src.ingest.tables import UNIT, inherit_header, mostly_percent, parse_table, serialize

# 10-Ks are served as XHTML; parsing them with the HTML parser works fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Items we keep. Item 1 = business, 1A = risk factors, 7 = MD&A.
WANTED_ITEMS = {"1": "business", "1a": "risk_factors", "7": "mdna"}
MIN_SECTION_CHARS = 2000

_BLOCK_TAGS = ["p", "div", "br", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
_HEADING_RE = re.compile(
    r"^[ \t]*item[ \t]+(\d{1,2}[ab]?)[ \t]*[\.:\-–—]?[ \t]*(.{0,100})$",
    re.IGNORECASE | re.MULTILINE,
)


# Some filers (e.g. JPMorgan) put only a cross-reference under "Item 7" and place the real
# MD&A in a separate financial section. The fallback looks for the MD&A title line and stops
# at the first end marker that follows it.
_MDNA_TITLE_RE = re.compile(
    r"^management.s discussion and analysis[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_MDNA_END_RE = re.compile(
    r"^(management.s report on internal control|report of independent registered public accounting"
    r" firm|consolidated financial statements[ \t]*$)",
    re.IGNORECASE | re.MULTILINE,
)


TABLE_START, TABLE_END = "[TABLE]", "[/TABLE]"
_NOT_A_TITLE = re.compile(r"^(table of contents|\d{1,3}|part [ivx]+)$", re.IGNORECASE)


def _caption(table) -> tuple[str | None, str | None]:
    """(title, unit) from the few text lines just above a table, e.g. "Net sales by category"
    and "(in millions)". Page furniture such as "Table of Contents" or a page number is skipped."""
    lines = [s.strip() for s in table.find_all_previous(string=True, limit=12) if s.strip()]
    unit = next((m.group(0).strip() for line in lines[:4] if (m := UNIT.search(line))), None)
    title = next(
        (
            line
            for line in lines[:4]
            if 3 <= len(line) <= 160 and not _NOT_A_TITLE.match(line) and not UNIT.search(line)
        ),
        None,
    )
    return title, unit


_OWN_TITLE = re.compile(r"^table\s+\d+[a-z]?\b", re.IGNORECASE)
_COMPANY_NAME = re.compile(
    r"(\b(inc|corp|corporation|company|co|ltd|plc|llc)\.?|&.*)$", re.IGNORECASE
)


def _good_title(title: str | None) -> str | None:
    """Drop captions that are not titles: a page header (company name), a long sentence."""
    if not title or len(title) > 110 or title.endswith((".", ":")) and len(title) > 60:
        return None
    return None if _COMPANY_NAME.search(title) else title


def _adjacent(tag, previous_block) -> bool:
    """True when nothing but blanks lies between the previous table's text and this table."""
    if previous_block is None:
        return False
    for node in tag.previous_elements:
        if node is previous_block:
            return True
        if isinstance(node, NavigableString) and node.strip():
            return False
    return False


def _structure_tables(soup, v2: bool = False) -> None:
    """Replace every table that holds numbers by text in which each value carries its row and
    column (see tables.py). The block is wrapped in markers so the chunker can keep it together.

    With `v2`: the table's own first line ("Table 8 Average Balances ...") is the title when
    present; page headers and sentences are not titles; a percentage table gets no money unit;
    a table split in two (header only in the first part) takes the header of the part above; rows
    with only the value columns are aligned to them."""
    previous: tuple | None = None  # (parsed table, its text block)
    for tag in soup.find_all("table"):
        if tag.find("table"):  # a table of tables: the inner ones are handled on their own
            continue
        parsed = parse_table(tag)
        # a lone number with no label (a page number in a table cell) is not a table
        if parsed is None or not any(label and values for label, values in parsed.rows):
            continue
        title, unit = _caption(tag)
        if v2:
            own = next((c for c in parsed.context if _OWN_TITLE.match(c)), None)
            if own:
                parsed.context.remove(own)
                title = own
            else:
                title = _good_title(title)
            if previous and _adjacent(tag, previous[1]):
                inherit_header(parsed, previous[0])
        parsed.unit = parsed.unit or unit
        if v2 and mostly_percent(parsed):
            parsed.unit = None
        block = NavigableString(
            f"\n{TABLE_START}\n{serialize(parsed, title, align_to_base=v2)}\n{TABLE_END}\n"
        )
        tag.replace_with(block)
        previous = (parsed, block)


def html_to_text(html: str, tables: str = "flat") -> str:
    """Plain text of a filing. `tables`: "flat" (cells joined by spaces), "structured", or
    "structured2" (structured with title, unit and alignment fixes)."""
    if tables not in {"flat", "structured", "structured2"}:
        raise ValueError("tables must be 'flat', 'structured' or 'structured2'")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    if tables != "flat":
        _structure_tables(soup, v2=tables == "structured2")
    # Newlines only at block boundaries, so inline links stay inside their sentence.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in soup.find_all(["td", "th"]):
        tag.insert_after(" ")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_PAGE_NOISE_RE = re.compile(r"^[ \t]*(part[ \t]+[ivx]+|\d{1,3})[ \t]*$", re.IGNORECASE)


def _clean_section(segment: str) -> str:
    """Drop page furniture: running "PART II" / "Item 7" headers, bare page numbers and
    leftover "Item X." heading lines. Without this, every page break injects noise into chunks."""
    kept = [
        line
        for line in segment.splitlines()
        if not _PAGE_NOISE_RE.match(line) and not _HEADING_RE.fullmatch(line.strip())
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Return ({section_name: text}, warnings).

    Every "Item X" heading starts a candidate segment that ends at the next heading
    of a different item. The table of contents produces tiny candidates, so for each
    wanted item we keep the longest candidate.
    """
    headings = [(m.start(), m.end(), m.group(1).lower()) for m in _HEADING_RE.finditer(text)]
    best: dict[str, str] = {}
    for i, (_, end, key) in enumerate(headings):
        if key not in WANTED_ITEMS:
            continue
        stop = len(text)
        for next_start, _, next_key in headings[i + 1 :]:
            if next_key != key:
                stop = next_start
                break
        segment = _clean_section(text[end:stop])
        name = WANTED_ITEMS[key]
        if len(segment) > len(best.get(name, "")):
            best[name] = segment

    warnings = []
    if len(best.get("mdna", "")) < MIN_SECTION_CHARS:
        fallback = _mdna_fallback(text)
        if len(fallback) >= MIN_SECTION_CHARS:
            best["mdna"] = fallback
            warnings.append("mdna extracted with fallback (Item 7 is a cross-reference)")
    for name in WANTED_ITEMS.values():
        length = len(best.get(name, ""))
        if length == 0:
            warnings.append(f"missing section: {name}")
        elif length < MIN_SECTION_CHARS:
            warnings.append(f"short section: {name} ({length} chars)")
    return best, warnings


def _mdna_fallback(text: str) -> str:
    """Longest span from an MD&A title line to the next end marker ("" if none found).

    The table of contents and page running headers also match the title, but they give
    shorter spans, so taking the longest one selects the real section.
    """
    best = ""
    for title in _MDNA_TITLE_RE.finditer(text):
        end = _MDNA_END_RE.search(text, title.end())
        segment = _clean_section(text[title.end() : end.start() if end else len(text)])
        if end and len(segment) > len(best):
            best = segment
    return best
