"""Turn a 10-K HTML document into plain text and extract the sections we index.

Limitations (documented on purpose):
  * tables are flattened to text, so numeric tables lose their layout;
  * section detection relies on "Item X." headings, so unusual filings may need
    manual checks. `extract_sections` returns warnings instead of failing silently.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

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


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
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
