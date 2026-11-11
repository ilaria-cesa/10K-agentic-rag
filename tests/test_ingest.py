import pytest

from src.ingest.edgar import EdgarClient, Filing
from src.ingest.parse import MIN_SECTION_CHARS, extract_sections, html_to_text

UA = "Test User test@example.com"


class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None, text=""):
        self._payload, self.status_code, self.headers, self.text = (
            payload,
            status,
            headers or {},
            text,
        )

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.responses.pop(0)


def make_client(responses, **kwargs):
    sleeps = []
    client = EdgarClient(
        user_agent=UA, session=FakeSession(responses), sleep=sleeps.append, **kwargs
    )
    return client, sleeps


def test_user_agent_must_contain_email(monkeypatch):
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ValueError):
        EdgarClient(user_agent="no email here")


def test_rate_limit_is_capped_at_ten_per_second():
    with pytest.raises(ValueError):
        EdgarClient(user_agent=UA, max_rps=20)


def test_requests_are_spaced_by_the_rate_limit():
    ok = FakeResponse({"a": 1})
    client, sleeps = make_client([ok, ok], max_rps=5)
    client._clock = lambda: 0.0  # time does not advance between requests
    client.get_json("u1")
    client.get_json("u2")
    assert sleeps == [pytest.approx(0.2)]


def test_retries_on_429_using_retry_after():
    client, sleeps = make_client(
        [FakeResponse(status=429, headers={"Retry-After": "3"}), FakeResponse({"ok": True})]
    )
    assert client.get_json("u") == {"ok": True}
    assert 3.0 in sleeps


def test_gives_up_after_max_retries():
    client, _ = make_client([FakeResponse(status=503)] * 3, max_retries=2)
    with pytest.raises(RuntimeError):
        client.get_json("u")


def test_resolve_tickers_ignores_unknown():
    data = {"0": {"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"}}
    client, _ = make_client([FakeResponse(data)])
    assert client.resolve_tickers(["jpm", "NOPE"]) == {"jpm": (19617, "JPMORGAN CHASE & CO")}


def test_list_10k_filings_skips_amendments_and_other_forms():
    recent = {
        "form": ["8-K", "10-K", "10-K/A", "10-K", "10-Q"],
        "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
        "filingDate": ["2026-03-01", "2026-02-15", "2026-03-10", "2025-02-14", "2026-05-01"],
        "reportDate": ["", "2025-12-31", "2025-12-31", "2024-12-31", ""],
        "primaryDocument": ["x.htm", "k26.htm", "ka.htm", "k25.htm", "q.htm"],
    }
    client, _ = make_client([FakeResponse({"filings": {"recent": recent}})])
    filings = client.list_10k_filings(19617, n=2)
    assert [f.accession for f in filings] == ["a2", "a4"]
    assert filings[0].url == "https://www.sec.gov/Archives/edgar/data/19617/a2/k26.htm"


def test_filing_url_removes_dashes_from_accession():
    f = Filing(1, "0000001-25-000002", "10-K", "2025-02-01", "2024-12-31", "d.htm")
    assert "/000000125000002/d.htm" in f.url


BODY = "Lorem ipsum dolor sit amet. " * 200  # ~5,600 chars


def synthetic_filing() -> str:
    return f"""
    <html><body>
    <ix:header><div>hidden xbrl metadata</div></ix:header>
    <div>Table of Contents</div>
    <div>Item 1. Business</div><div>Item 1A. Risk Factors</div><div>Item 7. MD&amp;A</div>
    <div>Item 1. Business</div><p>{BODY} Business text.</p>
    <div>Item 1A. Risk Factors</div>
    <p>Risks described here. See <a href="#x">Item 7. MD&amp;A</a> for details. {BODY}</p>
    <div>Item 1B. Unresolved Staff Comments</div><p>None.</p>
    <div>Item 7. Management's Discussion</div><p>{BODY} Results text.</p>
    <div>Item 8. Financial Statements</div><p>Numbers.</p>
    </body></html>"""


def test_html_to_text_drops_hidden_xbrl_header_and_keeps_inline_links():
    text = html_to_text(synthetic_filing())
    assert "hidden xbrl metadata" not in text
    assert "See Item 7. MD&A for details." in text


def test_extract_sections_picks_real_sections_not_table_of_contents():
    sections, warnings = extract_sections(html_to_text(synthetic_filing()))
    assert warnings == []
    assert "Business text." in sections["business"]
    assert "Risks described here" in sections["risk_factors"]
    assert "for details" in sections["risk_factors"]  # inline cross-reference does not cut it
    assert "Results text." in sections["mdna"]
    assert all(len(s) >= MIN_SECTION_CHARS for s in sections.values())
    assert "Unresolved Staff" not in sections["risk_factors"]


def test_extract_sections_warns_when_a_section_is_missing():
    _, warnings = extract_sections("Item 1. Business\n" + BODY + "\nItem 2. Properties\nx")
    assert "missing section: risk_factors" in warnings
    assert "missing section: mdna" in warnings


def test_mdna_fallback_when_item_7_is_a_cross_reference():
    text = (
        "Item 1. Business\n" + BODY + "\nItem 1A. Risk Factors\n" + BODY + "\n"
        "Item 7. Management's Discussion and Analysis\nSee pages 46-160.\n"
        "Item 8. Financial Statements\nSee pages 165-314.\n"
        "Management's discussion and analysis\n"  # real section title in the financial part
         + BODY + "\nManagement's report on internal control over financial reporting\nend\n"
    )
    sections, warnings = extract_sections(text)
    assert len(sections["mdna"]) >= MIN_SECTION_CHARS
    assert "report on internal control" not in sections["mdna"]
    assert any("fallback" in w for w in warnings)


def test_list_10k_filings_reads_archive_files_when_recent_is_not_enough():
    recent = {
        "form": ["424B2", "10-K"],
        "accessionNumber": ["x1", "a2"],
        "filingDate": ["2026-06-01", "2026-02-15"],
        "reportDate": ["", "2025-12-31"],
        "primaryDocument": ["x.htm", "k26.htm"],
    }
    older = {
        "form": ["10-K", "10-Q"],
        "accessionNumber": ["a1", "q1"],
        "filingDate": ["2025-02-14", "2025-05-01"],
        "reportDate": ["2024-12-31", ""],
        "primaryDocument": ["k25.htm", "q.htm"],
    }
    top = {
        "filings": {
            "recent": recent,
            "files": [{"name": "CIK1-submissions-001.json", "filingTo": "2025-12-31"}],
        }
    }
    client, _ = make_client([FakeResponse(top), FakeResponse(older)])
    filings = client.list_10k_filings(1, n=2)
    assert [f.accession for f in filings] == ["a2", "a1"]
    assert client.session.calls[1].endswith("/CIK1-submissions-001.json")


def test_archive_files_are_not_fetched_when_recent_has_enough():
    recent = {
        "form": ["10-K", "10-K"],
        "accessionNumber": ["a2", "a1"],
        "filingDate": ["2026-02-15", "2025-02-14"],
        "reportDate": ["2025-12-31", "2024-12-31"],
        "primaryDocument": ["k26.htm", "k25.htm"],
    }
    top = {"filings": {"recent": recent, "files": [{"name": "old.json", "filingTo": "2020-01-01"}]}}
    client, _ = make_client([FakeResponse(top)])
    assert len(client.list_10k_filings(1, n=2)) == 2
    assert len(client.session.calls) == 1


def test_summary_flags_incomplete_filings(tmp_path):
    import json

    from src.ingest.summary import build_rows, health_report

    parsed = tmp_path / "parsed"
    parsed.mkdir()
    full = {n: "x" * MIN_SECTION_CHARS for n in ("business", "risk_factors", "mdna")}
    for ticker, sections in (("AAA", full), ("BBB", {"business": "short"})):
        record = {
            "ticker": ticker,
            "report_date": "2025-12-31",
            "sections": sections,
            "warnings": [],
        }
        (parsed / f"{ticker}.json").write_text(json.dumps(record), encoding="utf-8")
    rows = build_rows(parsed)
    assert [r["ok"] for r in rows] == [True, False]
    report = "\n".join(health_report(rows))
    assert "['BBB']" in report


def test_clean_section_removes_page_furniture_but_keeps_sentences():
    from src.ingest.parse import _clean_section

    raw = (
        "ITEM 1A. RISK FACTORS\nWe face risks. See Item 1A. Risk Factors on page 8 for details.\n"
        "34\nPART II\nItem 7\nEconomic conditions matter.\n"
    )
    cleaned = _clean_section(raw)
    assert cleaned == (
        "We face risks. See Item 1A. Risk Factors on page 8 for details.\n"
        "Economic conditions matter."
    )


TABLE_HTML = (
    "<p>Net sales by category</p><p>(in millions)</p>"
    "<table><tr><td></td><td>2026</td><td>2025</td></tr>"
    "<tr><td>iPhone</td><td>$</td><td>254,098</td><td>$</td><td>209,586</td></tr>"
    "<tr><td>Mac</td><td>$</td><td>30,000</td><td>$</td><td>29,000</td></tr></table>"
)


def test_structured_tables_keep_row_and_column():
    text = html_to_text(f"<html><body>{TABLE_HTML}</body></html>", tables="structured")
    assert "[TABLE]" in text and "[/TABLE]" in text
    assert "iPhone: 2026 = $254,098; 2025 = $209,586" in text
    assert "Net sales by category" in text and "in millions" in text


def test_flat_is_default_and_has_no_markers():
    text = html_to_text(f"<html><body>{TABLE_HTML}</body></html>")
    assert "[TABLE]" not in text and "iPhone" in text


def test_unknown_tables_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        html_to_text("<p>x</p>", tables="other")
