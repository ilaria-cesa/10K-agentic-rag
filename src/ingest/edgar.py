"""Minimal SEC EDGAR client that follows the SEC fair-access policy.

Rules enforced here (https://www.sec.gov/os/accessing-edgar-data):
  * every request declares a User-Agent with a name and a contact email;
  * at most 10 requests per second (we default to 5);
  * retry with backoff on 429 / 5xx responses.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str  # e.g. "0000019617-25-000123"
    form: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return DOCUMENT_URL.format(
            cik=self.cik,
            accession_nodash=self.accession.replace("-", ""),
            document=self.primary_document,
        )


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None = None,
        max_rps: float = 5.0,
        max_retries: int = 4,
        session: requests.Session | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        user_agent = user_agent or os.environ.get("EDGAR_USER_AGENT", "")
        if "@" not in user_agent:
            raise ValueError(
                "EDGAR requires a User-Agent with a name and a contact email. "
                "Set EDGAR_USER_AGENT, e.g. 'Jane Doe jane@example.com'."
            )
        if not 0 < max_rps <= 10:
            raise ValueError("max_rps must be in (0, 10] (SEC limit is 10 requests/second)")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / max_rps
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _get(self, url: str) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            response = self.session.get(url, timeout=60)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2.0**attempt
                self._sleep(delay)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_json(self, url: str) -> dict:
        return self._get(url).json()

    def get_text(self, url: str) -> str:
        return self._get(url).text

    def resolve_tickers(self, tickers: list[str]) -> dict[str, tuple[int, str]]:
        """Map ticker -> (CIK, company name). Unknown tickers are simply absent."""
        data = self.get_json(TICKERS_URL)
        by_ticker = {row["ticker"].upper(): row for row in data.values()}
        return {
            t: (int(by_ticker[t.upper()]["cik_str"]), by_ticker[t.upper()]["title"])
            for t in tickers
            if t.upper() in by_ticker
        }

    def list_10k_filings(self, cik: int, n: int = 2) -> list[Filing]:
        """Latest `n` original 10-K filings (amendments such as 10-K/A are skipped).

        The "recent" block of the submissions API only covers the latest filings. Companies that
        file thousands of documents (large banks) can push last year's 10-K out of it, so older
        archive files are read too until `n` filings are found.
        """
        data = self.get_json(SUBMISSIONS_URL.format(cik=cik))["filings"]
        filings = _parse_10k_rows(cik, data["recent"])
        archives = sorted(data.get("files", []), key=lambda e: e.get("filingTo", ""), reverse=True)
        for archive in archives:
            if len(filings) >= n:
                break
            older = self.get_json(SUBMISSIONS_FILE_URL.format(name=archive["name"]))
            filings += _parse_10k_rows(cik, older)
        filings.sort(key=lambda f: f.filing_date, reverse=True)
        return filings[:n]


def _parse_10k_rows(cik: int, block: dict) -> list[Filing]:
    rows = zip(
        block["form"],
        block["accessionNumber"],
        block["filingDate"],
        block["reportDate"],
        block["primaryDocument"],
        strict=True,
    )
    return [
        Filing(cik, acc, form, filed, report, doc)
        for form, acc, filed, report, doc in rows
        if form == "10-K"
    ]
