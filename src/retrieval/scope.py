"""Guess which filings a question is about (company and fiscal year) from its wording.

Rule-based on purpose: names come from the corpus itself plus a few short forms, so a question
that does not name the company ("the owner of ...") gets no company filter instead of a guess.
The fiscal-year label of every filing in this corpus equals the year of its period-end date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Short forms that people use in questions; the full names come from the chunks.
SHORT_NAMES: dict[str, list[str]] = {
    "AAPL": ["apple"],
    "AIG": ["aig"],
    "AMZN": ["amazon"],
    "BAC": ["bank of america"],
    "COF": ["capital one"],
    "GOOGL": ["alphabet"],
    "GS": ["goldman sachs", "goldman"],
    "JNJ": ["johnson & johnson", "j&j"],
    "JPM": ["jpmorgan", "jp morgan"],
    "MET": ["metlife"],
    "META": ["meta platforms", "meta"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "PFE": ["pfizer"],
    "PRU": ["prudential"],
    "SCHW": ["schwab"],
    "TFC": ["truist"],
    "TRV": ["travelers"],
    "WMT": ["walmart"],
}
_YEAR = re.compile(r"(?<!\d)(20\d\d)(?!\d)")


@dataclass
class Scope:
    tickers: list[str] = field(default_factory=list)  # empty = company not recognised
    report_dates: list[tuple[str, str]] = field(default_factory=list)  # (ticker, date) allowed
    years_found: list[int] = field(default_factory=list)

    @property
    def filtered(self) -> bool:
        return bool(self.report_dates)


def filings_of(chunks: list[dict]) -> dict[str, list[str]]:
    """ticker -> sorted report dates present in the corpus."""
    out: dict[str, set[str]] = {}
    for chunk in chunks:
        out.setdefault(chunk["ticker"], set()).add(chunk["report_date"])
    return {t: sorted(d) for t, d in out.items()}


def detect_scope(question: str, filings: dict[str, list[str]]) -> Scope:
    text = question.lower()
    tickers = [
        ticker
        for ticker, names in SHORT_NAMES.items()
        if ticker in filings
        and any(re.search(rf"(?<![\w&]){re.escape(n)}(?![\w&])", text) for n in names)
    ]
    years = sorted({int(y) for y in _YEAR.findall(question)})
    if tickers:
        allowed = []
        for ticker in tickers:
            picked = [d for d in filings[ticker] if int(d[:4]) in years]
            allowed += [(ticker, d) for d in picked or filings[ticker]]
    else:  # company not named: only a year can narrow the search, otherwise no filter at all
        allowed = [(t, d) for t, dates in filings.items() for d in dates if int(d[:4]) in years]
    return Scope(tickers=tickers, report_dates=allowed, years_found=years)
