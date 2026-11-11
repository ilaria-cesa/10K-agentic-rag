"""Company universe of the corpus (US-listed 10-K filers): 19 companies, last 2 10-Ks each.

Mix of banks, insurers and non-financial companies so that questions can be about different
sectors and so that cross-company comparisons are possible. This list is the final corpus
(38 filings); `python -m src.data_pipeline` rebuilds it from EDGAR.
"""

DEFAULT_TICKERS = [
    # Banks and other financials
    "JPM",
    "BAC",
    "GS",
    "COF",
    "SCHW",
    "TFC",
    # Insurers
    "MET",
    "AIG",
    "PRU",
    "TRV",
    # Technology
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "AAPL",
    # Other sectors
    "PFE",
    "JNJ",
    "WMT",
]
