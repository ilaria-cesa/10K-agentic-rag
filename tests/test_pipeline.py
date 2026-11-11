import numpy as np

from src.retrieval.pipeline import Corpus, RetrievalConfig, pool_scores, retrieve
from src.retrieval.scope import detect_scope, filings_of


def make_chunks():
    rows = []
    for ticker, company in [("AAPL", "Apple Inc."), ("MSFT", "MICROSOFT CORP")]:
        for date in ["2025-01-01", "2026-01-01"]:
            for pos in range(3):
                rows.append(
                    {
                        "chunk_id": f"{ticker}_{date}_mdna_{pos:04d}",
                        "ticker": ticker,
                        "report_date": date,
                        "company": company,
                        "embed_text": f"{company} {date} {pos}",
                    }
                )
    return rows


def make_corpus():
    chunks = make_chunks()
    rng = np.random.default_rng(0)
    return Corpus(chunks, rng.normal(size=(len(chunks), 8)))


def test_scope_detects_company_and_years_and_ignores_unknown_years():
    filings = filings_of(make_chunks())
    scope = detect_scope("How did Apple's revenue change between 2025 and 2026?", filings)
    assert scope.tickers == ["AAPL"] and len(scope.report_dates) == 2
    one = detect_scope("What did Microsoft report in fiscal 2026?", filings)
    assert one.report_dates == [("MSFT", "2026-01-01")]
    unknown_year = detect_scope("What did Microsoft report in 2019?", filings)
    assert len(unknown_year.report_dates) == 2  # no such year: keep both filings
    anon = detect_scope("How many employees does the company have?", filings)
    assert not anon.filtered


def test_filter_keeps_only_the_filings_in_scope():
    corpus = make_corpus()
    config = RetrievalConfig("f", use_filter=True, top_k=6)
    ids, scope = retrieve(corpus, "Apple 2025 report", corpus.vectors[0], config)
    assert scope is not None
    assert {corpus.chunks[i]["ticker"] for i in ids} == {"AAPL"}
    assert {corpus.chunks[i]["report_date"] for i in ids} == {"2025-01-01"}


def test_per_filing_covers_both_years_in_the_top_results():
    corpus = make_corpus()
    config = RetrievalConfig("p", use_filter=True, per_filing=True, pool=6, top_k=2)
    ids, _ = retrieve(corpus, "Apple change from 2025 to 2026", corpus.vectors[0], config)
    assert {corpus.chunks[i]["report_date"] for i in ids} == {"2025-01-01", "2026-01-01"}


def test_rerank_uses_the_scorer_order():
    corpus = make_corpus()

    class Prefer:
        def score(self, question, passages):
            return [1.0 if p.endswith(" 2") else 0.0 for p in passages]

    config = RetrievalConfig("r", rerank=True, pool=12, top_k=3)
    ids, _ = retrieve(corpus, "anything", corpus.vectors[0], config, Prefer())
    assert all(corpus.chunks[i]["chunk_id"].endswith("0002") for i in ids)


def test_hybrid_adds_chunks_found_by_keywords():
    chunks = make_chunks()
    chunks[2]["embed_text"] = "Adjusted earnings table row Other revenues"
    rng = np.random.default_rng(0)
    corpus = Corpus(chunks, rng.normal(size=(len(chunks), 8)))
    far = -corpus.vectors[2]  # a query vector pointing away from chunk 2
    plain = RetrievalConfig("d", use_filter=True, pool=2, top_k=10)
    hybrid = RetrievalConfig("h", use_filter=True, pool=2, top_k=10, hybrid=True, bm25_pool=2)
    question = "Apple 2025 other revenues adjusted earnings"
    ids_plain, _ = retrieve(corpus, question, far, plain)
    ids_hybrid, _ = retrieve(corpus, question, far, hybrid)
    assert 2 not in ids_plain
    assert 2 in ids_hybrid


def test_pool_scores_lists_every_candidate_with_its_scores():
    chunks = make_chunks()
    chunks[2]["embed_text"] = "Adjusted earnings table row Other revenues"
    rng = np.random.default_rng(0)
    corpus = Corpus(chunks, rng.normal(size=(len(chunks), 8)))

    class Fixed:
        def score(self, question, passages):
            return [float(len(p)) for p in passages]

    config = RetrievalConfig("h", use_filter=True, pool=2, hybrid=True, bm25_pool=2)
    rows = pool_scores(corpus, "Apple 2025 other revenues", -corpus.vectors[2], config, Fixed())
    by_id = {r["chunk_id"]: r for r in rows}
    keyword_only = by_id[chunks[2]["chunk_id"]]
    assert keyword_only["dense_rank"] is None and keyword_only["bm25_rank"] == 0
    assert all(r["rerank_score"] > 0 for r in rows)
