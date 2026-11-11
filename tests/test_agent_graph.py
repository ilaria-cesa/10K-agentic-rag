from src.agent.graph import AgentConfig, build_agent, regex_fallback, run_agent
from src.agent.plan import Search, parse_check, parse_plan

FILINGS = {"AAA": ["2024-12-31", "2025-12-31"], "BBB": ["2025-12-31"]}
NAMES = {"AAA": "Alpha Inc", "BBB": "Beta Corp"}


def chunk(cid, text="text"):
    return {
        "chunk_id": cid,
        "company": "X",
        "report_date": "2025-12-31",
        "section": "mdna",
        "text": text,
        "embed_text": text,
    }


class Script:
    """Fake LLM: returns the scripted replies in order and records the prompts."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, max_new_tokens):
        self.calls.append(messages)
        return self.replies.pop(0)


def make_search(log):
    def search(query, ticker, year, k):
        log.append((query, ticker, year, k))
        return [chunk(f"{ticker}_{year}-12-31_mdna_{i:04d}") for i in range(k)]

    return search


def test_parse_plan_accepts_noisy_json_and_rejects_garbage():
    text = 'Sure! {"searches": [{"query": "net income", "ticker": "aaa", "year": 2025}]} ok'
    assert parse_plan(text, {"AAA"}) == [Search("net income", "AAA", 2025)]
    assert parse_plan("no json here", {"AAA"}) is None
    assert parse_plan('{"searches": [{"query": ""}]}', {"AAA"}) is None
    unknown = parse_plan('{"searches": [{"query": "q", "ticker": "ZZZ", "year": "x"}]}', {"AAA"})
    assert unknown == [Search("q", None, None)]


def test_parse_check_only_returns_a_follow_up_when_not_enough():
    assert parse_check('{"enough": true}', {"AAA"}) is None
    follow = parse_check(
        '{"enough": false, "search": {"query": "dividends", "ticker": "AAA"}}', {"AAA"}
    )
    assert follow == Search("dividends", "AAA", None)
    assert parse_check("garbage", {"AAA"}) is None


def test_comparison_gets_one_search_per_year_and_answer_cites_real_chunks():
    log = []
    llm = Script(
        '{"searches": [{"query": "net income", "ticker": "AAA", "year": 2024},'
        ' {"query": "net income", "ticker": "AAA", "year": 2025}]}',
        '{"enough": true}',
        "It rose [AAA_2025-12-31_mdna_0000].",
    )
    agent = build_agent(llm, make_search(log), FILINGS, NAMES)
    out = run_agent(agent, "How did net income of Alpha change from 2024 to 2025?")
    assert [(t, y) for _, t, y, _ in log] == [("AAA", 2024), ("AAA", 2025), ("AAA", None)]
    assert log[-1][0] == "How did net income of Alpha change from 2024 to 2025?"
    assert out["cited_ids"] == ["AAA_2025-12-31_mdna_0000"] and not out["invalid_citations"]
    assert out["rounds"] == 1 and len(llm.calls) == 3


def test_follow_up_search_is_limited_by_max_rounds():
    log = []
    follow = '{"enough": false, "search": {"query": "capex", "ticker": "BBB", "year": 2025}}'
    llm = Script(
        '{"searches": [{"query": "revenue", "ticker": "BBB", "year": 2025}]}', follow, "NOT_FOUND"
    )
    agent = build_agent(llm, make_search(log), FILINGS, NAMES, AgentConfig(max_rounds=2))
    out = run_agent(agent, "q")
    assert out["rounds"] == 2 and [q for q, *_ in log][:2] == ["revenue", "q"]
    assert [q for q, *_ in log][-1] == "capex"
    assert out["refused"] is True
    # with max_rounds=1 the check step is skipped entirely
    llm2 = Script('{"searches": [{"query": "revenue"}]}', "answer")
    out2 = run_agent(
        build_agent(llm2, make_search([]), FILINGS, NAMES, AgentConfig(max_rounds=1)), "q"
    )
    assert out2["rounds"] == 1 and len(llm2.calls) == 2


def test_bad_plan_falls_back_to_a_single_search_on_the_question():
    log = []
    llm = Script("???", '{"enough": true}', "NOT_FOUND")
    agent = build_agent(llm, make_search(log), FILINGS, NAMES)
    run_agent(agent, "What is going on?")
    assert log[0][0] == "What is going on?"


def test_regex_fallback_uses_company_and_year_from_the_question():
    fallback = regex_fallback({"AAA": ["2025-12-31"]})
    assert fallback("What did Alpha earn?") == Search("What did Alpha earn?", None, None)
    assert fallback("What did aaa earn in 2025?").year == 2025


def test_select_context_keeps_one_passage_per_filing_and_orders_by_score():
    from src.agent.context import select_context

    def c(cid, ticker, date):
        return {"chunk_id": cid, "ticker": ticker, "report_date": date}

    evidence = [
        c("a1", "A", "2025"),
        c("a2", "A", "2025"),
        c("a3", "A", "2025"),
        c("b1", "A", "2024"),
    ]
    scores = {"a1": 0.9, "a2": 0.8, "a3": 0.7, "b1": 0.1}
    picked = select_context("q", evidence, lambda q, ch: [scores[x["chunk_id"]] for x in ch], 2)
    assert [x["chunk_id"] for x in picked] == ["a1", "b1"]  # the 2024 filing keeps its best
    assert select_context("q", evidence, None, 3) == evidence[:3]


def test_select_context_by_search_gives_every_search_its_own_passages():
    from src.agent.context import select_context_by_search

    def chunk(i):
        return {"chunk_id": i, "ticker": "T", "report_date": "2025-12-31"}

    evidence = [chunk(i) for i in ["a1", "a2", "b1", "b2", "c1", "x1", "x2"]]
    # for the original question the reranker prefers x1 and x2 over every per-company passage
    scores = {"x1": 9, "x2": 8, "a1": 3, "a2": 2, "b1": 3, "b2": 2, "c1": 1}
    score_fn = lambda q, ch: [scores[c["chunk_id"]] for c in ch]
    groups = [["a1", "a2"], ["b1", "b2"], ["c1"]]
    picked = [c["chunk_id"] for c in select_context_by_search("q", evidence, groups, score_fn, 5)]
    assert {"a1", "b1", "c1"} <= set(picked)  # the first passage of every search is in
    assert len(picked) == 5
    assert picked[0] == "x1"  # the best passage for the question still comes first
    assert select_context_by_search("q", evidence, [], None, 3) == evidence[:3]


def test_parse_check_list_reads_several_searches():
    from src.agent.plan import parse_check_list

    text = (
        "AAA: p1\\nBBB: MISSING\\n"
        '{"enough": false, "searches": [{"query": "BBB employees in 2025", "ticker": "BBB", '
        '"year": 2025}, {"query": "BBB employees in 2025", "ticker": "BBB", "year": 2025}]}'
    ).replace("\\\\n", "\\n")
    found = parse_check_list(text, {"AAA", "BBB"})
    assert [s.ticker for s in found] == ["BBB"]  # duplicates are dropped
    assert parse_check_list('{"enough": true}', {"AAA"}) == []
    assert parse_check_list("nothing here", {"AAA"}) == []


def test_stage_seconds_cover_every_stage_that_ran():
    llm = Script(
        '{"searches": [{"query": "revenue", "ticker": "BBB", "year": 2025}]}',
        '{"enough": true}',
        "NOT_FOUND",
    )
    agent = build_agent(llm, make_search([]), FILINGS, NAMES)
    out = run_agent(agent, "q")
    assert set(out["stage_seconds"]) == {"plan", "search", "check", "answer"}


def test_skip_check_single_saves_one_llm_call_for_one_company_and_year():
    llm = Script('{"searches": [{"query": "revenue", "ticker": "BBB", "year": 2025}]}', "NOT_FOUND")
    config = AgentConfig(skip_check_single=True)
    agent = build_agent(llm, make_search([]), FILINGS, NAMES, config)
    out = run_agent(agent, "q")
    assert len(llm.calls) == 2 and "check" not in out["stage_seconds"]


def test_skip_check_single_still_checks_multi_search_plans():
    llm = Script(
        '{"searches": [{"query": "a", "ticker": "AAA", "year": 2024},'
        ' {"query": "a", "ticker": "AAA", "year": 2025}]}',
        '{"enough": true}',
        "NOT_FOUND",
    )
    agent = build_agent(llm, make_search([]), FILINGS, NAMES, AgentConfig(skip_check_single=True))
    run_agent(agent, "q")
    assert len(llm.calls) == 3
