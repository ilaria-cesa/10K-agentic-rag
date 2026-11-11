"""LangGraph agent: plan -> search -> (check -> search again) -> answer.

Compared with the single-shot baseline the agent can (1) understand who and which years the
question is about even when the company is not named, (2) run one search per company and
year, so a comparison sees both years, and (3) look once more for a missing fact.
It is deliberately small: at most `max_rounds` search rounds and three LLM calls. Every round
also searches with the original question, and the final passages are chosen by reranking all
candidates against that question (see context.py).
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, TypedDict

from src.agent.context import ScoreFn, select_context_by_search
from src.agent.parsing import parse_answer
from src.agent.plan import (
    Search,
    check_messages,
    parse_check_list,
    parse_plan,
    plan_messages,
)
from src.agent.prompts import build_messages

LLM = Callable[[list[dict], int], str]  # (messages, max_new_tokens) -> text
SearchFn = Callable[[str, str | None, int | None, int], list[dict]]  # query, ticker, year, k


@dataclass(frozen=True)
class AgentConfig:
    max_rounds: int = 2  # search rounds: 1 = plan only, 2 = plus one follow-up
    chunks_per_search: int = 4  # candidates kept from each search
    single_search_chunks: int = 6  # candidates when a round has only one search
    max_context_chunks: int = 8  # passages given to the answering model
    per_search_chunks: int = 2  # passages each planned search is guaranteed in the context
    plan_tokens: int = 200
    check_tokens: int = 300
    answer_tokens: int = 400
    # Faster variant: with a single planned search (one company, one year) the check step is
    # skipped. Off by default because it changes behaviour and must be evaluated first.
    skip_check_single: bool = False


class State(TypedDict, total=False):
    question: str
    pending: list[Search]
    evidence: list[dict]
    rounds: int
    groups: Annotated[list[list[str]], operator.add]  # chunk ids per planned search, best first
    trace: Annotated[list[str], operator.add]
    raw_answer: str
    timings: Annotated[list[tuple[str, float]], operator.add]  # (stage, seconds)


def build_agent(
    llm: LLM,
    search: SearchFn,
    filings: dict[str, list[str]],
    names: dict[str, str],
    config: AgentConfig | None = None,
    default_scope: Callable[[str], Search | None] | None = None,
    score_fn: ScoreFn | None = None,
):
    """Return a compiled graph; `invoke({"question": ...})` returns the final state."""
    from langgraph.graph import END, START, StateGraph

    from src.agent.plan import company_table

    config = config or AgentConfig()

    def timed(stage: str, fn):
        def wrapper(state: State) -> State:
            start = time.time()
            update = fn(state)
            return update | {"timings": [(stage, round(time.time() - start, 2))]}

        return wrapper

    known = set(filings)
    table = company_table(filings, names)

    def plan(state: State) -> State:
        text = llm(plan_messages(state["question"], table), config.plan_tokens)
        searches = parse_plan(text, known)
        note = f"plan: {len(searches)} search(es)" if searches else "plan: fallback single search"
        if not searches:
            fallback = default_scope(state["question"]) if default_scope else None
            searches = [fallback or Search(state["question"])]
        searches = _with_question_search(state["question"], searches)
        return {"pending": searches, "evidence": [], "rounds": 0, "trace": [note], "groups": []}

    def run_search(state: State) -> State:
        searches = state["pending"]
        per = config.single_search_chunks if len(searches) == 1 else config.chunks_per_search
        seen = {c["chunk_id"] for c in state["evidence"]}
        evidence = list(state["evidence"])
        lines, groups = [], []
        for item in searches:
            found = search(item.query, item.ticker, item.year, per)
            ids = [chunk["chunk_id"] for chunk in found]
            lines.append(f"search: {item.query!r} ticker={item.ticker} year={item.year} -> {ids}")
            if item.query != state["question"]:  # the plain question search only fills places
                groups.append(ids)
            for chunk in found:
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    evidence.append(chunk)
        return {
            "evidence": evidence,
            "rounds": state["rounds"] + 1,
            "trace": lines,
            "groups": groups,
            "pending": [],
        }

    def check(state: State) -> State:
        snippets = [
            f"[{c['chunk_id']}] {c['text'][:350].replace(chr(10), ' ')}" for c in state["evidence"]
        ]
        text = llm(check_messages(state["question"], snippets), config.check_tokens)
        follow_ups = parse_check_list(text, known)
        if follow_ups:
            note = f"check: not enough, {len(follow_ups)} more search(es)"
            return {"pending": follow_ups, "trace": [note]}
        return {"pending": [], "trace": ["check: enough"]}

    def answer(state: State) -> State:
        context = select_context_by_search(
            state["question"],
            state["evidence"],
            state["groups"],
            score_fn,
            config.max_context_chunks,
            config.per_search_chunks,
        )
        raw = llm(build_messages(state["question"], context), config.answer_tokens)
        return {"raw_answer": raw, "evidence": context, "trace": ["answer"]}

    def after_search(state: State) -> str:
        if state["rounds"] >= config.max_rounds:
            return "answer"
        if config.skip_check_single and len(state["groups"]) <= 1:
            return "answer"
        return "check"

    def after_check(state: State) -> str:
        return "search" if state["pending"] else "answer"

    graph = StateGraph(State)
    graph.add_node("plan", timed("plan", plan))
    graph.add_node("search", timed("search", run_search))
    graph.add_node("check", timed("check", check))
    graph.add_node("answer", timed("answer", answer))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "search")
    graph.add_conditional_edges("search", after_search, {"check": "check", "answer": "answer"})
    graph.add_conditional_edges("check", after_check, {"search": "search", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()


def _with_question_search(question: str, searches: list[Search]) -> list[Search]:
    """Add a search on the original question, scoped like the plan (same single company/year).

    Rewritten queries can drift; the user's own wording is the most faithful query, so it is
    always run as well, and the final reranking decides which passages count.
    """
    tickers = {s.ticker for s in searches if s.ticker}
    years = {s.year for s in searches if s.year}
    base = Search(
        question,
        next(iter(tickers)) if len(tickers) == 1 else None,
        next(iter(years)) if len(years) == 1 else None,
    )
    return searches if base in searches else [*searches, base]


def run_agent(agent, question: str) -> dict:
    """Run the graph and parse the answer the same way as the baseline."""
    state = agent.invoke({"question": question})
    provided = [c["chunk_id"] for c in state["evidence"]]
    parsed = parse_answer(state["raw_answer"], provided)
    seconds: dict[str, float] = {}
    for stage, value in state.get("timings", []):
        seconds[stage] = round(seconds.get(stage, 0.0) + value, 2)
    return parsed | {
        "evidence_ids": provided,
        "trace": state["trace"],
        "rounds": state["rounds"],
        "stage_seconds": seconds,
    }


def regex_fallback(filings: dict[str, list[str]]) -> Callable[[str], Search]:
    """Single search on the question itself, scoped by the rule-based company/year detector."""
    from src.retrieval.scope import detect_scope

    def fallback(question: str) -> Search:
        scope = detect_scope(question, filings)
        years = {int(d[:4]) for _, d in scope.report_dates}
        return Search(
            question,
            scope.tickers[0] if len(scope.tickers) == 1 else None,
            years.pop() if len(years) == 1 else None,
        )

    return fallback
