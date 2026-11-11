"""The four systems compared in the project, and a resumable runner.

Systems (same model, same answer prompt, greedy decoding):
  rag_base : dense search over all filings, top 5 passages (no company/year filter)
  rag5     : company/year filter + cross-encoder reranker, top 5 passages
  rag8     : company/year filter + BM25/dense hybrid + reranker, top 8 passages
  agent_v3 : LangGraph agent (plan -> search -> checklist -> answer), 8 passages

Heavy objects (model, embedder, reranker) are passed in, so this module imports without a GPU
and the runner can be tested with fakes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from src.agent.graph import run_agent
from src.agent.parsing import parse_answer
from src.agent.prompts import build_messages
from src.eval.metrics import score_row
from src.retrieval.pipeline import RetrievalConfig, retrieve

RAG_BASE = RetrievalConfig("dense, no filter", use_filter=False, rerank=False)
RAG5 = RetrievalConfig("filter + rerank", use_filter=True, rerank=True)
RAG8 = RetrievalConfig("filter + hybrid + rerank", use_filter=True, hybrid=True, rerank=True)

SYSTEM_NAMES = ("rag_base", "rag5", "rag8", "agent_v3")
SystemFn = Callable[[dict], dict]  # golden question -> result row


def _identity(q: dict, system: str, model: str) -> dict:
    return {"system": system, "split": q.get("split"), "model": model, "question": q["question"]}


def make_rag_system(
    name: str,
    config: RetrievalConfig,
    k: int,
    *,
    corpus,
    embed_query: Callable[[str], object],
    reranker,
    generate: Callable[[list[dict]], dict],
    model_name: str,
) -> SystemFn:
    """`generate(messages)` returns {"text", "prompt_tokens", ...}; `embed_query` a vector."""
    chunks = corpus.chunks

    def run(q: dict) -> dict:
        start = time.time()
        idx, _ = retrieve(corpus, q["question"], embed_query(q["question"]), config, reranker)
        retrieval_seconds = time.time() - start
        retrieved_ids = [chunks[i]["chunk_id"] for i in idx]
        top = [chunks[i] for i in idx[:k]]
        provided = [c["chunk_id"] for c in top]
        out = generate(build_messages(q["question"], top))
        parsed = parse_answer(out["text"], provided)
        row = score_row(q, parsed, retrieved_ids)
        if q["expected"] == "answer":
            row["all@k_given"] = float(set(q["evidence_chunk_ids"]) <= set(provided))
        return (
            row
            | _identity(q, name, model_name)
            | {
                "answer": parsed["answer"],
                "cited_ids": parsed["cited_ids"],
                "given_ids": provided,
                "prompt_tokens": out["prompt_tokens"],
                "t_retrieval": round(retrieval_seconds, 2),
                "t_answer": out.get("seconds"),
                "seconds": round(time.time() - start, 1),
            }
        )

    return run


def make_agent_system(name: str, agent, *, model_name: str) -> SystemFn:
    def run(q: dict) -> dict:
        start = time.time()
        out = run_agent(agent, q["question"])
        row = score_row(q, out, out["evidence_ids"])
        if q["expected"] == "answer":
            row["all@k_given"] = float(set(q["evidence_chunk_ids"]) <= set(out["evidence_ids"]))
        times = {f"t_{stage}": sec for stage, sec in out["stage_seconds"].items()}
        return (
            row
            | _identity(q, name, model_name)
            | {
                "answer": out["answer"],
                "cited_ids": out["cited_ids"],
                "given_ids": out["evidence_ids"],
                "rounds": out["rounds"],
                "trace": out["trace"],
                "seconds": round(time.time() - start, 1),
            }
            | times
        )

    return run


def run_system(
    questions: list[dict],
    fn: SystemFn,
    out_path: str | Path,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run `fn` on every question, appending each row to `out_path` as soon as it is done.

    If the file already holds rows (a Colab session that died), those questions are skipped,
    so re-running the notebook continues instead of starting over.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    done = {r["id"] for r in rows}
    with open(path, "a", encoding="utf-8") as f:
        for q in questions:
            if q["id"] in done:
                continue
            row = fn(q)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            log(f"{row['system']} {q['id']} {q['type']} {'ok' if row['behaviour_ok'] else 'WRONG'}")
    order = {q["id"]: i for i, q in enumerate(questions)}
    return sorted(rows, key=lambda r: order.get(r["id"], 10**9))


def single_plan_ids(rows: list[dict]) -> list[str]:
    """Ids of the agent runs whose first search round had at most one planned search.

    Only these questions are affected by `AgentConfig(skip_check_single=True)`: for all the
    others the agent behaves exactly as before, so they do not need to be run again.
    """
    ids = []
    for row in rows:
        planned = 0
        for line in row["trace"]:
            if not line.startswith("search:"):
                if planned or line.startswith(("check", "answer")):
                    break
                continue
            if not line.startswith(f"search: {row['question']!r} ticker"):  # the plain question
                planned += 1
        if planned <= 1:
            ids.append(row["id"])
    return ids
