from src.eval.metrics import (
    key_numbers,
    numbers_match,
    retrieval_metrics,
    score_row,
    summarise,
)


def test_retrieval_metrics_recall_hit_all_and_mrr():
    m = retrieval_metrics(["a", "b"], ["x", "a", "y", "z", "b"])
    assert m["recall@1"] == 0 and m["hit@3"] == 1 and m["all@3"] == 0
    assert m["recall@5"] == 1 and m["all@5"] == 1
    assert m["mrr"] == 0.5
    assert retrieval_metrics(["a"], ["x", "y"])["mrr"] == 0


def test_numbers_match_handles_units_percentages_and_years():
    assert key_numbers("$1,927 branches in 5 states, 22% in 2025") == [
        (1927.0, "plain"),
        (22.0, "percent"),
    ]
    assert numbers_match("$38.219 billion", "fell from $38,219 million") is True
    assert numbers_match("Approximately $1.609 trillion.", "$1,609.1 billion") is True
    assert numbers_match("$5.943 billion", "about $5.9 billion") is True
    assert numbers_match("$168.6 billion", "about $150 billion") is False
    assert numbers_match("22% of deposits", "21% of deposits") is False
    assert numbers_match("It fell in FY2025.", "It fell in FY2024.") is None
    assert numbers_match("Florida", "Florida") is None
    assert numbers_match("In 2025 it rose", "In 2024 it rose") is None


def test_score_row_refusal_and_citations():
    q = {
        "id": "q",
        "type": "numeric",
        "expected": "answer",
        "answer": "47,400 employees",
        "evidence_chunk_ids": ["c1"],
    }
    parsed = {
        "refused": False,
        "answer": "47,400 [c1]",
        "cited_ids": ["c1"],
        "invalid_citations": [],
    }
    row = score_row(q, parsed, ["c1", "c2"])
    assert row["behaviour_ok"] and row["cites_evidence"] and row["numbers_match"]
    assert row["hit@1"] == 1
    refuse = {
        "id": "u",
        "type": "unanswerable",
        "expected": "refuse",
        "answer": "",
        "evidence_chunk_ids": [],
    }
    wrong = {"refused": False, "answer": "x", "cited_ids": [], "invalid_citations": []}
    assert score_row(refuse, wrong, [])["behaviour_ok"] is False


def test_summarise_groups_by_type():
    rows = [
        {"id": "1", "type": "fact", "behaviour_ok": True},
        {"id": "2", "type": "fact", "behaviour_ok": False},
        {"id": "3", "type": "numeric", "behaviour_ok": True},
    ]
    s = summarise(rows)
    assert s["all"]["n"] == 3 and s["fact"]["behaviour_ok"] == 0.5


def test_summarise_skips_text_and_list_columns():
    rows = [
        {
            "id": "1",
            "type": "fact",
            "ok": True,
            "answer": "text",
            "cited_ids": ["a"],
            "seconds": 2.0,
        },
        {"id": "2", "type": "fact", "ok": False, "answer": "more", "cited_ids": [], "seconds": 4.0},
    ]
    s = summarise(rows)["all"]
    assert s["ok"] == 0.5 and s["seconds"] == 3.0
    assert "answer" not in s and "cited_ids" not in s
