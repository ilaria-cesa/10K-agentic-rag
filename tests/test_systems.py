import json

from src.eval.systems import run_system


def question(i):
    return {"id": f"Q{i:03d}", "type": "fact", "split": "dev", "question": "q"}


def fake_system(calls):
    def run(q):
        calls.append(q["id"])
        return {"id": q["id"], "type": q["type"], "system": "fake", "behaviour_ok": True}

    return run


def test_run_system_saves_rows_and_resumes_after_a_crash(tmp_path):
    path = tmp_path / "out" / "fake.jsonl"
    questions = [question(1), question(2), question(3)]
    first = []
    run_system(questions[:2], fake_system(first), path, log=lambda _: None)
    second = []
    rows = run_system(questions, fake_system(second), path, log=lambda _: None)
    assert first == ["Q001", "Q002"] and second == ["Q003"]
    assert [r["id"] for r in rows] == ["Q001", "Q002", "Q003"]
    assert len(path.read_text().splitlines()) == 3
    assert json.loads(path.read_text().splitlines()[0])["id"] == "Q001"


def test_single_plan_ids_ignores_the_plain_question_search():
    from src.eval.systems import single_plan_ids

    one = {
        "id": "Q1",
        "question": "q1",
        "trace": [
            "plan: 1 search(es)",
            "search: 'revenue' ticker=AAA year=2025 -> []",
            "search: 'q1' ticker=AAA year=2025 -> []",
            "check: enough",
            "answer",
        ],
    }
    two = {
        "id": "Q2",
        "question": "q2",
        "trace": [
            "plan: 2 search(es)",
            "search: 'a' ticker=AAA year=2024 -> []",
            "search: 'a' ticker=AAA year=2025 -> []",
            "search: 'q2' ticker=None year=None -> []",
            "check: enough",
            "answer",
        ],
    }
    assert single_plan_ids([one, two]) == ["Q1"]
