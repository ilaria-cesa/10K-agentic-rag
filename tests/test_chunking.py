from itertools import pairwise

import pytest

from src.chunking.chunker import ChunkConfig, build_chunks, chunk_text, split_long_line

CFG = ChunkConfig(max_chars=200, overlap_chars=60, min_chars=50)


def lines(n, width=40):
    return "\n".join(f"line{i:03d} " + "x" * (width - 8) for i in range(n))


def test_chunks_respect_max_size_and_keep_all_text():
    text = lines(30)
    chunks = chunk_text(text, CFG)
    assert all(len(c) <= CFG.max_chars + CFG.min_chars for c in chunks)
    covered = {line for c in chunks for line in c.splitlines()}
    assert covered == set(text.splitlines())


def test_consecutive_chunks_overlap_by_whole_lines():
    chunks = chunk_text(lines(30), CFG)
    assert len(chunks) > 2
    for a, b in pairwise(chunks):
        shared = set(a.splitlines()) & set(b.splitlines())
        assert shared, "no overlap between consecutive chunks"
        assert len("\n".join(sorted(shared))) <= CFG.overlap_chars + 5


def test_lines_are_not_cut_in_the_middle():
    for chunk in chunk_text(lines(30), CFG):
        assert all(line.endswith("x") for line in chunk.splitlines())


def test_long_line_is_split_at_sentences_then_hard_wrapped():
    sentence = "This is a sentence about risk. "
    pieces = split_long_line(sentence * 20, 100)
    assert all(len(p) <= 100 for p in pieces) and len(pieces) > 1
    assert all(p.endswith(".") for p in pieces)
    hard = split_long_line("y" * 450, 100)
    assert [len(p) for p in hard] == [100, 100, 100, 100, 50]


def test_tiny_trailing_chunk_is_merged_without_duplicating_text():
    text = lines(5) + "\ntail"
    chunks = chunk_text(text, ChunkConfig(max_chars=200, overlap_chars=60, min_chars=100))
    assert chunks[-1].endswith("tail")
    joined = "\n".join(chunks)
    assert joined.count("tail") == 1


def test_empty_and_blank_input():
    assert chunk_text("", CFG) == []
    assert chunk_text("\n  \n", CFG) == []


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        ChunkConfig(max_chars=100, overlap_chars=60)


def test_build_chunks_adds_metadata_and_context_header():
    record = {
        "ticker": "JPM",
        "company": "JPMORGAN CHASE & CO",
        "report_date": "2025-12-31",
        "sections": {"risk_factors": lines(12)},
    }
    rows = build_chunks(record, CFG)
    assert rows[0]["chunk_id"] == "JPM_2025-12-31_risk_factors_0000"
    assert [r["position"] for r in rows] == list(range(len(rows)))
    assert rows[0]["embed_text"].startswith(
        "JPMORGAN CHASE & CO | Form 10-K | period ended 2025-12-31 | Item 1A Risk Factors\n"
    )
    assert rows[0]["text"] in rows[0]["embed_text"]
    assert "JPMORGAN" not in rows[0]["text"]


def test_lone_heading_is_attached_to_the_following_long_paragraph():
    cfg = ChunkConfig(max_chars=200, overlap_chars=40, min_chars=60)
    text = "HEADING\n" + "p" * 195 + "\n" + "q" * 195
    chunks = chunk_text(text, cfg)
    assert chunks[0].startswith("HEADING\n" + "p")
    assert all(len(c) > 60 for c in chunks)


def test_page_footers_are_dropped_but_sentences_mentioning_the_form_are_kept():
    text = (
        "Apple Inc. | 2025 Form 10-K | 7\n"
        + "See our Annual Report on Form 10-K for the year ended December 31, 2025 for more details on this."
    )
    chunks = chunk_text(text, ChunkConfig())
    assert len(chunks) == 1
    assert chunks[0].startswith("See our Annual Report")


TABLE = (
    "[TABLE]\nTable: Net sales (in millions):\n"
    + "\n".join(f"Product{i}: 2026 = {i * 10}; 2025 = {i * 9}" for i in range(12))
    + "\n[/TABLE]"
)


def test_text_without_tables_is_unchanged():
    assert chunk_text(lines(30), CFG) == chunk_text(lines(30) + "\n", CFG)
    assert all("[TABLE]" not in c for c in chunk_text(lines(30), CFG))


def test_table_is_separate_chunk_without_markers():
    chunks = chunk_text("Intro paragraph about sales.\n" + TABLE + "\nAfter the table.", CFG)
    tables = [c for c in chunks if c.startswith("Table:")]
    assert tables and all("[TABLE]" not in c and "[/TABLE]" not in c for c in chunks)
    assert not any("Intro" in c and "Product" in c for c in chunks)


def test_long_table_is_cut_between_rows_and_repeats_title():
    chunks = [c for c in chunk_text(TABLE, CFG)]
    assert len(chunks) > 1
    assert all(c.startswith("Table: Net sales (in millions):") for c in chunks)
    assert all(len(c) <= CFG.max_chars for c in chunks)
    rows = [line for c in chunks for line in c.splitlines()[1:]]
    assert rows == [f"Product{i}: 2026 = {i * 10}; 2025 = {i * 9}" for i in range(12)]
