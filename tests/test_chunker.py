def test_small_table_joins_the_paragraph_before_it():
    from src.chunking.chunker import ChunkConfig, chunk_text

    text = (
        "Gross margin was 48.8%.\n[TABLE]\nTable: Gross Margin:\nProducts: 2026 = 39.3%\n[/TABLE]\n"
    )
    assert len(chunk_text(text, ChunkConfig())) == 2
    merged = chunk_text(text, ChunkConfig(small_table_chars=400))
    assert len(merged) == 1 and merged[0].startswith("Gross margin") and "Products" in merged[0]
