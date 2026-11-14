def test_structured2_fixes_split_table_titles_and_units():
    html = (
        "<p>Gross Margin</p>"
        "<table><tr><td></td><td>2026</td><td>2025</td></tr>"
        "<tr><td>Products</td><td>$</td><td>100</td><td>$</td><td>90</td></tr></table>"
        "<table><tr><td>Gross margin percentage</td><td>39.3</td><td>%</td><td>36.8</td><td>%</td></tr></table>"
        "<p>The following table summarizes something about the company in a long sentence.</p>"
        "<table><tr><td>Services</td><td>1</td><td>2</td></tr></table>"
    )
    from src.ingest.parse import html_to_text

    text = html_to_text(html, tables="structured2")
    assert "Gross margin percentage: 2026 = 39.3%; 2025 = 36.8%" in text
    assert text.count("(in millions)") == 0
