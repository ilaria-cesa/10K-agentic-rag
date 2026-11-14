from bs4 import BeautifulSoup

from src.ingest.tables import merge_fragments, parse_table, serialize


def table(rows: list[list[str]]):
    html = (
        "<table>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        + "</table>"
    )
    return BeautifulSoup(html, "lxml").find("table")


# The structure of a real 10-K table (Apple): values split in pieces and many empty cells.
APPLE = [
    [""] * 12,
    ["", "2026", "", "Change", "", "2025", "", "Change", "", "2024"],
    [
        "iPhone",
        "$",
        "254,098",
        "",
        "",
        "21",
        "%",
        "",
        "$",
        "209,586",
        "",
        "",
        "4",
        "%",
        "",
        "$",
        "201,183",
    ],
    ["iPad", "27,929", "", "", "—", "%", "", "28,023", "", "", "5", "%", "", "26,694"],
    ["Wearables", "35,577", "", "", "—", "%", "", "35,686", "", "", "(4", ")", "%", "", "37,005"],
]


def test_merge_fragments_glues_values_and_drops_empty_cells():
    assert merge_fragments(["iPhone", "$", "254,098", "", "21", "%"]) == [
        "iPhone",
        "$254,098",
        "21%",
    ]
    assert merge_fragments(["x", "(4", ")", "%"]) == ["x", "(4)%"]
    assert merge_fragments(["—", "%"]) == ["—%"]


def test_every_value_carries_its_row_and_its_column():
    parsed = parse_table(table(APPLE))
    lines = serialize(parsed).splitlines()
    assert lines[1] == (
        "iPhone: 2026 = $254,098; Change 2026 vs 2025 = 21%; 2025 = $209,586; "
        "Change 2025 vs 2024 = 4%; 2024 = $201,183"
    )
    assert "iPad: 2026 = 27,929; Change 2026 vs 2025 = —%; 2025 = 28,023;" in lines[2]
    assert "Change 2025 vs 2024 = (4)%" in lines[3]


def test_unit_and_heading_rows_are_kept():
    rows = [
        ["(In millions)", "2025", "2024"],
        ["Assets"],
        ["Cash", "$", "1,000", "$", "900"],
    ]
    parsed = parse_table(table(rows))
    text = serialize(parsed, title="Balance sheet")
    assert text.splitlines()[0] == "Table: Balance sheet (In millions):"
    assert "Assets" in text.splitlines()
    assert "Cash: 2025 = $1,000; 2024 = $900" in text


def test_a_spanning_title_row_becomes_context_not_a_column():
    rows = [["Year ended December 31,"], ["", "2025", "2024"], ["Revenue", "10", "9"]]
    text = serialize(parse_table(table(rows)))
    assert "Year ended December 31," in text and "Revenue: 2025 = 10; 2024 = 9" in text


def test_a_row_that_does_not_match_the_header_is_not_guessed():
    rows = [["", "2025", "2024"], ["Revenue", "10", "9"], ["Odd row", "5"]]
    lines = serialize(parse_table(table(rows))).splitlines()
    assert lines[-1] == "Odd row: 5" and "2025 = 10" in lines[-2]


def test_a_table_without_numbers_is_left_as_text():
    assert parse_table(table([["Name", "Role"], ["Jane", "CEO"]])) is None


def test_a_row_label_can_be_a_year():
    rows = [["", "Amount"], ["2025", "$", "1,200"], ["2026", "$", "900"]]
    assert "2025: Amount = $1,200" in serialize(parse_table(table(rows)))


def test_group_titles_are_repeated_over_the_columns_they_cover():
    rows = [
        ["(in millions)", "Credit exposure", "Nonperforming"],
        ["", "2025", "2024", "2025", "2024"],
        ["Loans", "$", "100", "$", "90", "5", "4"],
    ]
    text = serialize(parse_table(table(rows)))
    assert (
        "Loans: Credit exposure 2025 = $100; Credit exposure 2024 = $90; "
        "Nonperforming 2025 = 5; Nonperforming 2024 = 4"
    ) in text


def test_percentage_rows_align_to_the_value_columns():
    from src.ingest.tables import Table, serialize

    table = Table(
        header=["2025", "Change 2025 vs 2024", "2024", "Change 2024 vs 2023", "2023"],
        rows=[
            ("R&D", ["$34,550", "10%", "$31,370", "5%", "$29,915"]),
            ("Percentage of total net sales", ["8%", "8%", "8%"]),
        ],
    )
    plain = serialize(table, "Operating Expenses")
    assert "Percentage of total net sales: 8% | 8% | 8%" in plain
    aligned = serialize(table, "Operating Expenses", align_to_base=True)
    assert "Percentage of total net sales: 2025 = 8%; 2024 = 8%; 2023 = 8%" in aligned


def test_mostly_percent_and_orphan_header():
    from src.ingest.tables import Table, inherit_header, mostly_percent

    above = Table(
        header=["2026", "2025", "2024"], rows=[("Products", ["$1", "$2", "$3"])], unit="in millions"
    )
    orphan = Table(header=[], rows=[("Services", ["76.2%", "75.4%", "73.9%"])])
    assert mostly_percent(orphan) and not mostly_percent(above)
    assert inherit_header(orphan, above) and orphan.header == ["2026", "2025", "2024"]
    other = Table(header=[], rows=[("Services", ["1", "2"])])  # different number of columns
    assert not inherit_header(other, above) and other.header == []
