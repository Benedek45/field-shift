"""Offline tests for the evidence-card corpus helpers."""

from __future__ import annotations

import pytest

from fieldshift.agent.evidence import (
    EvidenceError,
    check_quotes,
    export,
    load_cards,
    normalise,
)


def card_text(
    card_id: str,
    *,
    group: str = "group-a",
    findings: str = "- Rye cover crop cut nitrate load by 61% over four years.",
    quotes: str = '- "The rye cover crop reduced nitrate load by 61% in tile drainage"',
    topics: str = "cover crop, nitrate",
    region: str = "Iowa, USA",
    crops: str = "corn, soybean",
    extra: str = "",
) -> str:
    return f"""---
id: {card_id}
citation: Test citation for {card_id}.
doi: 10.0000/test
url: https://example.org/{card_id}
access: open
study_type: field experiment
region: {region}
climate: humid continental
soil: loam
crops: {crops}
topics: {topics}
years: 2002-2005
experiment_group: {group}
{extra}---
## Findings
{findings}
## Conditions and limits
- One site only.
## Quotes
{quotes}
"""


SOURCE = (
    "# url: https://example.org/test fetched: 2026-11-14\n"
    "The rye cover crop reduced nitrate load by 61% in tile drainage over four years of\n"
    "corn-soybean no-till at the Boone County site.\n"
)


def write_card(root, card_id: str, text: str) -> None:
    (root / "cards").mkdir(parents=True, exist_ok=True)
    (root / "cards" / f"{card_id}.md").write_text(text, encoding="utf-8")


def write_source(root, card_id: str, text: str = SOURCE) -> None:
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "sources" / f"{card_id}.txt").write_text(text, encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "evidence"
    write_card(root, "good-card", card_text("good-card"))
    write_source(root, "good-card")
    return root


def test_load_cards_parses_front_matter_and_sections(corpus):
    cards = load_cards(corpus)
    assert [card.id for card in cards] == ["good-card"]
    card = cards[0]
    assert card.meta["region"] == "Iowa, USA"
    assert card.group == "group-a"
    assert card.tags == ["cover crop", "nitrate"]
    assert card.findings == ["Rye cover crop cut nitrate load by 61% over four years."]
    assert card.quotes == ["The rye cover crop reduced nitrate load by 61% in tile drainage"]


def test_load_cards_rejects_missing_field(tmp_path):
    root = tmp_path / "evidence"
    broken = card_text("broken").replace("study_type: field experiment\n", "")
    write_card(root, "broken", broken)
    write_source(root, "broken")
    with pytest.raises(EvidenceError, match="missing required field"):
        load_cards(root)


def test_load_cards_rejects_duplicate_id(tmp_path):
    root = tmp_path / "evidence"
    write_card(root, "dup", card_text("dup"))
    write_card(root, "dup-other", card_text("dup").replace("good-card", "dup"))
    write_source(root, "dup")
    with pytest.raises(EvidenceError, match="duplicate id"):
        load_cards(root)


def test_check_quotes_passes_for_matching_card(corpus):
    assert check_quotes(corpus) == []


def test_check_quotes_flags_quote_not_in_source(corpus):
    write_card(
        corpus,
        "good-card",
        card_text(
            "good-card",
            quotes='- "The rye cover crop eliminated nitrate load entirely."',
        ),
    )
    problems = check_quotes(corpus)
    assert len(problems) == 1
    assert "quote not found" in problems[0]


def test_check_quotes_flags_fabricated_number(corpus):
    write_card(
        corpus,
        "good-card",
        card_text(
            "good-card",
            findings="- Rye cover crop cut nitrate load by 73% over four years.",
        ),
    )
    problems = check_quotes(corpus)
    assert len(problems) == 1
    assert "73" in problems[0]


def test_check_quotes_tolerates_thousands_separators(corpus):
    write_card(
        corpus,
        "good-card",
        card_text("good-card", findings="- Rye produced 1,250 kg per ha of biomass."),
    )
    problems = check_quotes(corpus)
    assert any("1,250" in problem for problem in problems)
    write_source(corpus, "good-card", SOURCE + "Biomass reached 1,250 kg per ha in spring.\n")
    assert check_quotes(corpus) == []


def test_check_quotes_requires_a_source_file(corpus):
    (corpus / "sources" / "good-card.txt").unlink()
    problems = check_quotes(corpus)
    assert len(problems) == 1
    assert "missing source text" in problems[0]


def test_export_writes_index_and_excludes_group(tmp_path):
    root = tmp_path / "evidence"
    write_card(root, "keep", card_text("keep", group="keep-me"))
    write_source(root, "keep")
    write_card(root, "drop", card_text("drop", group="hold-out"))
    write_source(root, "drop")
    dest = tmp_path / "out"

    cards = export(root, dest, exclude_groups=("hold-out",))

    assert [card.id for card in cards] == ["keep"]
    assert sorted(p.name for p in dest.glob("*.md")) == ["INDEX.md", "keep.md"]
    index = (dest / "INDEX.md").read_text(encoding="utf-8")
    assert "keep" in index
    assert "drop" not in index
    assert "hold-out" not in index
    # Sources are never copied to the export target.
    assert not (dest / "sources").exists()


def test_export_index_has_one_line_per_card(tmp_path):
    root = tmp_path / "evidence"
    for card_id, region in (("a-card", "Iowa, USA"), ("b-card", "Kansas, USA")):
        write_card(root, card_id, card_text(card_id, region=region))
        write_source(root, card_id)
    dest = tmp_path / "out"
    export(root, dest)
    index = (dest / "INDEX.md").read_text(encoding="utf-8")
    bullet_lines = [line for line in index.splitlines() if line.startswith("- **")]
    assert len(bullet_lines) == 2
    assert bullet_lines[0].startswith("- **a-card**")
    assert "Iowa, USA" in bullet_lines[0]
    assert "cover crop, nitrate" in bullet_lines[0]


def test_export_replaces_stale_cards(tmp_path):
    root = tmp_path / "evidence"
    write_card(root, "keep", card_text("keep"))
    write_source(root, "keep")
    dest = tmp_path / "out"
    export(root, dest)
    (dest / "stale.md").write_text("old card", encoding="utf-8")
    export(root, dest)
    assert not (dest / "stale.md").exists()


def test_normalise_collapses_whitespace_and_unicode():
    assert normalise("a  b\n\tc") == "a b c"
    assert normalise("non\u2010legumes \u201cquoted\u201d") == 'non-legumes "quoted"'
