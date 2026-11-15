"""Evidence cards: a greppable corpus of agronomic studies for the farming agent.

Layout of an evidence root (``data/evidence`` by default)::

    cards/<id>.md          one study per file, YAML-ish front matter + sections
    sources/<id>.txt       plain text of the source actually fetched, first line
                           ``# url: ... fetched: YYYY-MM-DD``

A card looks like::

    ---
    id: kaspar2007-rye-nitrate
    citation: ...
    doi: 10.2134/jeq2006.0468
    url: https://...
    access: open
    study_type: field experiment
    region: Iowa, USA (Boone County)
    climate: humid continental, ~850 mm/yr
    soil: loam, tile drained
    crops: corn, soybean, cereal rye
    topics: cover crop, nitrate leaching, drainage
    years: 2002-2005
    experiment_group: kaspar-boone-rye
    ---
    ## Findings
    - ...
    ## Conditions and limits
    - ...
    ## Quotes
    - "..."
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = ("Findings", "Conditions and limits", "Quotes")

#: Front-matter fields every card must have. ``doi`` is allowed to be ``n/a``
#: (extension guides and old agronomy bulletins have none) but must be present.
REQUIRED_FIELDS = (
    "id",
    "citation",
    "doi",
    "url",
    "access",
    "study_type",
    "region",
    "climate",
    "soil",
    "crops",
    "topics",
    "years",
    "experiment_group",
)

ACCESS_VALUES = ("open", "abstract-only")

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)


class EvidenceError(ValueError):
    """A card is malformed or the corpus is inconsistent."""


def normalise(text: str) -> str:
    """Collapse whitespace and unify quote/dash characters for comparison."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in (
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2010", "-"),
        ("\u2011", "-"),
        ("\u2012", "-"),
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2015", "-"),
        ("\u2212", "-"),
        ("\u00a0", " "),
    ):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


def _bullet_text(section: str) -> list[str]:
    """Return the text of the ``- `` bullets of a section body."""
    out: list[str] = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif line.startswith("-"):
            out.append(line[1:].strip())
        elif line and out:
            out[-1] = f"{out[-1]} {line}"
    return [item for item in out if item]


@dataclass
class Card:
    """One parsed study card."""

    path: Path
    meta: dict[str, str]
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def group(self) -> str:
        return self.meta["experiment_group"]

    @property
    def findings(self) -> list[str]:
        return _bullet_text(self.sections.get("Findings", ""))

    @property
    def quotes(self) -> list[str]:
        """Quote bullets with the wrapping quotation marks stripped."""
        out = []
        for bullet in _bullet_text(self.sections.get("Quotes", "")):
            text = bullet.strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\u201c\u201d':
                text = text[1:-1]
            else:
                text = text.strip('"')
            out.append(text.strip())
        return out

    @property
    def summary(self) -> str:
        findings = self.findings
        return findings[0] if findings else ""

    @property
    def tags(self) -> list[str]:
        return _split_list(self.meta.get("topics", ""))


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_card(path: Path) -> Card:
    """Parse one card file, raising :class:`EvidenceError` on malformed input."""
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        raise EvidenceError(f"{path.name}: missing front matter block delimited by '---'")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise EvidenceError(f"{path.name}: front-matter line without ':': {line!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()

    missing = [name for name in REQUIRED_FIELDS if not meta.get(name)]
    if missing:
        raise EvidenceError(f"{path.name}: missing required field(s): {', '.join(missing)}")
    if meta["access"] not in ACCESS_VALUES:
        raise EvidenceError(
            f"{path.name}: access must be one of {ACCESS_VALUES}, got {meta['access']!r}"
        )

    body = raw[match.end() :]
    marks = list(_SECTION_RE.finditer(body))
    sections: dict[str, str] = {}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        sections[mark.group(1).strip()] = body[mark.end() : end]
    missing_sections = [name for name in SECTIONS if name not in sections]
    if missing_sections:
        raise EvidenceError(f"{path.name}: missing section(s): {', '.join(missing_sections)}")
    if not _bullet_text(sections["Findings"]):
        raise EvidenceError(f"{path.name}: 'Findings' has no bullets")
    return Card(path=path, meta=meta, sections=sections)


def load_cards(root: str | Path) -> list[Card]:
    """Load every card under ``<root>/cards``, sorted by id."""
    root = Path(root)
    cards_dir = root / "cards"
    if not cards_dir.is_dir():
        raise EvidenceError(f"no cards directory at {cards_dir}")
    cards: list[Card] = []
    seen: dict[str, Path] = {}
    for path in sorted(cards_dir.glob("*.md")):
        card = parse_card(path)
        if card.id in seen:
            raise EvidenceError(
                f"duplicate id {card.id!r} in {path.name} and {seen[card.id].name}"
            )
        seen[card.id] = path
        cards.append(card)
    if not cards:
        raise EvidenceError(f"no cards found in {cards_dir}")
    return sorted(cards, key=lambda c: c.id)


def source_path(root: str | Path, card_id: str) -> Path:
    return Path(root) / "sources" / f"{card_id}.txt"


def read_source(root: str | Path, card_id: str) -> str:
    path = source_path(root, card_id)
    if not path.is_file():
        raise EvidenceError(f"missing source text for card {card_id!r}: {path}")
    return normalise(path.read_text(encoding="utf-8"))


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _number_in(number: str, haystack: str) -> bool:
    """True when ``number`` occurs in ``haystack``, ignoring thousands separators."""
    if number in haystack:
        return True
    plain = number.replace(",", "")
    return plain in haystack.replace(",", "")


def check_card(card: Card, root: str | Path) -> list[str]:
    """Return the violations of one card (empty list means it is clean)."""
    problems: list[str] = []
    try:
        source = read_source(root, card.id)
    except EvidenceError as exc:
        return [str(exc)]

    for quote in card.quotes:
        if not quote:
            problems.append(f"{card.id}: empty quote")
        elif normalise(quote) not in source:
            problems.append(f"{card.id}: quote not found in source text: {quote[:90]!r}")

    if not card.quotes:
        problems.append(f"{card.id}: no quotes (every card must quote its source)")

    findings_text = " ".join(card.findings)
    seen: set[str] = set()
    for number in _numbers(findings_text):
        if number in seen:
            continue
        seen.add(number)
        if not _number_in(number, source):
            problems.append(
                f"{card.id}: number {number!r} from Findings does not appear in the source text"
            )
    return problems


def check_quotes(root: str | Path) -> list[str]:
    """Check every card against its stored source text.

    Verifies that each quote appears verbatim (after whitespace normalisation)
    and that every number used in ``Findings`` occurs in the source text.
    """
    root = Path(root)
    violations: list[str] = []
    for card in load_cards(root):
        violations.extend(check_card(card, root))
    return violations


def _region_key(card: Card) -> str:
    """Coarse region label for grouping the index (``Iowa, USA``, ``global``)."""
    region = card.meta["region"].split("(")[0].strip().rstrip(",")
    parts = [part.strip() for part in region.split(",") if part.strip()]
    if len(parts) >= 2 and parts[-1].upper() == "USA":
        return f"{parts[0]}, USA"
    if not parts:
        return "other"
    return parts[0]


def _index_line(card: Card) -> str:
    meta = card.meta
    summary = normalise(card.summary)
    if len(summary) > 400:
        summary = summary[:397].rstrip() + "..."
    return (
        f"- **{card.id}** ({meta['region']}; {meta['crops']}; {meta['topics']}; "
        f"{meta['access']}) - {summary}"
    )


def export(
    root: str | Path,
    dest: str | Path,
    exclude_groups: tuple[str, ...] | list[str] = (),
) -> list[Card]:
    """Copy cards (not sources) to ``dest`` and write ``dest/INDEX.md``."""
    root, dest = Path(root), Path(dest)
    exclude = set(exclude_groups)
    cards = [card for card in load_cards(root) if card.group not in exclude]

    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.md"):
        if stale.name != "INDEX.md":
            stale.unlink()

    for card in cards:
        shutil.copyfile(card.path, dest / card.path.name)

    lines = [
        "# Evidence index",
        "",
        f"{len(cards)} study cards. Read this index first, pick the cards that match the",
        "farm's region, crops and question, then read only those files.",
        "",
    ]
    by_group: dict[str, list[Card]] = {}
    for card in cards:
        by_group.setdefault(_region_key(card), []).append(card)
    for region in sorted(by_group):
        lines.append(f"## {region}")
        lines.extend(_index_line(card) for card in sorted(by_group[region], key=lambda c: c.id))
        lines.append("")
    groups = sorted({card.group for card in cards})
    lines.append("## Experiment groups (hold out whole groups when testing transfer)")
    lines.append("")
    for group in groups:
        members = sorted(c.id for c in cards if c.group == group)
        lines.append(f"- `{group}`: {', '.join(members)}")
    lines.append("")
    (dest / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    return cards


def default_root() -> Path:
    """``<repo>/data/evidence`` for a source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent / "data" / "evidence"
    return Path.cwd() / "data" / "evidence"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldshift.agent.evidence", description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=None, help="evidence root (default: data/evidence)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify quotes and numbers against the stored sources")

    exp = sub.add_parser("export", help="copy cards to a directory and write INDEX.md")
    exp.add_argument("--dest", type=Path, required=True)
    exp.add_argument(
        "--exclude-group",
        action="append",
        default=[],
        dest="exclude_groups",
        help="experiment_group to leave out (repeatable)",
    )

    args = parser.parse_args(argv)
    root = args.root or default_root()

    if args.command == "check":
        cards = load_cards(root)
        violations = check_quotes(root)
        groups = sorted({card.group for card in cards})
        print(f"{len(cards)} cards, {len(groups)} experiment groups, {len(violations)} violations")
        for problem in violations:
            print(f"  {problem}")
        return 1 if violations else 0

    cards = export(root, args.dest, tuple(args.exclude_groups))
    print(f"exported {len(cards)} cards to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
