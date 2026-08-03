"""Structural checks for the Paper-A group-meeting claim deck (shipped artifact)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

DECK = Path(__file__).resolve().parents[1] / "paper" / "PaperA_claims_group_meeting.pptx"


def _slide_xmls(zf: zipfile.ZipFile) -> list[str]:
    names = [n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)]
    names.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)))
    return names


def _all_text(zf: zipfile.ZipFile) -> str:
    chunks: list[str] = []
    for name in _slide_xmls(zf):
        xml = zf.read(name).decode("utf-8", errors="ignore")
        chunks.extend(re.findall(r"<a:t[^>]*>([^<]*)</a:t>", xml))
    return "\n".join(chunks)


def _all_srgb(zf: zipfile.ZipFile) -> set[str]:
    colors: set[str] = set()
    for name in zf.namelist():
        if not name.endswith(".xml"):
            continue
        xml = zf.read(name).decode("utf-8", errors="ignore")
        colors.update(re.findall(r'srgbClr val="([0-9A-Fa-f]{6})"', xml, flags=re.I))
    return {c.upper() for c in colors}


def _is_gray(hex6: str) -> bool:
    h = hex6.upper()
    return h[0:2] == h[2:4] == h[4:6]


@pytest.mark.skipif(not DECK.exists(), reason="group meeting deck not generated")
def test_deck_exists_and_slide_count_in_range():
    assert DECK.is_file()
    with zipfile.ZipFile(DECK) as zf:
        n = len(_slide_xmls(zf))
    assert 7 <= n <= 14, f"expected 7–14 slides, got {n}"


@pytest.mark.skipif(not DECK.exists(), reason="group meeting deck not generated")
def test_deck_states_all_claims_and_key_experiments():
    with zipfile.ZipFile(DECK) as zf:
        text = _all_text(zf)
    for claim in ("C1", "C2", "C3", "C4", "C5"):
        assert claim in text
    for exp in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        assert exp in text
    # headline anchors from Paper-A reports
    for anchor in ("byte-identical", "label-swap", "dose 10", "0.733", "θ*"):
        assert anchor in text
    # pure English body: no CJK
    assert not re.search(r"[\u4e00-\u9fff]", text)


@pytest.mark.skipif(not DECK.exists(), reason="group meeting deck not generated")
def test_deck_has_claim_bridge_for_each_of_c1_through_c5():
    """Criterion 2: each claim needs an explicit bridge/falsifier line."""
    with zipfile.ZipFile(DECK) as zf:
        text = _all_text(zf)
    # C1/C3 use "Claim bridge:"; C2/C4/C5 use "C2 bridge:" style
    for needle in (
        "Claim bridge:",  # C1 and C3
        "C2 bridge:",
        "C5 bridge:",
        "C4 bridge:",
        "Falsifier",
    ):
        assert needle in text, f"missing claim-bridge language: {needle}"
    # A5 risk-coverage must not assert always < never
    assert "always 2.80 < never" not in text
    assert "always-intervene 2.80 and never-intervene 1.71" in text or (
        "1.63" in text and "2.80" in text and "1.71" in text and "and never" in text
    )


@pytest.mark.skipif(not DECK.exists(), reason="group meeting deck not generated")
def test_deck_is_grayscale_only():
    with zipfile.ZipFile(DECK) as zf:
        colors = _all_srgb(zf)
    nongray = sorted(c for c in colors if not _is_gray(c))
    assert nongray == [], f"chromatic colors found: {nongray}"


@pytest.mark.skipif(not DECK.exists(), reason="group meeting deck not generated")
def test_deck_glosses_specialist_terms():
    with zipfile.ZipFile(DECK) as zf:
        text = _all_text(zf)
    # glossary slide + first-use glosses
    required = [
        ("Proprioception", "Body sensing"),
        ("VLA", "Vision"),
        ("Kino-Tokens", "latent"),
        ("C2ST", "two-sample"),
        ("McNemar", "Paired"),
        ("ERS", "Expected recovery"),
        ("Abstention", "Refuse"),
    ]
    for term, gloss_hint in required:
        assert term in text, f"missing term {term}"
        assert gloss_hint in text, f"missing gloss for {term}: expected fragment {gloss_hint!r}"
