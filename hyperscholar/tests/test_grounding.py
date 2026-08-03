"""Conformance tests for the grounding layer's deterministic core.

No LLM / no network — these exercise the hard-guarantee path (specific-fact
extraction + grounding check + decision logic), which is where the layer makes
non-probabilistic promises. The LLM-verifier and end-to-end behavior are
measured separately by the labeled-eval harness (needs an API key).
"""
from hyperscholar.grounding import (
    build_source_index, decide, extract_specifics, flagged_specifics,
    specific_grounded,
)

SOURCE = (
    "Marley was dead, to begin with. This edition was published in 1843. "
    "The bell tolled one. 'Mankind was my business,' cried the ghost, "
    "wringing its hands. Project Gutenberg is a registered trademark."
)
IDX = build_source_index(SOURCE)


# ── extraction ────────────────────────────────────────────────────────────────

def test_extract_catches_year_number_date_quote():
    ans = ('Registered in 2000 under number 2,399,140 on June 6, 2000. '
           'He said "the surplus population must decrease".')
    kinds = {k for k, _ in extract_specifics(ans)}
    assert "year" in kinds
    assert "number" in kinds
    assert "date" in kinds
    assert "quote" in kinds


def test_possessive_apostrophe_is_not_a_quote():
    # The apostrophe bug: possessives/contractions must NOT register as quotes.
    ans = "Scrooge's nephew visited the ghost's chamber; it didn't matter."
    assert not [v for k, v in extract_specifics(ans) if k == "quote"]


# ── grounding check ───────────────────────────────────────────────────────────

def test_year_present_vs_absent():
    assert specific_grounded("year", "1843", IDX) is True
    assert specific_grounded("year", "2000", IDX) is False


def test_number_absent_is_flagged():
    assert specific_grounded("number", "2,399,140", IDX) is False


def test_short_quote_substring_match():
    # < 6 words -> substring path
    assert specific_grounded("quote", "Mankind was my business", IDX) is True
    assert specific_grounded("quote", "the surplus population", IDX) is False


def test_long_quote_sixgram_match():
    # >= 6 words -> 6-gram overlap path
    assert specific_grounded("quote", "Mankind was my business cried the ghost", IDX) is True
    assert specific_grounded("quote", "the ghost of christmas past appeared before him", IDX) is False


def test_date_absent_is_flagged():
    assert specific_grounded("date", "june 6", IDX) is False


def test_flagged_specifics_only_returns_absent():
    text = 'Published in 1843, registered in 2000 under 2,399,140.'
    flagged = {v for _, v in flagged_specifics(text, IDX)}
    assert "1843" not in flagged        # present -> not flagged
    assert "2000" in flagged            # absent  -> flagged
    assert "2,399,140" in flagged


# ── decision logic ────────────────────────────────────────────────────────────

def test_decide_serve_when_grounded():
    v = {"supported": ["Marley was dead"], "unsupported_fact": [], "unsupported_interp": []}
    assert decide(v, IDX)["decision"] == "SERVE"


def test_decide_decline_when_nothing_survives():
    v = {"supported": [], "unsupported_fact": ["registered in 2000 under 2,399,140"],
         "unsupported_interp": []}
    assert decide(v, IDX)["decision"] == "DECLINE"


def test_decide_interp_only():
    v = {"supported": [], "unsupported_fact": [],
         "unsupported_interp": ["This suggests a theme of mortality"]}
    assert decide(v, IDX)["decision"] == "SERVE_INTERP_ONLY"


def test_decide_demotes_supported_claim_with_fabricated_specific():
    # LLM wrongly marks a fabricated-specific claim 'supported'; matcher demotes it.
    v = {"supported": ["The trademark was registered in 2000"],
         "unsupported_fact": [], "unsupported_interp": []}
    d = decide(v, IDX)
    assert d["decision"] == "DECLINE"
    assert d["demoted"] == ["The trademark was registered in 2000"]
