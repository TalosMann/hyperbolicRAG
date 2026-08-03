"""Conformance tests for cograg_safe — the zero-hallucination pipeline.

No LLM / no network. Exercises the deterministic core (alias forms, entity
binding, Gate R1 co-mention, assertion parsing, compose/decline logic) plus a
full end-to-end run of safe_answer() with a stub LLM, reproducing the canonical
"Fan is Tiny Tim's sister" failure and proving it is caught.
"""
import asyncio

import pytest

from hyperscholar.cograg_safe.assertions import classify_intent, parse_assertions
from hyperscholar.cograg_safe.evidence import EvidencePack, surface_forms
from hyperscholar.cograg_safe.pipeline import (
    bind_question_entities, compose, compose_synthesis, safe_answer, _direct_link,
)

# ── synthetic corpus mirroring the christmas_carol structure ─────────────────

CHARLIST = ("CHARACTERS. Bob Cratchit, clerk to Ebenezer Scrooge. "
            "Tim Cratchit (Tiny Tim), youngest son of Bob Cratchit. "
            "Fan, the sister of Scrooge. Published in 1843.")
FAN_SCENE = ("A little girl came in. It is Fan, who says: I have come to bring "
             "you home, dear brother, said the child. Home, for good, said Fan. "
             "Master Scrooge wept.")
TIM_SCENE = ("Bob Cratchit carried Tiny Tim upon his shoulder. Tiny Tim, "
             "Bob's youngest son, bore a little crutch.")
BELLE_SCENE = "Belle sat opposite her daughter. Belle had once loved a man."

CHUNKS = {"c-list": CHARLIST, "c-fan": FAN_SCENE, "c-tim": TIM_SCENE,
          "c-belle": BELLE_SCENE}
VERTICES = {
    "FAN": {"source_ids": {"c-list", "c-fan"}, "description": "Sister of Scrooge."},
    "TINY TIM": {"source_ids": {"c-list", "c-tim"}, "description": "Youngest son of Bob Cratchit."},
    "TIM CRATCHIT": {"source_ids": {"c-list"}, "description": "Alias of Tiny Tim."},
    "TIM": {"source_ids": {"c-tim"}, "description": "Short alias."},
    "SCROOGE": {"source_ids": {"c-fan"}, "description": "The protagonist."},
    "EBENEZER SCROOGE": {"source_ids": {"c-list"}, "description": "Full name."},
    "BOB CRATCHIT": {"source_ids": {"c-list", "c-tim"}, "description": "Scrooge's clerk."},
    "BELLE": {"source_ids": {"c-belle"}, "description": "Scrooge's former love."},
}
EDGES = [
    {"id": "rel-0", "entities": ["TINY TIM", "BOB CRATCHIT"],
     "description": "Tiny Tim is Bob Cratchit's youngest son.", "source_ids": {"c-tim"}},
    {"id": "rel-1", "entities": ["BOB CRATCHIT", "EBENEZER SCROOGE"],
     "description": "Bob Cratchit is clerk to Ebenezer Scrooge.", "source_ids": {"c-list"}},
]


@pytest.fixture()
def pack():
    return EvidencePack(dict(CHUNKS), {k: dict(v) for k, v in VERTICES.items()},
                        [dict(e) for e in EDGES])


# ── alias surface forms + binding ────────────────────────────────────────────

def test_surface_forms_alias_expansion():
    names = list(VERTICES)
    f = surface_forms("Scrooge", names)
    assert "scrooge" in f and "ebenezer scrooge" in f          # subset expansion
    f2 = surface_forms("Tiny Tim", names)
    assert "tiny tim" in f2 and "tim" in f2                    # TIM token-subset
    assert "bob cratchit" not in f2                            # no over-merge


def test_bind_question_entities_longest_match(pack):
    m = bind_question_entities(pack, "How is Fan related to Tiny Tim?")
    assert "FAN" in m and "TINY TIM" in m
    assert "TIM" not in m                                      # suppressed by TINY TIM


def test_binding_is_corpus_scoped(pack):
    # An entity not in the corpus binds nothing (parametric namesakes excluded)
    assert bind_question_entities(pack, "Who is Herbert Khaury?") == []


# ── Gate R1: deterministic co-mention ────────────────────────────────────────

def test_r1_comention_via_chunk_text(pack):
    # Fan and Tiny Tim co-occur ONLY in the character list chunk
    hits = pack.co_mention("Fan", "Tiny Tim")
    assert "c-list" in hits


def test_r1_rejects_never_cooccurring_pair(pack):
    # Belle and Tiny Tim share no chunk, no edge, no source_id -> R1 reject
    assert pack.co_mention("Belle", "Tiny Tim") == []


def test_r1_via_shared_hyperedge(pack):
    hits = pack.co_mention("Tiny Tim", "Bob Cratchit")
    assert any(h.startswith("rel-") for h in hits) or "c-tim" in hits


def test_r1_alias_robust(pack):
    # SCROOGE + BOB CRATCHIT co-occur only via EBENEZER SCROOGE alias (c-list, rel-1)
    hits = pack.co_mention("Scrooge", "Bob Cratchit")
    assert hits, "alias expansion must bridge SCROOGE ~ EBENEZER SCROOGE"


# ── assertion parsing ────────────────────────────────────────────────────────

def test_parse_assertions_valid_and_fenced():
    j = ('```json\n{"relations": [{"a": "FAN", "relation": "sister of", '
         '"b": "SCROOGE", "evidence_quote": "Fan, the sister of Scrooge."}], '
         '"facts": [{"statement": "Tiny Tim bore a crutch.", '
         '"evidence_quote": "bore a little crutch"}]}\n```')
    p = parse_assertions(j)
    assert len(p["relations"]) == 1 and len(p["facts"]) == 1


def test_parse_assertions_malformed_dropped():
    assert parse_assertions("not json at all") == {"relations": [], "facts": []}
    p = parse_assertions('{"relations": [{"a": "X"}], "facts": [{}]}')
    assert p["relations"] == [] and p["facts"] == []


# ── compose / decline logic ──────────────────────────────────────────────────

def test_compose_declines_when_nothing_verified(pack):
    out = compose(pack, "How?", ["FAN", "TINY TIM"],
                  {"verified_relations": [], "verified_facts": []})
    assert out["decision"] == "DECLINE"


def test_compose_adds_relational_decline_note(pack):
    rels = [{"a": "FAN", "relation": "sister of", "b": "SCROOGE",
             "provenance": ["c-list"]},
            {"a": "TINY TIM", "relation": "son of", "b": "BOB CRATCHIT",
             "provenance": ["rel-0"]}]
    out = compose(pack, "How is Fan related to Tiny Tim?", ["FAN", "TINY TIM"],
                  {"verified_relations": rels, "verified_facts": []})
    assert out["decision"] == "SERVE"
    assert "does not state a direct relationship between FAN and TINY TIM" in out["answer"]
    assert "c-list" in out["answer"]                            # provenance carried


def test_direct_link_respects_aliases(pack):
    rels = [{"a": "Ebenezer Scrooge", "relation": "employer of", "b": "Bob Cratchit"}]
    assert _direct_link(pack, "Scrooge", "Bob Cratchit", rels)  # via alias forms


# ── end-to-end: the canonical §9 case, zero API ──────────────────────────────

BAD_AND_GOOD_ASSERTIONS = """{
  "relations": [
    {"a": "FAN", "relation": "sister of", "b": "SCROOGE",
     "evidence_quote": "Fan, the sister of Scrooge."},
    {"a": "FAN", "relation": "sister of", "b": "TINY TIM",
     "evidence_quote": "Fan, the sister of Scrooge."},
    {"a": "BELLE", "relation": "aunt of", "b": "TINY TIM",
     "evidence_quote": "Belle sat opposite her daughter."}
  ],
  "facts": [
    {"statement": "Tiny Tim bore a little crutch.",
     "evidence_quote": "Tiny Tim, Bob's youngest son, bore a little crutch."},
    {"statement": "Tiny Tim won a Grammy Award in 1968.",
     "evidence_quote": "this quote does not exist in the source"},
    {"statement": "Fan is a character in young Scrooge's books.",
     "evidence_quote": "Fan, the sister of Scrooge."}
  ]
}"""


async def _stub_llm(prompt: str, **kw) -> str:
    if "CLAIMED RELATIONSHIP" in prompt:
        # R2 verdict: the char-list states Fan is SCROOGE's sister — so the
        # FAN-SCROOGE triple is entailed; FAN-TINY TIM 'sister' is NOT.
        if "FAN — sister of — SCROOGE" in prompt:
            return "YES"
        return "NO"
    if "CLAIMED STATEMENT" in prompt:
        # F2 verdict: the crutch fact is entailed by its chunk; the
        # 'Fan is a character in young Scrooge's books' mis-statement is NOT
        # (its quote is real, but the chunk says something different).
        return "YES" if "crutch" in prompt else "NO"
    return BAD_AND_GOOD_ASSERTIONS


def test_end_to_end_canonical_case(pack):
    out = asyncio.run(safe_answer(pack, None, _stub_llm,
                                  "How is Fan related to Tiny Tim?"))
    assert out["decision"] == "SERVE"
    answer = out["answer"]

    # the TRUE relation survives, with provenance
    assert "FAN — sister of — SCROOGE" in answer
    # the FALSE relational conclusion is rejected (R2: co-mention exists but
    # the evidence states a different relation)
    r2_rejected = [r for r in out["rejected_relations"] if r["gate"] == "R2"]
    assert any(r["b"] == "TINY TIM" and r["a"] == "FAN" for r in r2_rejected)
    assert "FAN — sister of — TINY TIM" not in answer
    # the never-co-occurring pair dies at R1 with zero LLM involvement
    r1_rejected = [r for r in out["rejected_relations"] if r["gate"] == "R1"]
    assert any(r["a"] == "BELLE" for r in r1_rejected)
    # fabricated fact (fake quote + ungrounded year/award) is dropped
    assert any("Grammy" in f["statement"] for f in out["dropped_facts"])
    assert "Grammy" not in answer and "1968" not in answer
    # facts-entailment gate (F2): real quote but wrong statement is dropped
    f2 = [f for f in out["dropped_facts"] if f.get("gate") == "F2"]
    assert any("young Scrooge's books" in f["statement"] for f in f2)
    assert "young Scrooge's books" not in answer
    # true fact survives all gates
    assert any("crutch" in f["statement"] for f in out["verified_facts"])
    # relational decline is explicit
    assert "does not state a direct relationship between FAN and TINY TIM" in answer


# ── SYNTHESIS mode: comparative questions reaching outside the corpus ────────

def _intent_stub(reply: str):
    async def _llm(prompt: str, **kw) -> str:
        return reply
    return _llm


def test_classify_intent_corpus():
    llm = _intent_stub("CORPUS")
    assert asyncio.run(classify_intent(llm, "How is Fan related to Tiny Tim?")) == "CORPUS"


def test_classify_intent_synthesis():
    llm = _intent_stub("SYNTHESIS")
    assert asyncio.run(classify_intent(
        llm, "How does this mirror Heathcliff and Hindley?")) == "SYNTHESIS"


def test_classify_intent_defaults_to_corpus_on_malformed_reply():
    # malformed/ambiguous classifier output -> the SAFE default, not a guess
    llm = _intent_stub("uh, I'm not sure, maybe both?")
    assert asyncio.run(classify_intent(llm, "anything")) == "CORPUS"


def test_compose_synthesis_has_two_labeled_sections(pack):
    v = {"verified_relations": [{"a": "FAN", "relation": "sister of", "b": "SCROOGE",
                                "provenance": ["c-list"]}],
         "verified_facts": []}
    out = compose_synthesis(pack, "q", ["FAN", "TINY TIM"], v,
                            "Heathcliff and Hindley's rivalry in Wuthering Heights...")
    answer = out["answer"]
    assert "From this text:" in answer
    assert "FAN — sister of — SCROOGE" in answer
    assert "General literary knowledge (not from this corpus" in answer
    assert "Heathcliff and Hindley" in answer
    # the general-knowledge section must come strictly after the corpus section
    assert answer.index("From this text:") < answer.index("General literary knowledge")


SYNTHESIS_ASSERTIONS = """{
  "relations": [
    {"a": "FAN", "relation": "sister of", "b": "SCROOGE",
     "evidence_quote": "Fan, the sister of Scrooge."}
  ],
  "facts": []
}"""

HEATHCLIFF_ANSWER = ("Fan and Tiny Tim's dynamics with Scrooge echo the sibling/"
                     "quasi-familial rivalries between Heathcliff and Hindley in "
                     "Wuthering Heights: both pairs involve a childhood bond "
                     "curdled or tested by adult resentment and inheritance.")


async def _synthesis_stub_llm(prompt: str, **kw) -> str:
    if "CORPUS or SYNTHESIS" in prompt:
        return "SYNTHESIS"
    if "CLAIMED RELATIONSHIP" in prompt:
        return "YES"  # FAN sister-of SCROOGE is genuinely stated
    if "CLAIMED STATEMENT" in prompt:
        return "YES"
    if "STUDENT'S FULL QUESTION" in prompt:
        return HEATHCLIFF_ANSWER
    return SYNTHESIS_ASSERTIONS


def test_end_to_end_synthesis_mode_labels_both_tracks(pack):
    out = asyncio.run(safe_answer(
        pack, None, _synthesis_stub_llm,
        "How does the Fan/Tiny Tim relationship mirror Heathcliff and Hindley's "
        "in Wuthering Heights?"))
    assert out["intent"] == "SYNTHESIS"
    assert out["decision"] == "SERVE"
    answer = out["answer"]

    # corpus track: real, verified, cited -- exactly as strict mode would produce
    assert "From this text:" in answer
    assert "FAN — sister of — SCROOGE" in answer

    # general-knowledge track: present, clearly labeled, AFTER the corpus track
    assert "General literary knowledge (not from this corpus" in answer
    assert "Heathcliff" in answer and "Hindley" in answer
    assert answer.index("From this text:") < answer.index("General literary knowledge")

    # the external comparison was NEVER submitted to R1/R2 as a corpus claim --
    # it appears in neither verified_relations nor rejected_relations
    all_rel_names = {n for r in out["verified_relations"] + out["rejected_relations"]
                     for n in (r["a"], r["b"])}
    assert "HEATHCLIFF" not in all_rel_names and "HINDLEY" not in all_rel_names


async def _misclassifying_stub_llm(prompt: str, **kw) -> str:
    """A (deliberately wrong) classifier that ALWAYS says SYNTHESIS -- models the
    live failure observed on the free-tier model, where the canonical corpus-only
    question 'How is Fan related to Tiny Tim?' was misrouted. Proves the
    deterministic _has_external_reference override catches it regardless."""
    if "CORPUS or SYNTHESIS" in prompt:
        return "SYNTHESIS"
    if "CLAIMED RELATIONSHIP" in prompt:
        return "YES" if "FAN — sister of — SCROOGE" in prompt else "NO"
    if "STUDENT'S FULL QUESTION" in prompt:
        pytest.fail("general-knowledge generation must not run once the "
                   "deterministic override forces CORPUS mode")
    return BAD_AND_GOOD_ASSERTIONS


def test_misclassified_synthesis_is_overridden_back_to_corpus(pack):
    # question names ONLY corpus-bound entities (Fan, Tiny Tim) -- no external
    # reference -- so even though the stub classifier wrongly says SYNTHESIS,
    # the deterministic guard must force it back to CORPUS.
    out = asyncio.run(safe_answer(pack, None, _misclassifying_stub_llm,
                                  "How is Fan related to Tiny Tim?"))
    assert out["intent"] == "CORPUS"
    assert "General literary knowledge" not in out["answer"]
    assert "Based only on the source material:" in out["answer"]


def test_has_external_reference_detects_unbound_proper_nouns(pack):
    from hyperscholar.cograg_safe.pipeline import _has_external_reference
    assert _has_external_reference(
        pack, "How does this mirror Heathcliff and Hindley in Wuthering Heights?",
        ["FAN", "TINY TIM"]) is True
    assert _has_external_reference(
        pack, "How is Fan related to Tiny Tim?", ["FAN", "TINY TIM"]) is False
