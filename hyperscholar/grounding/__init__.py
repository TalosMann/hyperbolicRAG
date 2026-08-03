"""Grounding layer — strategy-agnostic hallucination prevention.

Sits above any RAGBackend. Depends on the backend for one thing only: retrieving
raw source chunks for a query. Grounds an answer against those raw chunks via a
deterministic specific-fact checker (hard guarantee on dates/numbers/quotations)
plus a calibrated LLM verifier, and decides serve / quarantine / decline.

See GROUNDING_LAYER_REPORT.md (repo root) for design and cross-strategy rationale.
"""
from .verifier import (
    ANCHORED_GEN_PROMPT,
    VERIFY_PROMPT,
    build_source_index,
    decide,
    extract_specifics,
    flagged_specifics,
    grounded_answer,
    llm_verify,
    retrieve_raw_chunks,
    specific_grounded,
)

__all__ = [
    "ANCHORED_GEN_PROMPT", "VERIFY_PROMPT", "build_source_index", "decide",
    "extract_specifics", "flagged_specifics", "grounded_answer", "llm_verify",
    "retrieve_raw_chunks", "specific_grounded",
]
