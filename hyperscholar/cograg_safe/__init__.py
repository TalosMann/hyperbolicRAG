"""cograg_safe — zero-hallucination Cog-RAG variant.

Design: "structure for retrieval, raw text for grounding, triples for reasoning."
Verify-then-compose: the LLM proposes relation assertions as structured triples,
each triple must pass a two-gate relational verifier (R1 deterministic co-mention,
R2 scoped entailment) before the final prose is composed FROM the verified set —
so the answer cannot contain a relationship that was not independently grounded.

See COGRAG_ZERO_HALLUCINATION_BRIEF.md (repo root) for the full problem statement.
"""
from .evidence import EvidencePack, surface_forms
from .pipeline import safe_answer

__all__ = ["EvidencePack", "surface_forms", "safe_answer"]
