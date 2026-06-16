"""Core datatypes shared across HyperScholar's retrieval layer.

These are deliberately small, dependency-free dataclasses so every module
(storage, backends, router, API) can import them without pulling in heavy deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# ─── Tenancy ──────────────────────────────────────────────────────────────────
# A tenant namespace partitions every table in every schema.
#   global            → curated textbook baseline
#   inst_{id}         → institutional DB
#   personal_{uid}    → personal DB
TenantNS = str

GLOBAL_NS: TenantNS = "global"


def inst_ns(institution_id: str | int) -> TenantNS:
    return f"inst_{institution_id}"


def personal_ns(user_id: str | int) -> TenantNS:
    return f"personal_{user_id}"


# ─── Documents / corpus ──────────────────────────────────────────────────────
CorpusCategory = Literal["exam", "research_paper", "textbook", "general"]


@dataclass
class Document:
    content: str
    title: str = ""
    category: CorpusCategory = "general"
    meta: dict = field(default_factory=dict)


# ─── Retrieval results ───────────────────────────────────────────────────────
@dataclass
class SourceRef:
    """Traceability: where an answer came from."""
    chunk_id: str
    doc_id: str = ""
    score: float = 0.0
    excerpt: str = ""


@dataclass
class QueryResult:
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    backend: str = ""
    namespace: TenantNS = ""
    mode: str = ""
    ok: bool = True              # False → backend judged it had no grounded answer
    raw: Any = None


@dataclass
class IndexResult:
    namespace: TenantNS
    documents: int
    chunks: int
    backend: str
    detail: dict = field(default_factory=dict)


# ─── Concept graph (Poincaré sphere feed) ────────────────────────────────────
@dataclass
class ConceptNode:
    id: str
    label: str
    summary: str = ""
    level: int = 0               # 0 = center; +n broader; -n more specific (sphere latitude)
    is_center: bool = False


@dataclass
class ConceptEdge:
    source: str
    target: str
    label: str = ""


@dataclass
class ConceptGraph:
    center: str
    nodes: list[ConceptNode] = field(default_factory=list)
    edges: list[ConceptEdge] = field(default_factory=list)


# ─── Router output ───────────────────────────────────────────────────────────
RouterSource = Literal["institutional", "global", "personal", "blended"]


@dataclass
class RouterResult:
    result: QueryResult
    source: RouterSource
    out_of_scope: bool = False
    # UI renders when out_of_scope:
    # "This is outside your institution's database — please verify."
    disclaimer: Optional[str] = None
