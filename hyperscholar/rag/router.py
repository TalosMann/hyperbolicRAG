"""RAGRouter — the three-tier query strategy. Backend-agnostic by design:
it only ever calls `backend.query(namespace, text)`, so the routing logic is
identical whichever backend the config selects.

Classroom: institutional DB primary → global fallback flagged out-of-scope.
Personal:  exam / research_paper → personal corpus only;
           textbook / general   → personal + global blended.
"""
from __future__ import annotations

from ..core.types import (
    GLOBAL_NS, CorpusCategory, QueryResult, RouterResult, TenantNS,
)
from .base import RAGBackend

OUT_OF_SCOPE_DISCLAIMER = (
    "This is outside your institution's database — please verify responses."
)

CORPUS_ONLY_CATEGORIES: set = {"exam", "research_paper"}


class RAGRouter:
    def __init__(self, backend: RAGBackend, global_ns: TenantNS = GLOBAL_NS,
                 min_sources: int = 1):
        self.backend = backend
        self.global_ns = global_ns
        self.min_sources = min_sources

    def _sufficient(self, result: QueryResult) -> bool:
        return result.ok

    # ── Classroom mode ─────────────────────────────────────────────────────────
    async def query_classroom(self, text: str, inst_ns: TenantNS,
                              top_k: int = 60) -> RouterResult:
        result = await self.backend.query(inst_ns, text, top_k=top_k)
        if self._sufficient(result):
            return RouterResult(result=result, source="institutional",
                                out_of_scope=False)
        fallback = await self.backend.query(self.global_ns, text, top_k=top_k)
        return RouterResult(result=fallback, source="global", out_of_scope=True,
                            disclaimer=OUT_OF_SCOPE_DISCLAIMER)

    # ── Personal mode ──────────────────────────────────────────────────────────
    async def query_personal(self, text: str, personal_ns: TenantNS,
                             category: CorpusCategory = "general",
                             top_k: int = 60) -> RouterResult:
        if category in CORPUS_ONLY_CATEGORIES:
            result = await self.backend.query(personal_ns, text, top_k=top_k)
            return RouterResult(result=result, source="personal",
                                out_of_scope=not result.ok)
        personal = await self.backend.query(personal_ns, text, top_k=top_k)
        glob = await self.backend.query(self.global_ns, text, top_k=top_k)
        return RouterResult(result=self._merge(personal, glob), source="blended",
                            out_of_scope=False)

    def _merge(self, personal: QueryResult, glob: QueryResult) -> QueryResult:
        """Personal corpus leads; global supplements. If only one side answered,
        return it; if both, concatenate with the personal answer first."""
        if personal.ok and not glob.ok:
            return personal
        if glob.ok and not personal.ok:
            return glob
        if not personal.ok and not glob.ok:
            return personal
        merged = QueryResult(
            answer=(f"{personal.answer}\n\n— Broader context from the global "
                    f"knowledge base —\n{glob.answer}"),
            sources=personal.sources + glob.sources,
            backend=personal.backend,
            namespace=personal.namespace,
            mode=f"{personal.mode}+global",
            ok=True,
        )
        return merged
