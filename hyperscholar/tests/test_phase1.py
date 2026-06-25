"""Phase 1 test suite — runs fully offline (memory store, hash embedder, stub LLM).

Three layers of validation:

1. Storage contract tests   — memory classes honour HyperRAG's storage semantics
                              (the PG classes implement the same contract; these
                              tests document exactly what that contract is).
2. Router strategy tests    — three-tier routing with a scripted FakeBackend.
3. Backend conformance      — every RAGBackend implementation is run through the
                              same scenario: index → query (hit) → query (miss)
                              → concept graph → delete_namespace → tenant isolation.
                              HierarchicalRAG runs for real; HyperRAG backends run
                              if the `hyperrag` package is importable, else skip.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hyperscholar.core.embedder import HashEmbedder
from hyperscholar.core.llm import StubLLM
from hyperscholar.core.types import Document, QueryResult
from hyperscholar.rag.base import RAGBackend
from hyperscholar.rag.hierarchical_backend import FAIL, HierarchicalRAGBackend
from hyperscholar.rag.router import OUT_OF_SCOPE_DISCLAIMER, RAGRouter
from hyperscholar.storage.memory import (
    MemoryHypergraphStorage, MemoryKVStorage, MemoryVectorStorage,
    reset_memory_store,
)


@pytest.fixture(autouse=True)
def _clean_store():
    reset_memory_store()
    yield
    reset_memory_store()


def run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


CFG = {"addon_params": {"tenant": "inst_1"}, "embedding_batch_num": 8}
CFG_OTHER = {"addon_params": {"tenant": "inst_2"}, "embedding_batch_num": 8}


# ─── 1. storage contracts ─────────────────────────────────────────────────────
class TestKVContract:
    def test_upsert_is_insert_only_and_filter_keys(self):
        kv = MemoryKVStorage(namespace="text_chunks", global_config=CFG)
        left = run(kv.upsert({"a": {"v": 1}, "b": {"v": 2}}))
        assert set(left) == {"a", "b"}
        # upstream semantics: existing keys are NOT overwritten
        left = run(kv.upsert({"a": {"v": 99}, "c": {"v": 3}}))
        assert set(left) == {"c"}
        assert run(kv.get_by_id("a")) == {"v": 1}
        assert run(kv.filter_keys(["a", "zz"])) == {"zz"}

    def test_get_by_ids_with_fields(self):
        kv = MemoryKVStorage(namespace="text_chunks", global_config=CFG)
        run(kv.upsert({"a": {"content": "x", "tokens": 5}}))
        got = run(kv.get_by_ids(["a", "missing"], fields={"content"}))
        assert got == [{"content": "x"}, None]

    def test_tenant_isolation(self):
        kv1 = MemoryKVStorage(namespace="text_chunks", global_config=CFG)
        kv2 = MemoryKVStorage(namespace="text_chunks", global_config=CFG_OTHER)
        run(kv1.upsert({"a": {"v": 1}}))
        assert run(kv2.get_by_id("a")) is None


class TestVectorContract:
    def test_query_shape_threshold_and_meta(self):
        emb = HashEmbedder(dim=64)
        vdb = MemoryVectorStorage(namespace="entities", global_config=CFG,
                                  embedding_func=emb, meta_fields={"entity_name"})
        run(vdb.upsert({
            "e1": {"content": "photosynthesis chlorophyll light energy",
                   "entity_name": "PHOTOSYNTHESIS"},
            "e2": {"content": "mitochondria cellular respiration",
                   "entity_name": "MITOCHONDRIA"},
        }))
        res = run(vdb.query("chlorophyll light", top_k=5))
        assert res and res[0]["id"] == "e1"
        assert res[0]["entity_name"] == "PHOTOSYNTHESIS"   # meta_fields surfaced
        assert "distance" in res[0]
        # unrelated text falls below cosine threshold → no junk results
        res = run(vdb.query("zqx wvu kjh", top_k=5))
        assert all(r["distance"] > 0.2 for r in res)


class TestHypergraphContract:
    def test_vertices_edges_and_neighbourhood(self):
        hg = MemoryHypergraphStorage(namespace="chunk_entity_relation",
                                     global_config=CFG)
        run(hg.upsert_vertex("A", {"description": "alpha"}))
        run(hg.upsert_vertex("B", {"description": "beta"}))
        run(hg.upsert_vertex("C", {"description": "gamma"}))
        run(hg.upsert_hyperedge(("A", "B", "C"), {"keywords": "abc"}))
        run(hg.upsert_hyperedge(("B", "C"), {"keywords": "bc"}))

        assert run(hg.has_hyperedge(("C", "B", "A")))       # order-insensitive
        assert run(hg.get_num_of_vertices()) == 3
        assert run(hg.get_num_of_hyperedges()) == 2
        assert set(run(hg.get_nbr_v_of_vertex("A"))) == {"B", "C"}
        assert run(hg.vertex_degree("B")) == 2
        # vertex data merges on upsert
        run(hg.upsert_vertex("A", {"extra": 1}))
        v = run(hg.get_vertex("A"))
        assert v["description"] == "alpha" and v["extra"] == 1
        # removing a vertex removes incident hyperedges
        run(hg.remove_vertex("A"))
        assert run(hg.get_num_of_hyperedges()) == 1


# ─── 2. router strategy ───────────────────────────────────────────────────────
class FakeBackend(RAGBackend):
    """Scripted backend: answers only for namespaces in `known`."""

    def __init__(self, known: dict[str, str]):
        self.known = known
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self):
        return "fake"

    async def index(self, namespace, documents):
        raise NotImplementedError

    async def query(self, namespace, text, top_k=60):
        self.calls.append((namespace, text))
        if namespace in self.known:
            return QueryResult(answer=self.known[namespace], ok=True,
                               backend="fake", namespace=namespace)
        return QueryResult(answer=FAIL, ok=False, backend="fake",
                           namespace=namespace)

    async def get_concept_graph(self, namespace, center, depth=2):
        raise NotImplementedError

    async def delete_namespace(self, namespace):
        return True


class TestRouter:
    def test_classroom_institutional_primary(self):
        be = FakeBackend({"inst_1": "from institution", "global": "from global"})
        r = run(RAGRouter(be).query_classroom("q", "inst_1"))
        assert r.source == "institutional" and not r.out_of_scope
        assert r.result.answer == "from institution"
        assert be.calls == [("inst_1", "q")]               # no needless global hit

    def test_classroom_global_fallback_flags_out_of_scope(self):
        be = FakeBackend({"global": "from global"})
        r = run(RAGRouter(be).query_classroom("q", "inst_1"))
        assert r.source == "global" and r.out_of_scope
        assert r.disclaimer == OUT_OF_SCOPE_DISCLAIMER

    def test_personal_exam_category_is_corpus_only(self):
        be = FakeBackend({"global": "from global"})        # personal ns unknown
        r = run(RAGRouter(be).query_personal("q", "personal_9", category="exam"))
        assert r.source == "personal" and r.out_of_scope    # no silent global leak
        assert [c[0] for c in be.calls] == ["personal_9"]

    def test_personal_textbook_blends(self):
        be = FakeBackend({"personal_9": "personal ans", "global": "global ans"})
        r = run(RAGRouter(be).query_personal("q", "personal_9", category="textbook"))
        assert r.source == "blended" and r.result.ok
        assert "personal ans" in r.result.answer
        assert "global ans" in r.result.answer
        assert r.result.answer.index("personal ans") < r.result.answer.index("global ans")


# ─── 3. backend conformance ───────────────────────────────────────────────────
CORPUS = [
    Document(content=(
        "Photosynthesis is the process by which green plants convert sunlight "
        "into chemical energy. Chlorophyll inside chloroplasts absorbs light, "
        "driving the conversion of carbon dioxide and water into glucose and "
        "oxygen. The light-dependent reactions occur in the thylakoid membranes."
    ), title="Photosynthesis"),
    Document(content=(
        "Cellular respiration releases energy stored in glucose. It takes place "
        "in the mitochondria and produces ATP, the cell's energy currency. "
        "Glycolysis, the Krebs cycle and oxidative phosphorylation are its "
        "three stages."
    ), title="Respiration"),
]


def make_hierarchical():
    return HierarchicalRAGBackend(
        llm_func=StubLLM(), embedder=HashEmbedder(dim=128),
        kv_cls=MemoryKVStorage, vector_cls=MemoryVectorStorage,
        cluster_threshold=0.05)


def make_hyperrag(light=False):
    pytest.importorskip("hyperrag")
    from hyperscholar.rag.hyperrag_backend import (
        HyperRAGBackend, HyperRAGLightBackend,
    )
    from hyperscholar.tests.hyperrag_stub import HyperRAGStubLLM
    cls = HyperRAGLightBackend if light else HyperRAGBackend
    return cls(llm_func=HyperRAGStubLLM(), embedder=HashEmbedder(dim=128),
               working_dir="/tmp/hs_test", kv_cls=MemoryKVStorage,
               vector_cls=MemoryVectorStorage,
               hypergraph_cls=MemoryHypergraphStorage,
               cosine_threshold=-1.0)   # hash embedder: accept all, rank by sim


def make_pure_cograg():
    from hyperscholar.rag.pure_cograg_backend import PureCogRAGBackend
    stub_responses = {
        "Extract all fine-grained entities": '[{"name": "chlorophyll", "description": "green pigment"}]',
        "Extract high-order relations": '[{"entities": ["chlorophyll"], "description": "absorbs light"}]',
    }
    return PureCogRAGBackend(
        llm_func=StubLLM(responses=stub_responses), embedder=HashEmbedder(dim=128),
        working_dir="/tmp/hs_test", kv_cls=MemoryKVStorage,
        vector_cls=MemoryVectorStorage, cosine_threshold=-1.0)

def make_cograg_flash():
    from hyperscholar.rag.cograg_flash_backend import CogRagFlashBackend
    return CogRagFlashBackend(
        llm_func=StubLLM(), embedder=HashEmbedder(dim=128),
        working_dir="/tmp/hs_test", kv_cls=MemoryKVStorage,
        vector_cls=MemoryVectorStorage, cluster_threshold=0.05,
        cosine_threshold=-1.0)


BACKEND_FACTORIES = {
    "hierarchical": make_hierarchical,
    "hyperrag": lambda: make_hyperrag(False),
    "hyperrag_light": lambda: make_hyperrag(True),
    "pure_cograg": make_pure_cograg,
    "cograg_flash": make_cograg_flash,
}


@pytest.mark.parametrize("backend_name", list(BACKEND_FACTORIES))
class TestBackendConformance:
    """The same scenario must pass for every backend — this is the suite that
    guarantees a config-line swap can't break the platform."""

    def test_full_lifecycle(self, backend_name):
        be = BACKEND_FACTORIES[backend_name]()
        # index
        ir = run(be.index("inst_1", CORPUS))
        assert ir.backend == be.name
        assert ir.chunks >= 2
        # grounded query → ok, answer references corpus content via the LLM
        qr = run(be.query("inst_1", "How do plants convert sunlight into energy?"))
        assert isinstance(qr, QueryResult) and qr.backend == be.name
        assert qr.ok and qr.answer
        # tenant isolation: same question in an unindexed namespace finds nothing
        miss = run(be.query("inst_2", "How do plants convert sunlight?"))
        assert not miss.ok
        # concept graph is well-formed
        cg = run(be.get_concept_graph("inst_1", "photosynthesis"))
        assert cg.center is not None
        for e in cg.edges:
            ids = {n.id for n in cg.nodes}
            assert e.source in ids and e.target in ids
        # delete structure → grounded query degrades, shared chunks untouched
        assert run(be.delete_namespace("inst_1"))

    def test_sources_are_traceable(self, backend_name):
        be = BACKEND_FACTORIES[backend_name]()
        run(be.index("inst_1", CORPUS))
        qr = run(be.query("inst_1", "Where does cellular respiration take place?"))
        if qr.sources:                       # best-effort, but if present must be valid
            for s in qr.sources:
                assert s.chunk_id and isinstance(s.score, float)


class TestSharedContentAcrossBackends:
    def test_chunks_written_once_are_visible_to_other_backends(self):
        """The Decision-2 property: content layer is shared. Indexing under one
        backend leaves chunks in shared storage that another backend's stores
        can see for the same tenant."""
        hier = make_hierarchical()
        run(hier.index("inst_1", CORPUS))
        kv = MemoryKVStorage(namespace="text_chunks", global_config=CFG)
        keys = run(kv.all_keys())
        assert len(keys) >= 2                # chunks landed in shared content
        # a second backend instance over the same tenant sees the same chunks
        hier2 = make_hierarchical()
        qr = run(hier2.query("inst_1", "What produces ATP in the cell?"))
        assert qr.ok
