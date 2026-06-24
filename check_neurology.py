import asyncio
from hyperscholar.core.config import load_config
from hyperscholar.core.embedder import build_embedder
from hyperscholar.core.llm import build_llm_func
from hyperscholar.rag.hyperrag_backend import HyperRAGBackend
from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage, HypergraphStorage

async def check():
    cfg = load_config()
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)
    backend = HyperRAGBackend(
        llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
        kv_cls=JsonKVStorage, vector_cls=NanoVectorDBStorage,
        hypergraph_cls=HypergraphStorage, pg_dsn=None,
        fail_markers=cfg.rag.fail_markers)
    rag = backend._rag('neurology')
    chunk_ids = await rag.text_chunks.all_keys()
    vertices = await rag.chunk_entity_relation_hypergraph.get_all_vertices()
    edges = await rag.chunk_entity_relation_hypergraph.get_all_hyperedges()
    print('chunks:', len(chunk_ids))
    print('vertices:', len(vertices))
    print('hyperedges:', len(edges))

asyncio.run(check())
