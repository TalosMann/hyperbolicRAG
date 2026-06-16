import sys, asyncio
sys.path.insert(0, 'D:/Projects/hyperbolic/Hyper-RAG')
from hyperscholar.core.config import load_config
from hyperscholar.core.embedder import build_embedder
from hyperscholar.core.llm import build_llm_func
from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage

async def check():
    cfg = load_config()
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)
    backend = HierarchicalRAGBackend(
        llm_func=llm, embedder=embedder,
        working_dir=cfg.working_dir,
        kv_cls=JsonKVStorage,
        vector_cls=NanoVectorDBStorage,
        pg_dsn=None, fail_markers=cfg.rag.fail_markers)
    docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = backend._stores('demo')
    cfg2 = backend._cfg('demo')
    print('working_dir:', cfg2['working_dir'])
    chunk_ids = await chunks.all_keys()
    tree_ids = await tree_kv.all_keys()
    print('chunks:', len(chunk_ids))
    print('tree nodes:', len(tree_ids))
    if tree_ids:
        rows = await tree_kv.get_by_ids(tree_ids[:2])
        for tid, row in zip(tree_ids[:2], rows):
            print('  node', tid[:20], 'level=', (row or {}).get('level'), 'children=', len((row or {}).get('children', [])))

asyncio.run(check())
