"""
HyperScholar Phase 1 — Run all three RAG pipelines and compare results.

Usage (from the hyperscholar/ directory):
    python run_demo.py                          # uses built-in demo corpus
    python run_demo.py --file notes.txt         # index your own text file
    python run_demo.py --query "your question"  # custom query

Modes:
    --offline   hash embedder + stub LLM (no internet, no API key needed)
    --live      bge-m3 + DeepSeek (needs DEEPSEEK_API_KEY env var)

Default is --offline so you can verify the pipelines immediately.
"""
import argparse
import asyncio
import os
import sys
import time

# ── make sure we can find the hyperscholar package ────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── also add the Hyper-RAG checkout if it's in the usual location ─────────────
HYPERRAG_PATH = os.environ.get(
    "HYPERRAG_PATH",
    r"C:\Users\talosmann\Projects\moonlabs\Hyper-RAG",   # ← adjust if different
)
if os.path.isdir(HYPERRAG_PATH) and HYPERRAG_PATH not in sys.path:
    sys.path.insert(0, HYPERRAG_PATH)

# ── Windows asyncio: ProactorEventLoop (the default) works fine on 3.8+
# No policy override needed; WindowsSelectorEventLoopPolicy is deprecated in 3.14.


# ── built-in demo corpus ──────────────────────────────────────────────────────
DEMO_CORPUS = [
    {
        "title": "Photosynthesis",
        "content": (
            "Photosynthesis is the biological process by which green plants, algae, "
            "and some bacteria convert light energy into chemical energy stored as glucose. "
            "This process occurs primarily in the chloroplasts, where the pigment chlorophyll "
            "captures sunlight. The light-dependent reactions take place in the thylakoid "
            "membranes, producing ATP and NADPH. The light-independent reactions (Calvin cycle) "
            "occur in the stroma, using CO2 and the products of the light reactions to synthesise "
            "glucose. Overall: 6CO2 + 6H2O + light → C6H12O6 + 6O2."
        ),
    },
    {
        "title": "Cellular Respiration",
        "content": (
            "Cellular respiration is the process by which cells break down glucose to release "
            "energy in the form of ATP. It occurs in three main stages: glycolysis (in the "
            "cytoplasm, producing 2 ATP), the Krebs cycle (in the mitochondrial matrix, "
            "producing 2 ATP and electron carriers), and oxidative phosphorylation (in the "
            "inner mitochondrial membrane, producing ~32 ATP via the electron transport chain "
            "and ATP synthase). Aerobic respiration requires oxygen; the final electron "
            "acceptor is O2, producing H2O as a by-product. Overall: C6H12O6 + 6O2 → "
            "6CO2 + 6H2O + ~38 ATP."
        ),
    },
    {
        "title": "Enzyme Kinetics",
        "content": (
            "Enzymes are biological catalysts that speed up chemical reactions without being "
            "consumed. They work by lowering the activation energy of a reaction. The "
            "Michaelis-Menten model describes enzyme kinetics: v = (Vmax × [S]) / (Km + [S]). "
            "Km is the substrate concentration at half-maximum velocity — a low Km means high "
            "affinity. Inhibitors can be competitive (compete with substrate for the active site, "
            "increasing apparent Km) or non-competitive (bind elsewhere, reducing Vmax). "
            "Temperature and pH affect enzyme activity: most human enzymes peak around 37°C "
            "and pH 7.4."
        ),
    },
]

DEMO_QUERIES = [
    "How do plants convert sunlight into chemical energy?",
    "What is the role of mitochondria in energy production?",
    "How do competitive inhibitors affect enzyme activity?",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def banner(text, width=72, char="="):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def section(text):
    print(f"\n--- {text} ---")


def ok_badge(ok):
    return "✓ GROUNDED" if ok else "✗ NO ANSWER"


async def run_pipelines(docs, queries, offline: bool):
    from hyperscholar.core.embedder import HashEmbedder, build_embedder
    from hyperscholar.core.config import load_config
    from hyperscholar.core.types import Document
    from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
    from hyperscholar.storage.memory import MemoryKVStorage, MemoryVectorStorage, MemoryHypergraphStorage, reset_memory_store

    cfg = load_config(os.path.join(HERE, "config.yaml"))

    # ── LLM + embedder ────────────────────────────────────────────────────────
    if offline:
        from hyperscholar.core.llm import StubLLM
        embedder = HashEmbedder(dim=256)
        # HyperRAGStubLLM knows HyperRAG's extraction prompt format and is used
        # when HyperRAG is available. For HierarchicalRAG (and as fallback) the
        # plain StubLLM is enough — it just needs to produce summary text.
        try:
            from hyperscholar.tests.hyperrag_stub import HyperRAGStubLLM
            hyper_llm = HyperRAGStubLLM()
        except Exception:
            hyper_llm = StubLLM()
        hier_llm = StubLLM(responses={
            "PASSAGES": "Energy conversion in biology: photosynthesis converts sunlight to glucose; respiration breaks glucose down to ATP; enzymes catalyse both processes.",
        })
        print("\n[mode] OFFLINE — hash embedder + stub LLM (no API calls)")
        print("        Switch to --live to use bge-m3 + DeepSeek.\n")
    else:
        embedder = build_embedder(cfg.embedding)
        from hyperscholar.core.llm import build_llm_func
        hyper_llm = hier_llm = build_llm_func(cfg.llm)
        print(f"\n[mode] LIVE — bge-m3 ({cfg.embedding.device}) + "
              f"DeepSeek ({cfg.llm.model})\n")

    hs_docs = [Document(content=d["content"], title=d["title"]) for d in docs]

    # ── shared storage ────────────────────────────────────────────────────────
    reset_memory_store()
    kv_cls   = MemoryKVStorage
    vec_cls  = MemoryVectorStorage
    hg_cls   = MemoryHypergraphStorage

    # ── build all three backends ──────────────────────────────────────────────
    # Probe importability first — the backends do a lazy import inside _rag()
    # so constructing them succeeds even when hyperrag isn't on sys.path.
    hyperrag_available = False
    hyper_be = light_be = None
    try:
        import hyperrag as _hr_probe  # noqa: F401
        from hyperscholar.rag.hyperrag_backend import HyperRAGBackend, HyperRAGLightBackend
        hyper_be = HyperRAGBackend(
            llm_func=hyper_llm, embedder=embedder,
            working_dir=os.path.join(HERE, "hyperscholar_runtime"),
            kv_cls=kv_cls, vector_cls=vec_cls, hypergraph_cls=hg_cls,
            cosine_threshold=-1.0 if offline else None,
        )
        light_be = HyperRAGLightBackend(
            llm_func=hyper_llm, embedder=embedder,
            working_dir=os.path.join(HERE, "hyperscholar_runtime"),
            kv_cls=kv_cls, vector_cls=vec_cls, hypergraph_cls=hg_cls,
            cosine_threshold=-1.0 if offline else None,
        )
        hyperrag_available = True
    except ModuleNotFoundError:
        print("[warn] HyperRAG not found on sys.path.")
        print(f"       Set HYPERRAG_PATH to your Hyper-RAG checkout, e.g.:")
        print(f"         set HYPERRAG_PATH=D:\\Projects\\moonlabs\\Hyper-RAG")
        print(f"       Only HierarchicalRAG will run for now.\n")

    hier_be = HierarchicalRAGBackend(
        llm_func=hier_llm, embedder=embedder,
        kv_cls=kv_cls, vector_cls=vec_cls,
        cosine_threshold=-1.0 if offline else None,
        cluster_threshold=0.05 if offline else 0.45,
    )

    # ── index ─────────────────────────────────────────────────────────────────
    banner("INDEXING")
    if hyperrag_available:
        section("HyperRAG (hyper + hyper-lite share one index)")
        t0 = time.time()
        ir = await hyper_be.index("demo", hs_docs)
        print(f"  chunks={ir.chunks}  time={time.time()-t0:.1f}s")

    section("HierarchicalRAG")
    t0 = time.time()
    ir = await hier_be.index("demo", hs_docs)
    print(f"  chunks={ir.chunks}  tree_nodes={ir.detail.get('tree_nodes', 0)}"
          f"  reused_chunks={ir.detail.get('reused_shared_chunks', 0)}"
          f"  time={time.time()-t0:.1f}s")

    # ── queries ───────────────────────────────────────────────────────────────
    banner("PIPELINE COMPARISON")
    for q in queries:
        print(f"\n{'─'*72}")
        print(f"QUERY: {q}")
        print(f"{'─'*72}")

        backends = []
        if hyperrag_available:
            backends += [("HyperRAG (hyper)", hyper_be),
                         ("HyperRAG-lite (hyper-lite)", light_be)]
        backends += [("HierarchicalRAG (collapsed-tree)", hier_be)]

        for label, be in backends:
            t0 = time.time()
            r = await be.query("demo", q)
            elapsed = time.time() - t0
            print(f"\n  [{label}]  {ok_badge(r.ok)}  ({elapsed:.1f}s)")
            print(f"  sources={len(r.sources)}")
            answer = r.answer.strip()
            # indent and wrap for readability
            for line in answer.split("\n"):
                print(f"    {line}")

    # ── concept graph spot-check ──────────────────────────────────────────────
    banner("CONCEPT GRAPH  (photosynthesis — feeds the Poincaré sphere)")
    spot_backends = []
    if hyperrag_available:
        spot_backends.append(("HyperRAG", hyper_be))
    spot_backends.append(("HierarchicalRAG", hier_be))
    for label, be in spot_backends:
        cg = await be.get_concept_graph("demo", "photosynthesis", depth=2)
        print(f"\n  [{label}]  center={cg.center}  "
              f"nodes={len(cg.nodes)}  edges={len(cg.edges)}")
        for n in cg.nodes[:6]:
            marker = "●" if n.is_center else "○"
            print(f"    {marker} [{n.level:+d}] {n.label[:60]}")
        if len(cg.nodes) > 6:
            print(f"    … and {len(cg.nodes)-6} more nodes")


def main():
    ap = argparse.ArgumentParser(description="HyperScholar Phase 1 demo")
    ap.add_argument("--file", help="path to a .txt file to index instead of the demo corpus")
    ap.add_argument("--query", action="append", help="query (repeatable); defaults to demo questions")
    ap.add_argument("--offline", action="store_true", default=True,
                    help="hash embedder + stub LLM (default)")
    ap.add_argument("--live", action="store_true",
                    help="real bge-m3 + DeepSeek (needs DEEPSEEK_API_KEY)")
    args = ap.parse_args()

    offline = not args.live

    docs = DEMO_CORPUS
    if args.file:
        text = open(args.file, encoding="utf-8").read()
        docs = [{"title": os.path.basename(args.file), "content": text}]

    queries = args.query or DEMO_QUERIES

    banner("HyperScholar Phase 1 — RAG Pipeline Comparison", char="█")
    print(f"  corpus : {len(docs)} document(s)")
    print(f"  queries: {len(queries)}")

    asyncio.run(run_pipelines(docs, queries, offline=offline))

    banner("DONE", char="─")


if __name__ == "__main__":
    main()
