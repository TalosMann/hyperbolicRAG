"""
HyperScholar — Pipeline Comparison GUI (Streamlit)
Run:  streamlit run gui.py
      Opens automatically at http://localhost:8501
"""
import asyncio
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HYPERRAG_PATH = os.environ.get("HYPERRAG_PATH", r"D:\Projects\hyperbolic\Hyper-RAG")
if os.path.isdir(HYPERRAG_PATH) and HYPERRAG_PATH not in sys.path:
    sys.path.insert(0, HYPERRAG_PATH)

import streamlit as st

from hyperscholar.core.embedder import HashEmbedder, build_embedder
from hyperscholar.core.config import load_config
from hyperscholar.core.types import Document
from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
from hyperscholar.storage.memory import (
    MemoryKVStorage, MemoryVectorStorage,
    MemoryHypergraphStorage, reset_memory_store,
)
from hyperscholar.ingestion import load_corpus, corpus_summary

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HyperScholar",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0d1117;
    color: #e6edf3;
}
.stApp { background-color: #0d1117; }

/* title */
.hs-title {
    font-family: 'Space Mono', monospace;
    color: #c9a84c;
    font-size: 1.6em;
    letter-spacing: 0.2em;
    text-align: center;
    padding: 8px 0 2px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 4px;
}
.hs-sub {
    font-family: 'Space Mono', monospace;
    color: #484f58;
    font-size: 0.7em;
    letter-spacing: 0.15em;
    text-align: center;
    margin-bottom: 24px;
}

/* section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    color: #c9a84c;
    font-size: 0.68em;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin: 16px 0 6px 0;
    border-left: 2px solid #c9a84c;
    padding-left: 8px;
}

/* result cards */
.result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
    min-height: 100px;
}
.result-card-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.78em;
    color: #c9a84c;
    margin-bottom: 10px;
    letter-spacing: 0.05em;
}
.badge-grounded {
    background: #1a3a2a;
    color: #3fb950;
    border: 1px solid #3fb95055;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75em;
    font-family: 'Space Mono', monospace;
    margin-right: 8px;
}
.badge-miss {
    background: #3a1a1a;
    color: #f85149;
    border: 1px solid #f8514955;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75em;
    font-family: 'Space Mono', monospace;
    margin-right: 8px;
}
.badge-meta {
    color: #484f58;
    font-size: 0.72em;
    font-family: 'Space Mono', monospace;
}
.answer-text {
    color: #cdd9e5;
    font-size: 0.92em;
    line-height: 1.65;
    margin-top: 10px;
    border-top: 1px solid #21262d;
    padding-top: 10px;
    font-family: 'Inter', sans-serif;
}
.graph-node-center { color: #c9a84c; font-family: 'Space Mono', monospace; font-size: 0.82em; }
.graph-node { color: #8b949e; font-family: 'Space Mono', monospace; font-size: 0.8em; }
.log-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75em;
    color: #58a6ff;
    white-space: pre-wrap;
}
.divider { border-color: #21262d; margin: 16px 0; }

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #010409 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #8b949e !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8em !important;
}

/* inputs */
.stTextInput input, .stTextArea textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px #c9a84c22 !important;
}

/* buttons */
.stButton button {
    background: #c9a84c !important;
    color: #0d1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100%;
}
.stButton button:hover { background: #dfc068 !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #21262d;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #484f58 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75em !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom: 2px solid #c9a84c !important;
    background: transparent !important;
}

/* file uploader */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}

/* selectbox */
.stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def _get_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def _run(coro):
    return _get_loop().run_until_complete(coro)


DEMO_DOCS = [
    Document(title="Photosynthesis", category="textbook", content=(
        "Photosynthesis is the biological process by which green plants convert light "
        "energy into chemical energy stored as glucose. Chlorophyll in the chloroplasts "
        "captures sunlight. Light-dependent reactions in the thylakoid membranes produce "
        "ATP and NADPH. The Calvin cycle in the stroma uses CO2 to synthesise glucose. "
        "Overall: 6CO2 + 6H2O + light → C6H12O6 + 6O2."
    )),
    Document(title="Cellular Respiration", category="textbook", content=(
        "Cellular respiration breaks down glucose to release ATP. Three stages: "
        "glycolysis (cytoplasm, 2 ATP), Krebs cycle (mitochondrial matrix, 2 ATP + "
        "electron carriers), and oxidative phosphorylation (inner mitochondrial membrane, "
        "~32 ATP via ATP synthase). Aerobic respiration uses O2; produces CO2 and H2O."
    )),
    Document(title="Enzyme Kinetics", category="textbook", content=(
        "Enzymes are biological catalysts that lower activation energy. "
        "Michaelis-Menten: v = (Vmax × [S]) / (Km + [S]). Low Km = high affinity. "
        "Competitive inhibitors increase apparent Km; non-competitive reduce Vmax. "
        "Human enzymes typically peak at 37°C and pH 7.4."
    )),
]


# ── session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "backends": {},
        "indexed": False,
        "index_log": "",
        "corpus_info": "",
        "results": {},
        "graphs": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── backend setup ─────────────────────────────────────────────────────────────
def setup_backends(mode: str):
    reset_memory_store()
    st.session_state.backends = {}
    st.session_state.indexed = False
    offline = (mode == "Offline")

    cfg = load_config(os.path.join(HERE, "config.yaml"))

    if offline:
        from hyperscholar.tests.hyperrag_stub import HyperRAGStubLLM
        from hyperscholar.core.llm import StubLLM
        embedder  = HashEmbedder(dim=256)
        hyper_llm = HyperRAGStubLLM()
        hier_llm  = StubLLM(responses={
            "PASSAGES": "Biology: photosynthesis converts sunlight to glucose; "
                        "respiration produces ATP; enzymes catalyse reactions.",
        })
    else:
        embedder = build_embedder(cfg.embedding)
        from hyperscholar.core.llm import build_llm_func
        live_llm  = build_llm_func(cfg.llm)
        hyper_llm = hier_llm = live_llm

    kv, vec, hg = MemoryKVStorage, MemoryVectorStorage, MemoryHypergraphStorage
    threshold = -1.0 if offline else None

    try:
        from hyperscholar.rag.hyperrag_backend import HyperRAGBackend, HyperRAGLightBackend
        st.session_state.backends["HyperRAG"] = HyperRAGBackend(
            llm_func=hyper_llm, embedder=embedder,
            working_dir=os.path.join(HERE, "hyperscholar_runtime"),
            kv_cls=kv, vector_cls=vec, hypergraph_cls=hg,
            cosine_threshold=threshold,
        )
        st.session_state.backends["HyperRAG-lite"] = HyperRAGLightBackend(
            llm_func=hyper_llm, embedder=embedder,
            working_dir=os.path.join(HERE, "hyperscholar_runtime"),
            kv_cls=kv, vector_cls=vec, hypergraph_cls=hg,
            cosine_threshold=threshold,
        )
    except ModuleNotFoundError:
        pass

    st.session_state.backends["HierarchicalRAG"] = HierarchicalRAGBackend(
        llm_func=hier_llm, embedder=embedder,
        kv_cls=kv, vector_cls=vec,
        cosine_threshold=threshold,
        cluster_threshold=0.05 if offline else 0.45,
    )
    return embedder


def do_index(docs: list, mode: str):
    if not docs:
        return "⚠ No documents to index."
    if not st.session_state.backends:
        setup_backends(mode)

    log = [f"Indexing {len(docs)} document(s)...\n"]
    if "HyperRAG" in st.session_state.backends:
        t0 = time.time()
        ir = _run(st.session_state.backends["HyperRAG"].index("demo", docs))
        log.append(f"✓ HyperRAG + HyperRAG-lite  —  {ir.chunks} chunks  "
                   f"({time.time()-t0:.1f}s)")
    else:
        log.append("⚠ HyperRAG not available — set HYPERRAG_PATH env var")

    t0 = time.time()
    ir = _run(st.session_state.backends["HierarchicalRAG"].index("demo", docs))
    log.append(
        f"✓ HierarchicalRAG  —  {ir.chunks} chunks · "
        f"{ir.detail.get('tree_nodes', 0)} tree nodes · "
        f"{ir.detail.get('reused_shared_chunks', 0)} reused  "
        f"({time.time()-t0:.1f}s)"
    )
    st.session_state.indexed = True
    return "\n".join(log)


# ── result card renderer ──────────────────────────────────────────────────────
def render_result(name: str):
    if name not in st.session_state.results:
        st.markdown(
            '<div class="result-card"><span style="color:#484f58;font-family:\'Space Mono\',monospace;font-size:0.8em">'
            'Run a query to see results.</span></div>',
            unsafe_allow_html=True)
        return

    r, t = st.session_state.results[name]["r"], st.session_state.results[name]["t"]
    badge = '<span class="badge-grounded">✓ GROUNDED</span>' if r.ok \
            else '<span class="badge-miss">✗ NO ANSWER</span>'
    meta  = f'<span class="badge-meta">{t:.2f}s &nbsp;·&nbsp; {len(r.sources)} source(s)</span>'
    answer = r.answer.strip().replace("\n", "<br>") if r.answer else "—"
    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-header">{name}</div>
        <div>{badge}{meta}</div>
        <div class="answer-text">{answer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_graph(name: str):
    if name not in st.session_state.graphs:
        st.markdown(
            '<div class="result-card"><span style="color:#484f58;font-family:\'Space Mono\',monospace;font-size:0.8em">'
            'Enter a concept to build the graph.</span></div>',
            unsafe_allow_html=True)
        return

    cg = st.session_state.graphs[name]
    if not cg.nodes:
        st.markdown(f'<div class="result-card"><span style="color:#484f58">No concept found.</span></div>',
                    unsafe_allow_html=True)
        return

    rows = []
    for n in sorted(cg.nodes, key=lambda x: x.level):
        if n.is_center:
            rows.append(f'<div class="graph-node-center">● [{n.level:+d}] {n.label[:70]}</div>')
        else:
            rows.append(f'<div class="graph-node">○ [{n.level:+d}] {n.label[:70]}</div>')

    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-header">{name} &nbsp;·&nbsp;
            <span style="color:#484f58">{len(cg.nodes)} nodes · {len(cg.edges)} edges</span>
        </div>
        {''.join(rows[:12])}
        {'<div class="graph-node" style="color:#484f58">… ' + str(len(cg.nodes)-12) + ' more</div>' if len(cg.nodes) > 12 else ''}
    </div>
    """, unsafe_allow_html=True)


# ── layout ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hs-title">◈ HYPERSCHOLAR</div>', unsafe_allow_html=True)
st.markdown('<div class="hs-sub">RAG PIPELINE COMPARISON · PHASE 1</div>', unsafe_allow_html=True)

# ── sidebar: corpus + mode ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Offline", "Live (LM Studio / DeepSeek)"],
                    label_visibility="collapsed")

    st.markdown('<div class="section-label">Corpus</div>', unsafe_allow_html=True)
    source_tab = st.selectbox("Source", ["Demo corpus", "Upload files",
                                          "Folder path", "JSON / JSONL file"],
                               label_visibility="collapsed")

    if source_tab == "Demo corpus":
        st.caption("3-doc biology corpus — verifies the pipelines.")
        if st.button("▶  Load demo corpus"):
            setup_backends(mode)
            with st.spinner("Indexing…"):
                log = do_index(DEMO_DOCS, mode)
            st.session_state.index_log = log
            st.session_state.corpus_info = corpus_summary(DEMO_DOCS)

    elif source_tab == "Upload files":
        st.caption("PDF, TXT, MD, JSON, JSONL")
        uploaded = st.file_uploader("", type=["pdf","txt","md","json","jsonl"],
                                    accept_multiple_files=True,
                                    label_visibility="collapsed")
        if st.button("▶  Index uploaded files"):
            if uploaded:
                tmp = tempfile.mkdtemp()
                paths = []
                for f in uploaded:
                    dst = os.path.join(tmp, f.name)
                    with open(dst, "wb") as fh:
                        fh.write(f.read())
                    paths.append(dst)
                setup_backends(mode)
                with st.spinner("Loading & indexing…"):
                    docs = load_corpus(paths)
                    shutil.rmtree(tmp, ignore_errors=True)
                    log = do_index(docs, mode)
                st.session_state.index_log = log
                st.session_state.corpus_info = corpus_summary(docs)
            else:
                st.warning("Upload at least one file first.")

    elif source_tab == "Folder path":
        st.caption("Point at a local folder. Works with NeurologyCorp and similar datasets.")
        folder = st.text_input("Folder path",
                               placeholder=r"e.g. D:\Datasets\NeurologyCorp",
                               label_visibility="collapsed")
        recursive = st.checkbox("Include subfolders", value=True)
        if st.button("▶  Scan & index folder"):
            if folder and os.path.isdir(folder):
                setup_backends(mode)
                with st.spinner("Scanning & indexing…"):
                    docs = load_corpus(folder, recursive=recursive)
                    log = do_index(docs, mode)
                st.session_state.index_log = log
                st.session_state.corpus_info = corpus_summary(docs)
            else:
                st.warning("Enter a valid folder path.")

    elif source_tab == "JSON / JSONL file":
        st.caption("iMoonLab dataset format or any JSON list of documents.")
        json_path = st.text_input("File path",
                                   placeholder=r"e.g. D:\Datasets\neurology.json",
                                   label_visibility="collapsed")
        category = st.selectbox("Category",
                                 ["general","textbook","exam","research_paper"])
        if st.button("▶  Index dataset file"):
            if json_path and os.path.isfile(json_path):
                setup_backends(mode)
                with st.spinner("Loading & indexing…"):
                    docs = load_corpus(json_path, category=category)
                    log = do_index(docs, mode)
                st.session_state.index_log = log
                st.session_state.corpus_info = corpus_summary(docs)
            else:
                st.warning("Enter a valid file path.")

    # index log
    if st.session_state.corpus_info:
        st.markdown('<div class="section-label">Corpus</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="log-box">{st.session_state.corpus_info}</div>',
                    unsafe_allow_html=True)
    if st.session_state.index_log:
        st.markdown('<div class="section-label">Index log</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="log-box">{st.session_state.index_log}</div>',
                    unsafe_allow_html=True)

# ── main: query ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input("", placeholder="Ask something about the indexed corpus…",
                           label_visibility="collapsed", key="query_input")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_query = st.button("⟳  Run")

if run_query and query.strip():
    if not st.session_state.indexed:
        st.warning("Load and index a corpus first (use the sidebar).")
    else:
        with st.spinner("Querying all backends…"):
            for name, be in st.session_state.backends.items():
                t0 = time.time()
                r = _run(be.query("demo", query))
                st.session_state.results[name] = {"r": r, "t": time.time() - t0}

# results — three tabs
tab1, tab2, tab3 = st.tabs(["HyperRAG", "HyperRAG-lite", "HierarchicalRAG"])
with tab1: render_result("HyperRAG")
with tab2: render_result("HyperRAG-lite")
with tab3: render_result("HierarchicalRAG")

# ── concept graph ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Concept Graph</div>', unsafe_allow_html=True)
col_g, col_gbtn = st.columns([5, 1])
with col_g:
    concept = st.text_input("", placeholder="e.g. photosynthesis, neuron, mitochondria…",
                             label_visibility="collapsed", key="graph_input")
with col_gbtn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_graph = st.button("◈  Build")

if run_graph and concept.strip():
    if not st.session_state.indexed:
        st.warning("Index a corpus first.")
    else:
        with st.spinner("Building concept graphs…"):
            for name, be in st.session_state.backends.items():
                if name == "HyperRAG-lite":
                    continue  # shares index with HyperRAG
                cg = _run(be.get_concept_graph("demo", concept, depth=2))
                st.session_state.graphs[name] = cg

gcol1, gcol2 = st.columns(2)
with gcol1: render_graph("HyperRAG")
with gcol2: render_graph("HierarchicalRAG")
