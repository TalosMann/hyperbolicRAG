"""
cograg_safe — interactive test GUI (Streamlit)

Upload/paste a corpus, index it (optionally with full entity/relation
extraction for multi-hop testing), then ask your own questions and see:
  - the intent classification (CORPUS vs SYNTHESIS) and serve/decline decision
  - the answer, split into its two tracks exactly as composed:
      "From this text:"  (verified, cited, R1/R2/F2-gated)
      "General literary knowledge:" (unverified, only in SYNTHESIS mode)
  - the full verification audit trail (what was proposed, what survived,
    what got rejected and by which gate) -- for testing the gates themselves
  - optionally, the raw (unsafe) cograg_official answer side-by-side, so you
    can see directly what the safety layer is preventing

Run:  streamlit run gui_cograg_safe.py --server.fileWatcherType none
      Opens at http://localhost:8501
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st

from hyperscholar.core.config import load_config
from hyperscholar.core.embedder import build_embedder
from hyperscholar.core.llm import build_llm_func
from hyperscholar.core.types import Document
from hyperscholar.ingestion import load_corpus, corpus_summary
from hyperscholar.cograg_safe.backend import CogRAGSafeBackend
from hyperscholar.rag.cograg_official_backend import CogRAGOfficialBackend

RUNTIME = os.path.join(HERE, "hyperscholar_runtime", "cograg_official")

st.set_page_config(page_title="cograg_safe tester", page_icon="\U0001F6E1",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
.cs-title { font-family: monospace; color: #c9a84c; font-size: 1.4em;
            letter-spacing: 0.15em; text-align: center; padding: 6px 0 2px 0;
            border-bottom: 1px solid #21262d; margin-bottom: 4px; }
.cs-sub { font-family: monospace; color: #484f58; font-size: 0.7em;
          letter-spacing: 0.1em; text-align: center; margin-bottom: 20px; }
.section-label { font-family: monospace; color: #c9a84c; font-size: 0.68em;
                  letter-spacing: 0.14em; text-transform: uppercase;
                  margin: 14px 0 6px 0; border-left: 2px solid #c9a84c; padding-left: 8px; }
.badge { border-radius: 4px; padding: 2px 9px; font-size: 0.75em;
         font-family: monospace; margin-right: 8px; display: inline-block; }
.badge-serve { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb95055; }
.badge-decline { background: #3a1a1a; color: #f85149; border: 1px solid #f8514955; }
.badge-corpus { background: #1a2a3a; color: #58a6ff; border: 1px solid #58a6ff55; }
.badge-synthesis { background: #3a2a1a; color: #d29922; border: 1px solid #d2992255; }
.corpus-block { background: #161b22; border-left: 3px solid #3fb950; border-radius: 6px;
                padding: 12px 16px; margin: 10px 0; font-family: inherit; white-space: pre-wrap; }
.gk-block { background: #161b22; border-left: 3px solid #d29922; border-radius: 6px;
            padding: 12px 16px; margin: 10px 0; white-space: pre-wrap; }
.gk-label { color: #d29922; font-family: monospace; font-size: 0.72em;
            letter-spacing: 0.06em; margin-bottom: 6px; }
.corpus-label { color: #3fb950; font-family: monospace; font-size: 0.72em;
                letter-spacing: 0.06em; margin-bottom: 6px; }
.log-box { background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
           padding: 10px; font-family: monospace; font-size: 0.75em; color: #58a6ff;
           white-space: pre-wrap; }
.rel-ok { color: #3fb950; font-family: monospace; font-size: 0.82em; }
.rel-bad { color: #f85149; font-family: monospace; font-size: 0.82em; }
.unsafe-box { background: #2a1414; border: 1px solid #f8514955; border-radius: 6px;
              padding: 12px 16px; margin: 10px 0; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)


# ── async helper ──────────────────────────────────────────────────────────────
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


# ── session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {"backend": None, "unsafe_backend": None, "namespace": None,
               "indexed_namespaces": [], "index_log": "", "corpus_info": "",
               "history": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _list_namespaces() -> list[str]:
    if not os.path.isdir(RUNTIME):
        return []
    out = []
    for d in sorted(os.listdir(RUNTIME)):
        if os.path.isfile(os.path.join(RUNTIME, d, "kv_store_text_chunks.json")):
            out.append(d)
    return out


def _namespace_stats(ns: str) -> dict:
    p = os.path.join(RUNTIME, ns, "kv_store_text_chunks.json")
    if not os.path.isfile(p):
        return {}
    chunks = json.loads(Path(p).read_text(encoding="utf-8"))
    docs = {c.get("full_doc_id") for c in chunks.values()}
    stats = {"chunks": len(chunks), "documents": len(docs)}
    hg_path = os.path.join(RUNTIME, ns, "hypergraph_chunk_entity_relation.hgdb")
    if os.path.isfile(hg_path) and os.path.getsize(hg_path) > 200:
        try:
            from hyperdb import HypergraphDB
            hg = HypergraphDB(storage_file=hg_path)
            stats["entities"] = len(list(hg.all_v))
            stats["hyperedges"] = len(list(hg.all_e))
        except Exception:
            pass
    return stats


def _ensure_backends():
    if st.session_state.backend is None:
        cfg = load_config(os.path.join(HERE, "config.yaml"))
        st.session_state.cfg = cfg
        embedder = build_embedder(cfg.embedding)
        llm = build_llm_func(cfg.llm)
        st.session_state.backend = CogRAGSafeBackend(
            llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
            fail_markers=cfg.rag.fail_markers)
        st.session_state.unsafe_backend = CogRAGOfficialBackend(
            llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
            query_mode="cog-entity", fail_markers=cfg.rag.fail_markers)
    return st.session_state.backend, st.session_state.unsafe_backend


# ── chunks-only indexing (no entity-extraction LLM calls; local embed only) ──
def _index_chunks_only(be: CogRAGSafeBackend, ns: str, docs: list[Document]):
    """No-ops entity extraction for this call only, so indexing is fast/free
    (embedding only). Multi-hop/relational tests need the real hypergraph --
    use the full extraction option for those. No edits to the Cog-RAG folder;
    this only patches the function reference for the duration of the call."""
    import cograg.cograg as cc
    original = cc.extract_entities

    async def _noop(inserting_chunks=None, knowledge_hypergraph_inst=None,
                    knowledge_hypergraph_theme=None, **kw):
        return knowledge_hypergraph_inst, knowledge_hypergraph_theme

    cc.extract_entities = _noop
    try:
        return _run(be._inner.index(ns, docs))
    finally:
        cc.extract_entities = original


def do_index(ns: str, docs: list[Document], full_extraction: bool):
    if not docs:
        return "No documents to index."
    be, _ = _ensure_backends()
    t0 = time.time()
    if full_extraction:
        ir = _run(be.index(ns, docs))
    else:
        ir = _index_chunks_only(be, ns, docs)
        be.invalidate(ns)
    dt = time.time() - t0
    stats = _namespace_stats(ns)
    log = [f"Indexed {ir.documents} document(s) -> {ir.chunks} chunks in {dt:.1f}s"]
    if "entities" in stats:
        log.append(f"Hypergraph: {stats['entities']} entities, {stats['hyperedges']} hyperedges")
    elif full_extraction:
        log.append("(entity extraction ran but produced an empty/near-empty graph -- "
                   "check the LLM provider is actually reachable)")
    else:
        log.append("Chunks-only index (no hypergraph) -- fast, but R1 relies on raw-text "
                   "co-mention only. Use full extraction for real multi-hop tests.")
    st.session_state.namespace = ns
    st.session_state.indexed_namespaces = _list_namespaces()
    return "\n".join(log)


# ── answer rendering ──────────────────────────────────────────────────────────
def _split_answer(answer: str) -> tuple[str, str | None]:
    marker = "\n\nGeneral literary knowledge"
    if marker in answer:
        i = answer.index(marker)
        return answer[:i], answer[i + 2:]
    return answer, None


def render_answer(entry: dict):
    r = entry["result"]
    intent = (r.raw or {}).get("intent", "?")
    decision_badge = ('<span class="badge badge-serve">SERVE</span>' if r.ok
                      else '<span class="badge badge-decline">DECLINE</span>')
    intent_badge = (f'<span class="badge badge-{"synthesis" if intent=="SYNTHESIS" else "corpus"}">'
                    f'{intent}</span>')
    st.markdown(f"{decision_badge}{intent_badge}"
               f'<span style="color:#484f58;font-family:monospace;font-size:0.78em">'
               f'{entry["dt"]:.1f}s</span>', unsafe_allow_html=True)

    corpus_part, gk_part = _split_answer(r.answer)
    st.markdown(f'<div class="corpus-label">FROM THIS TEXT (verified, cited)</div>'
               f'<div class="corpus-block">{corpus_part}</div>', unsafe_allow_html=True)
    if gk_part:
        st.markdown(f'<div class="gk-label">GENERAL KNOWLEDGE (not verified -- double-check)</div>'
                   f'<div class="gk-block">{gk_part}</div>', unsafe_allow_html=True)

    if entry.get("unsafe_answer") is not None:
        st.markdown('<div class="section-label">Raw cograg_official (unsafe) answer, for comparison</div>',
                   unsafe_allow_html=True)
        st.markdown(f'<div class="unsafe-box">{entry["unsafe_answer"]}</div>', unsafe_allow_html=True)

    raw = r.raw or {}
    with st.expander("Verification audit (what was proposed, verified, rejected)"):
        vr = raw.get("verified_relations", [])
        rr = raw.get("rejected_relations", [])
        vf = raw.get("verified_facts", [])
        df = raw.get("dropped_facts", [])
        st.caption(f"{len(vr)} relation(s) verified · {len(rr)} rejected · "
                  f"{len(vf)} fact(s) verified · {len(df)} dropped")
        for x in vr:
            st.markdown(f'<div class="rel-ok">+ {x["a"]} -- {x["relation"]} -- {x["b"]}  '
                       f'[{", ".join(x.get("provenance", [])[:2])}]</div>', unsafe_allow_html=True)
        for x in rr:
            st.markdown(f'<div class="rel-bad">- [{x.get("gate","?")}] {x["a"]} -- '
                       f'{x["relation"]} -- {x["b"]}  ({x.get("reason","")})</div>',
                       unsafe_allow_html=True)
        for x in vf:
            st.markdown(f'<div class="rel-ok">+ {x["statement"]}</div>', unsafe_allow_html=True)
        for x in df:
            st.markdown(f'<div class="rel-bad">- [{x.get("gate","?")}] {x["statement"]}  '
                       f'({x.get("reason","")})</div>', unsafe_allow_html=True)

    if r.sources:
        with st.expander(f"Retrieved evidence chunks ({len(r.sources)} shown)"):
            for s in r.sources:
                st.markdown(f'<div class="log-box"><b>{s.chunk_id}</b><br>{s.excerpt}</div>',
                           unsafe_allow_html=True)


# ── layout ────────────────────────────────────────────────────────────────────
st.markdown('<div class="cs-title">\U0001F6E1 COGRAG_SAFE TESTER</div>', unsafe_allow_html=True)
st.markdown('<div class="cs-sub">UPLOAD A CORPUS · ASK YOUR OWN QUESTIONS · '
           'SEE WHAT GETS VERIFIED VS DECLINED</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-label">LLM provider</div>', unsafe_allow_html=True)
    try:
        _cfg_preview = load_config(os.path.join(HERE, "config.yaml"))
        _active = next((p.name for p in _cfg_preview.llm.providers
                       if os.environ.get(p.api_key_env, "")), "none configured")
        st.caption(f"Active: **{_active}**  (first provider in config.yaml whose "
                  f"API key env var is set)")
    except Exception as e:
        st.caption(f"Could not read config: {e}")

    st.markdown('<div class="section-label">Namespace</div>', unsafe_allow_html=True)
    existing = _list_namespaces()
    pick = st.selectbox("Use an already-indexed namespace",
                        ["(new)"] + existing, label_visibility="collapsed")
    if pick != "(new)":
        st.session_state.namespace = pick
        stats = _namespace_stats(pick)
        st.caption(f"{stats.get('chunks','?')} chunks · {stats.get('documents','?')} docs"
                  + (f" · {stats['entities']} entities · {stats['hyperedges']} hyperedges"
                     if "entities" in stats else " · chunks-only (no hypergraph)"))

    st.markdown('<div class="section-label">Add / index a corpus</div>', unsafe_allow_html=True)
    new_ns = st.text_input("New namespace name", value="my_test_corpus" if pick == "(new)" else pick)
    source_tab = st.selectbox("Source", ["Paste text", "Upload files", "Folder path",
                                        "JSON / JSONL file"], label_visibility="collapsed")
    full_extraction = st.checkbox(
        "Extract entities & relations (needed for multi-hop testing)", value=True)
    if not full_extraction:
        st.caption("Fast, free (local embedding only). No hypergraph -- R1 falls back to "
                  "raw-text co-mention only.")
    else:
        st.caption("Real LLM calls per chunk (entity + theme extraction). Can take a "
                  "while on a free-tier model -- fine for a few short documents.")

    docs = None
    if source_tab == "Paste text":
        pasted = st.text_area("Paste corpus text", height=150,
                              placeholder="Paste a passage, chapter, or article here...")
        if st.button("\u25B6  Index pasted text"):
            if pasted.strip():
                docs = [Document(content=pasted, title=new_ns)]
    elif source_tab == "Upload files":
        uploaded = st.file_uploader("PDF, TXT, MD, JSON, JSONL", type=["pdf", "txt", "md", "json", "jsonl"],
                                    accept_multiple_files=True, label_visibility="collapsed")
        if st.button("\u25B6  Index uploaded files"):
            if uploaded:
                tmp = tempfile.mkdtemp()
                paths = []
                for f in uploaded:
                    dst = os.path.join(tmp, f.name)
                    with open(dst, "wb") as fh:
                        fh.write(f.read())
                    paths.append(dst)
                docs = load_corpus(paths)
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                st.warning("Upload at least one file first.")
    elif source_tab == "Folder path":
        folder = st.text_input("Folder path", placeholder=r"e.g. D:\Datasets\mycorpus")
        recursive = st.checkbox("Include subfolders", value=True)
        if st.button("\u25B6  Scan & index folder"):
            if folder and os.path.isdir(folder):
                docs = load_corpus(folder, recursive=recursive)
            else:
                st.warning("Enter a valid folder path.")
    elif source_tab == "JSON / JSONL file":
        json_path = st.text_input("File path", placeholder=r"e.g. D:\Datasets\corpus.jsonl")
        if st.button("\u25B6  Index dataset file"):
            if json_path and os.path.isfile(json_path):
                docs = load_corpus(json_path)
            else:
                st.warning("Enter a valid file path.")

    if docs:
        with st.spinner("Indexing... (see log below once done)"):
            log = do_index(new_ns, docs, full_extraction)
        st.session_state.index_log = log
        st.session_state.corpus_info = corpus_summary(docs)

    if st.session_state.corpus_info:
        st.markdown('<div class="section-label">Last corpus loaded</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="log-box">{st.session_state.corpus_info}</div>', unsafe_allow_html=True)
    if st.session_state.index_log:
        st.markdown('<div class="section-label">Index log</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="log-box">{st.session_state.index_log}</div>', unsafe_allow_html=True)

# ── main: query ───────────────────────────────────────────────────────────────
ns = st.session_state.namespace
if ns:
    st.markdown(f'<div class="section-label">Querying: {ns}</div>', unsafe_allow_html=True)
else:
    st.info("Index a corpus in the sidebar first (or pick an existing namespace).")

show_unsafe = st.checkbox(
    "Also show cograg_official's raw (unsafe) answer, for comparison", value=False)

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input("", placeholder="Ask anything -- factual, relational, or a "
                          "cross-work comparison (e.g. 'How does X mirror Y in another story?')",
                          label_visibility="collapsed", key="query_input")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_query = st.button("\u23FA  Ask")

if run_query and query.strip():
    if not ns:
        st.warning("Index a corpus first.")
    else:
        be, unsafe_be = _ensure_backends()
        with st.spinner("Retrieving, asserting, verifying..."):
            t0 = time.time()
            r = _run(be.query(ns, query))
            dt = time.time() - t0
            unsafe_answer = None
            if show_unsafe:
                ur = _run(unsafe_be.query(ns, query))
                unsafe_answer = ur.answer
        st.session_state.history.insert(0, {"q": query, "result": r, "dt": dt,
                                            "unsafe_answer": unsafe_answer})

# ── history (most recent first) ───────────────────────────────────────────────
for i, entry in enumerate(st.session_state.history):
    st.markdown(f'<div class="section-label">Q{len(st.session_state.history)-i}: '
               f'{entry["q"]}</div>', unsafe_allow_html=True)
    render_answer(entry)
    st.markdown("---")
