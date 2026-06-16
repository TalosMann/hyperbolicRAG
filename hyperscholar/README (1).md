# HyperScholar — Phase 1 + Evaluation Framework

**Status:** Phase 1 complete · 16/16 tests passing · GUI running on Windows (Python 3.12) · Eval framework added

HyperScholar is a hypergraph-based RAG platform for personalized education. Phase 1 implements a swappable RAG abstraction layer with three backends, a corpus ingestion pipeline, and a browser GUI for comparing results side by side. The evaluation framework adds a reproducible HyperRAG vs HierarchicalRAG comparison following the iMoonLab Hyper-RAG paper's protocol.

---

## Project structure

```
hyperscholar/
├── config.yaml                   # all config — backend, LLM, embedder, store
├── gui.py                        # Streamlit browser GUI  ← main entry point
├── run_demo.py                   # terminal demo (offline smoke test)
├── ingestion.py                  # corpus loader: PDF, TXT, JSON, JSONL, folders
├── requirements.txt
│
├── core/
│   ├── types.py                  # Document, QueryResult, ConceptGraph, RouterResult
│   ├── config.py                 # YAML loader + ${ENV:default} interpolation
│   ├── embedder.py               # BgeM3Embedder (auto device) · HashEmbedder (tests)
│   └── llm.py                    # build_llm_func (provider fallback chain) · StubLLM
│
├── storage/
│   ├── memory.py                 # in-process stores (dsn: memory://)
│   └── pg.py                     # pgvector stores (dsn: postgresql://...)
│
├── db/
│   └── schema.sql                # shared / hyperrag / hierarchical schemas
│
├── rag/
│   ├── base.py                   # RAGBackend ABC (4 methods — the only contract)
│   ├── hyperrag_backend.py       # HyperRAG + HyperRAG-lite (shared index)
│   ├── hierarchical_backend.py   # RAPTOR-style summary tree
│   ├── router.py                 # three-tier classroom/personal routing
│   └── factory.py                # config → wired backend + router
│
├── scripts/
│   └── compare_backends.py       # terminal: index once, query all backends
│
├── eval/                         # ← evaluation framework (new)
│   ├── preindex.py               # headless GPU indexing into both backends
│   ├── corpus_export.py          # full hypergraph / summary-tree structure dumps
│   ├── question_generator.py     # N chunk-anchored questions per corpus
│   ├── runner.py                 # answers + provenance from both backends
│   ├── judge.py                  # LLM-as-judge, 5 metrics, blind + randomized
│   ├── report.py                 # markdown comparison tables
│   ├── run_all.py                # one-command orchestrator (steps 2–5)
│   ├── provenance.py             # query wrappers that capture retrieval provenance
│   ├── hierarchical_prompts.py   # mirror of backend answer prompts
│   └── results/
│       └── <corpus>/             # per-corpus outputs (json + md)
│
└── tests/
    ├── test_phase1.py            # 16 tests: storage contracts, router, conformance
    └── hyperrag_stub.py          # offline HyperRAG-aware stub LLM
```

---

## Directory layout on disk (Windows)

```
D:\Projects\hyperbolic\
├── venv\                         # Python 3.12 venv
├── hyperscholar\                 # this repo
└── Hyper-RAG\                    # iMoonLab Hyper-RAG checkout
    └── hyperrag\                 # the actual package (utils.py, operate.py, etc.)
```

**Critical:** `Hyper-RAG\` must sit one level above `hyperscholar\`. The `run_demo.py` and `gui.py` both look for it via the `HYPERRAG_PATH` env var, defaulting to `D:\Projects\hyperbolic\Hyper-RAG`.

---

## One-time setup

### 1. Python environment

```powershell
cd D:\Projects\hyperbolic
py -3.12 -m venv venv
venv\Scripts\activate
pip install pyyaml numpy openai sentence-transformers pytest pytest-asyncio ^
    nano-vectordb hypergraph-db aiohttp tenacity pydantic pdfplumber ^
    streamlit accelerate xxhash pillow
```

### 2. Hyper-RAG checkout

```powershell
cd D:\Projects\hyperbolic
git clone https://github.com/iMoonLab/Hyper-RAG
# Or wrap an existing hyperrag/ subfolder:
mkdir Hyper-RAG
move hyperrag Hyper-RAG\hyperrag
```

### 3. Tiktoken patch (required — network blocks BPE vocab downloads)

In `D:\Projects\hyperbolic\Hyper-RAG\hyperrag\utils.py` replace the bodies of:

```python
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    return list(content)

def decode_tokens_by_tiktoken(tokens: list, model_name: str = "gpt-4o"):
    return "".join(tokens)
```

Comment out `import tiktoken` in the same file, and `import aioboto3` in `llm.py`.

### 4. GPU / PyTorch — RTX 50-series (Blackwell, sm_120)

**If you have an RTX 50-series card (e.g. RTX 5060 Ti):** the stable PyTorch wheels only support up to `sm_90` and will fail at embedding time with `CUDA error: no kernel image is available for execution on the device`. This error is silently swallowed by HyperRAG's `ainsert` try/finally, so the symptom is indexing that embeds and then exits with no entity-extraction step and no error.

Install the CUDA 12.8 nightly which includes Blackwell support:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 2.12.0.dev...+cu128   True   NVIDIA GeForce RTX 5060 Ti
```

Then set `embedding.device: auto` in `config.yaml` to use the GPU. On older cards, or if you skip the nightly, set `device: cpu` (works, just slow — ~25s/chunk).

---

## Running the GUI

```powershell
cd D:\Projects\hyperbolic\hyperscholar
streamlit run gui.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` flag avoids a noisy `transformers`/`torchvision` introspection crash in Streamlit's watcher (tradeoff: no auto-reload on edit).

Opens at `http://localhost:8501`.

### GUI workflow

1. **Sidebar → Mode**: Offline (stub LLM) or Live (real LLM)
2. **Sidebar → Corpus**: pick a source and click Index
3. **Main area → Query**: type a question and Run
4. **Tabs**: HyperRAG / HyperRAG-lite / HierarchicalRAG results

---

## Corpus sources

| Source | How to use |
|--------|-----------|
| Demo | One click — 3-doc biology corpus, good for verifying pipelines |
| Upload files | Drag-and-drop PDF, TXT, MD, JSON, JSONL into the browser |
| Folder path | Paste a local folder path — scans all supported files recursively |
| JSON / JSONL file | Paste a file path — handles iMoonLab dataset format automatically |

### iMoonLab dataset format (NeurologyCorp, physics, agriculture, etc.)

All iMoonLab corpora come from the same `Step_0.py` preprocessing (1200-token chunks, 100-token overlap) and share a JSONL shape with these per-line keys:

```
id, title, context, contexts
```

The document body is in **`context`** (singular). `contexts` (plural) is the QA-evaluation source list and is ignored. The ingestion parser checks field names in this order:

```
content → context → text → passage → abstract
```

so all iMoonLab corpora parse without modification. Other field names need a one-line addition in `ingestion.py`'s `_parse_json_corpus`.

---

## LLM providers

Configured in `config.yaml`; first provider with a usable key/endpoint wins:

```yaml
llm:
  providers:
    - name: deepseek
      base_url: https://api.deepseek.com/v1
      model: deepseek-chat
      api_key_env: DEEPSEEK_API_KEY
    - name: siliconflow
      base_url: https://api.siliconflow.cn/v1
      model: deepseek-ai/DeepSeek-V3
      api_key_env: SILICONFLOW_API_KEY
    - name: local_ollama
      base_url: http://localhost:11434/v1
      model: deepseek-r1:7b
      api_key_env: ""
    - name: local_lmstudio
      base_url: http://localhost:1234/v1
      model: local-model
      api_key_env: ""
```

**On local LLMs (LM Studio / Ollama) and entity extraction:** HyperRAG fires an LLM call per chunk in parallel via `asyncio.gather`. LM Studio is single-threaded and returns `502 Bad Gateway` under that load, especially when it is also competing with bge-m3 for GPU memory. For large-corpus indexing, prefer a remote provider (DeepSeek/SiliconFlow) which handles concurrency. To use a local model anyway: keep the embedder on GPU and the LLM with enough headroom, use a model with ≥8K context (small models like Llama-3.2-3B reject the extraction prompt), and free other GPU processes first.

### Setting API keys

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
streamlit run gui.py --server.fileWatcherType none
```

---

## Embedding

`BAAI/bge-m3` runs locally via `sentence-transformers` (1024-dim). Downloads once (~2.3GB) to `~/.cache/huggingface/`. Device auto-detected: `cuda → mps → cpu`.

```yaml
embedding:
  model: BAAI/bge-m3
  dim: 1024
  device: auto      # cpu if no Blackwell-capable torch (see GPU setup above)
  batch_size: 8
```

**Dimension mismatch:** stale vector caches built with a different embedding model (e.g. 768 vs 1024) cause silent failures. Clear the runtime when changing embedders (see below).

---

## Architecture: what each backend does

### HyperRAG
Extracts entities and relationships from chunks via the LLM, builds a hypergraph where hyperedges connect N entities at once. Query: keyword extraction → entity + relationship vector search → hypergraph traversal → context → answer. Best for relational questions.

### HyperRAG-lite
Same hypergraph, entity-only retrieval (no relationship hop). Shares the index with HyperRAG — index once, compare both.

### HierarchicalRAG
RAPTOR-style: clusters chunks by embedding similarity, LLM-summarizes each cluster, recurses into a tree. Query: collapsed-tree retrieval searches all levels at once. Best for synthesis/overview questions.

---

## Swapping the HierarchicalRAG design

The `RAGBackend` ABC is only four methods: `index`, `query`, `get_concept_graph`, `delete_namespace`. Everything above the backend — factory, router, GUI, eval framework — calls only these. The entire RAPTOR interior (`_greedy_clusters`, the tree-build loop, `SUMMARY_PROMPT`, `_embed_all`) is private to `hierarchical_backend.py`.

**Consequence:** you can replace the hierarchical algorithm wholesale (different clustering, different tree shape, a newer paper's method) by rewriting only that one file. Nothing else changes.

Two caveats:
- **`query_with_provenance` is not on the ABC.** The eval framework's provenance capture (`eval/provenance.py`) reaches into backend internals to record what was retrieved. For HierarchicalRAG it records tree nodes and levels; for HyperRAG it uses upstream's `return_type="json"` path (see below). If you change the hierarchical internals, update `hierarchical_query_with_provenance` alongside.
- **Storage namespace `tree_nodes`.** The summary tree persists under the `tree_nodes` namespace (→ the `hierarchical` schema in Postgres). A structurally different redesign should wipe `hyperscholar_runtime/hierarchical/<namespace>/` before re-indexing to avoid mixing incompatible node shapes.

---

## Provenance capture (how the eval framework sees inside a query)

Both backends expose what they retrieved, without any upstream HyperRAG modification:

**HyperRAG** — upstream `hyper_query` already assembles an entity/hyperedge/text-unit bundle (`contextJson`) and returns the whole dict instead of the bare answer string when `QueryParam(return_type="json")` is set. `eval/provenance.py::hyperrag_query_with_provenance` flips that flag and normalizes the result to `{answer, ok, provenance:{entities, hyperedges, text_units}}`.

**HierarchicalRAG** — `eval/provenance.py::hierarchical_query_with_provenance` mirrors the backend's collapsed-tree query but records which tree nodes (and which levels) were touched, plus the leaf chunks: `{answer, ok, provenance:{nodes_accessed, chunks_accessed, levels_accessed}}`.

Both shapes are written per-question into `answers.json` so the retrieval path behind every answer is inspectable and later visualizable (hypergraph for HyperRAG, tree for HierarchicalRAG).

---

## Evaluation framework

Reproduces the iMoonLab Hyper-RAG comparison protocol: chunk-anchored question generation → dual-backend answering with provenance → blind LLM-as-judge on five metrics → aggregated report.

### Metrics (1–10, higher is better)

| Metric | Measures |
|--------|----------|
| Comprehensiveness | Coverage of all relevant aspects |
| Diversity | Variety and richness of perspective |
| Empowerment | How well it helps the reader understand/act |
| Logical | Soundness and coherence of reasoning |
| Readability | Clarity and structure |

The judge sees both answers blind (no backend labels) with A/B order randomized per question to cancel position bias.

### Workflow

**Step 0 — index the corpus headlessly (run overnight for large ones):**

```powershell
python -m eval.preindex --corpus neurology ^
    --file D:\Datasets\neurology\neurology.jsonl --backend both
```

**Step 1 — run the whole pipeline:**

```powershell
python -m eval.run_all --corpus neurology --domain medicine --n 50
```

This runs: corpus structure export → question generation → dual-backend answering → judging → report.

**Or run stages individually:**

```powershell
python -m eval.corpus_export --corpus neurology --backend hyperrag
python -m eval.corpus_export --corpus neurology --backend hierarchical
python -m eval.question_generator --corpus neurology --domain medicine --n 50
python -m eval.runner --corpus neurology
python -m eval.judge --corpus neurology
python -m eval.report                         # all corpora, with cross-corpus summary
```

### Outputs (per corpus, under `eval/results/<corpus>/`)

| File | Contents |
|------|----------|
| `corpus_hyperrag.json` / `.md` | Every entity + hyperedge in the corpus hypergraph |
| `corpus_hierarchical.json` / `.md` | The full summary tree (all levels) |
| `questions.json` | N questions, each with its source chunk id |
| `answers.json` | Both backends' answers + provenance per question |
| `eval_results.json` | Per-question scores + aggregate means + win counts |

And `eval/results/eval_report.md` — the cross-corpus comparison tables.

### Planned corpora

neurology, physics, agriculture (and more from the iMoonLab set — medicine, mathematics, finance, law, art). Each indexed under a namespace equal to its corpus name.

### JSON shapes (for the later visualizer)

`eval_results.json` per-question entry:
```json
{
  "id": 1,
  "question": "...",
  "source_chunk_id": "chunk-...",
  "hyperrag":     { "answer": "...", "ok": true,
                    "provenance": { "entities": [...], "hyperedges": [...], "text_units": [...] } },
  "hierarchical": { "answer": "...", "ok": true,
                    "provenance": { "nodes_accessed": [...], "chunks_accessed": [...], "levels_accessed": [0,1,2] } },
  "scores": { "hyperrag": {...,"mean":8.0}, "hierarchical": {...,"mean":6.2}, "winner": "hyperrag" }
}
```

The provenance blocks are the visualizer feed: a force-directed hypergraph for HyperRAG (entities as nodes, hyperedges as halos grouping ≥2 nodes), a collapsible tree for HierarchicalRAG (root summary → sub-clusters → leaf chunks, highlighting the matched path).

---

## Storage

Currently `memory://` (RAM, wiped on GUI close). Phase 2 adds PostgreSQL persistence; `db/schema.sql` already defines `shared` / `hyperrag` / `hierarchical` schemas (the last holding `tree_nodes`).

```powershell
$env:HYPERSCHOLAR_DSN = "postgresql://localhost/hyperscholar"
```

---

## Clearing the runtime (when switching corpora or embedders)

```powershell
Remove-Item -Recurse -Force D:\Projects\hyperbolic\hyperscholar\hyperscholar_runtime
```

Required when: changing embedding model (dimension mismatch), switching corpora under the same namespace, or recovering from a half-finished index. Caches are isolated per `backend/<namespace>/`.

---

## Running tests

```powershell
python -m pytest tests/test_phase1.py -v
```

16 tests: storage contracts, router strategy, backend conformance, shared content across backends.

---

## Known issues

- **Upstream empty-retrieval crash:** `_build_relation_query_context` returns `None` on a miss; the `HyperRAGBackend` wrapper catches the resulting `AttributeError` and returns `ok=False`.
- **Memory store doesn't persist** between sessions (fixed in Phase 2).
- **Scanned PDFs** (no text layer) are skipped with a warning.
- **Streamlit empty-label warnings** at `gui.py` lines ~428/518/545 — cosmetic; add `label_visibility="collapsed"` to silence.
- **RTX 50-series silent index exit** — fixed by the CUDA 12.8 nightly (see GPU setup); the underlying `no kernel image` CUDA error is otherwise swallowed by HyperRAG's try/finally.

---

## Next

- **Phase 2:** PostgreSQL persistence (no re-indexing), institutions/users, classroom access codes, student profiles.
- **Visualizer:** plug `answers.json` provenance into a side panel — hypergraph for HyperRAG, tree for HierarchicalRAG.
- **Eval at scale:** index + evaluate neurology, physics, agriculture, then the remaining iMoonLab domains.

Full roadmap: `HYPERSCHOLAR_ROADMAP.md`.
