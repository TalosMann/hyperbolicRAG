"""HyperScholar — adaptive, hallucination-free learning platform.

Phase 1: swappable RAG abstraction layer (HyperRAG / HyperRAG-light /
HierarchicalRAG) over shared content + per-backend structure storage.
"""
import os
import sys
from pathlib import Path

# Make `hyperrag` importable — Hyper-RAG/ sits one level above hyperscholar/.
# Override with HYPERRAG_PATH env var if your layout differs.
_hyper_rag_path = os.environ.get(
    "HYPERRAG_PATH",
    str(Path(__file__).resolve().parent.parent / "Hyper-RAG")
)
if _hyper_rag_path not in sys.path:
    sys.path.insert(0, _hyper_rag_path)

__version__ = "0.1.0"