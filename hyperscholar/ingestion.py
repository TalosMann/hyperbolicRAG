"""
hyperscholar/ingestion.py

Corpus ingestion: turns files on disk into lists of Document objects
ready for backend.index().

Supported sources:
  - PDF files (pdfplumber, text-layer; falls back gracefully on scans)
  - Plain text files (.txt, .md)
  - iMoonLab dataset JSON format (list of {"content": "...", "title": "..."})
  - JSONL files (one JSON object per line)
  - Folder of any of the above (recursive optional)

iMoonLab NeurologyCorp format
------------------------------
Their Step_0.py preprocesses the raw dataset into a JSON file like:
  [
    {"title": "...", "content": "..."},
    ...
  ]
or sometimes a plain list of strings. Both are handled.

Usage
-----
    from hyperscholar.ingestion import load_corpus
    docs = load_corpus("path/to/neurology.json")
    docs = load_corpus("path/to/papers/")      # folder of PDFs/txts
    docs = load_corpus(["a.pdf", "b.txt"])     # explicit list of files
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

from .core.types import CorpusCategory, Document

# ── PDF ──────────────────────────────────────────────────────────────────────

def _read_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber.
    Returns empty string if no text layer (scanned PDF)."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  [warn] PDF read failed for {path}: {e}")
        return ""


# ── single-file loaders ───────────────────────────────────────────────────────

def _load_file(path: str, category: CorpusCategory = "general") -> list[Document]:
    """Load a single file → list of Documents (usually one, sometimes many)."""
    p = Path(path)
    ext = p.suffix.lower()
    title = p.stem

    if ext == ".pdf":
        content = _read_pdf(path)
        if not content.strip():
            print(f"  [warn] No text extracted from {p.name} (scanned PDF?)")
            return []
        return [Document(content=content, title=title, category=category)]

    if ext in (".txt", ".md"):
        content = p.read_text(encoding="utf-8", errors="replace")
        return [Document(content=content, title=title, category=category)]

    if ext == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        return _parse_json_corpus(raw, category, source=str(p))

    if ext == ".jsonl":
        docs = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.extend(_parse_json_corpus(obj if isinstance(obj, list) else [obj],
                                           category, source=f"{p}:{i}"))
        return docs

    print(f"  [skip] Unsupported file type: {p.name}")
    return []


def _parse_json_corpus(raw, category: CorpusCategory, source: str) -> list[Document]:
    """Handle the various JSON shapes iMoonLab and others use."""
    docs = []

    # plain list of strings
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        for i, s in enumerate(raw):
            if s.strip():
                docs.append(Document(content=s, title=f"doc_{i}", category=category))
        return docs

    # list of objects
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                docs.append(Document(content=item, title="", category=category))
            elif isinstance(item, dict):
                content = (item.get("content") or item.get("context") or item.get("text") or
                            item.get("passage") or item.get("abstract") or "")
                title   = (item.get("title") or item.get("name") or
                           item.get("id") or "")
                if content.strip():
                    docs.append(Document(content=content, title=str(title),
                                         category=category))
        return docs

    # single object with a "corpus" or "documents" key
    if isinstance(raw, dict):
        for key in ("corpus", "documents", "data", "passages", "texts"):
            if key in raw:
                return _parse_json_corpus(raw[key], category, source)
        # single document object
        content = (raw.get("content") or raw.get("context") or raw.get("text") or
                    raw.get("passage") or raw.get("abstract") or "")
        title   = raw.get("title") or raw.get("name") or ""
        if content.strip():
            docs.append(Document(content=content, title=str(title), category=category))

    return docs


# ── public API ────────────────────────────────────────────────────────────────

def load_corpus(
    source: Union[str, list[str]],
    category: CorpusCategory = "general",
    recursive: bool = False,
) -> list[Document]:
    """
    Load a corpus from one of:
      - a single file path (PDF / TXT / MD / JSON / JSONL)
      - a folder path (all supported files inside it)
      - a list of file paths

    Args:
        source:    file path, folder path, or list of file paths
        category:  HyperScholar corpus category tag
        recursive: if source is a folder, scan sub-folders too

    Returns:
        list of Document objects ready for backend.index()
    """
    SUPPORTED = {".pdf", ".txt", ".md", ".json", ".jsonl"}
    docs: list[Document] = []

    # list of files
    if isinstance(source, list):
        for f in source:
            docs.extend(_load_file(f, category))
        return docs

    p = Path(source)

    # single file
    if p.is_file():
        return _load_file(str(p), category)

    # folder
    if p.is_dir():
        pattern = "**/*" if recursive else "*"
        files = sorted(p.glob(pattern))
        for f in files:
            if f.is_file() and f.suffix.lower() in SUPPORTED:
                batch = _load_file(str(f), category)
                docs.extend(batch)
                if batch:
                    print(f"  loaded {len(batch)} doc(s) from {f.name}")
        return docs

    raise FileNotFoundError(f"Source not found: {source}")


def corpus_summary(docs: list[Document]) -> str:
    """One-line summary of a loaded corpus."""
    if not docs:
        return "Empty corpus."
    total_chars = sum(len(d.content) for d in docs)
    avg = total_chars // len(docs)
    categories = {}
    for d in docs:
        categories[d.category] = categories.get(d.category, 0) + 1
    cat_str = ", ".join(f"{v}× {k}" for k, v in categories.items())
    return (f"{len(docs)} document(s) · {total_chars:,} chars total "
            f"· avg {avg:,} chars/doc · [{cat_str}]")
