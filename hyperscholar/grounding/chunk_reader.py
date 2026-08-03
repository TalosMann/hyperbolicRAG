r"""chunk_reader — inspect the raw source-chunk store the grounding layer reads.

The chunks live as plain JSON at
  hyperscholar_runtime/cograg_official/<namespace>/kv_store_text_chunks.json

Commands (run from repo root):

  # list corpora and chunk counts
  python -m hyperscholar.grounding.chunk_reader list

  # corpus stats
  python -m hyperscholar.grounding.chunk_reader stats --ns agriculture_20

  # keyword / regex search over raw chunk text (instant, no model)
  python -m hyperscholar.grounding.chunk_reader grep "honey extraction" --ns agriculture_20 -i

  # dump one chunk in full by id (prefix ok)
  python -m hyperscholar.grounding.chunk_reader show chunk-0a5bf926 --ns agriculture_20

  # SEMANTIC search: the top-k chunks the grounding layer would actually retrieve
  # (loads bge-m3; this is what `grounded_answer` grounds against)
  python -m hyperscholar.grounding.chunk_reader search "how is honey extracted" --ns agriculture_20 -k 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent.parent / "hyperscholar_runtime" / "cograg_official"

# allow running the file directly (python chunk_reader.py ...) not just `-m`:
# put the repo root on sys.path so the `search` command can import hyperscholar.*
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _store(ns: str) -> Path:
    return RUNTIME / ns / "kv_store_text_chunks.json"


def _load(ns: str) -> dict:
    p = _store(ns)
    if not p.exists():
        sys.exit(f"no chunk store for namespace '{ns}'\n  expected: {p}\n"
                 f"  (run `... chunk_reader list` to see available namespaces)")
    return json.loads(p.read_text(encoding="utf-8"))


def _oneline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cmd_list(_args):
    if not RUNTIME.exists():
        sys.exit(f"no runtime dir at {RUNTIME}")
    print(f"{'namespace':24s} {'chunks':>8s}  {'documents':>9s}")
    print("-" * 46)
    for d in sorted(RUNTIME.iterdir()):
        p = d / "kv_store_text_chunks.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        docs = {c.get("full_doc_id") for c in data.values()}
        print(f"{d.name:24s} {len(data):>8d}  {len(docs):>9d}")


def cmd_stats(args):
    d = _load(args.ns)
    docs = {c.get("full_doc_id") for c in d.values()}
    toks = sum(c.get("tokens", 0) for c in d.values())
    chars = sum(len(c.get("content", "")) for c in d.values())
    print(f"namespace : {args.ns}")
    print(f"chunks    : {len(d):,}")
    print(f"documents : {len(docs):,}")
    print(f"tokens    : ~{toks:,}")
    print(f"chars     : ~{chars:,}")
    print(f"store     : {_store(args.ns)}")


def cmd_grep(args):
    d = _load(args.ns)
    rx = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
    ctx, shown, matched = args.context, 0, 0
    for cid, c in d.items():
        content = c.get("content", "")
        m = rx.search(content)
        if not m:
            continue
        matched += 1
        if shown >= args.limit:
            continue
        shown += 1
        s, e = max(0, m.start() - ctx), min(len(content), m.end() + ctx)
        snip = _oneline(content[s:e])
        # mark the hit
        hit = content[m.start():m.end()]
        snip = snip.replace(_oneline(hit), f">>>{_oneline(hit)}<<<", 1)
        print(f"[{cid[:22]}  doc={str(c.get('full_doc_id',''))[:14]}  #{c.get('chunk_order_index','?')}]")
        print(f"  …{snip}…\n")
    print(f"{matched} chunk(s) matched"
          + (f"; showing first {args.limit}" if matched > args.limit else ""))


def cmd_show(args):
    d = _load(args.ns)
    matches = [k for k in d if k.startswith(args.id) or k == args.id
               or k[len("chunk-"):].startswith(args.id)]
    if not matches:
        sys.exit(f"no chunk id starting with '{args.id}' in {args.ns}")
    if len(matches) > 1:
        print(f"{len(matches)} chunks match prefix '{args.id}':")
        for k in matches[:20]:
            print(f"  {k}")
        return
    c = d[matches[0]]
    print(f"id     : {matches[0]}")
    print(f"doc    : {c.get('full_doc_id')}")
    print(f"order  : {c.get('chunk_order_index')}   tokens: {c.get('tokens')}")
    print("-" * 70)
    print(c.get("content", ""))


async def _search(args):
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.rag.cograg_official_backend import CogRAGOfficialBackend

    cfg = load_config()

    async def _noop(*a, **k):
        return ""

    be = CogRAGOfficialBackend(llm_func=_noop, embedder=build_embedder(cfg.embedding),
                               working_dir=cfg.working_dir, query_mode="cog-entity",
                               fail_markers=cfg.rag.fail_markers)
    rag = be._rag(args.ns)
    hits = await rag.chunks_vdb.query(args.query, top_k=args.k)
    if not hits:
        print("no hits")
        return
    chunks = await rag.text_chunks.get_by_ids([h["id"] for h in hits])
    print(f"top {len(hits)} chunks the grounding layer would retrieve for:")
    print(f"  \"{args.query}\"\n")
    for rank, (h, c) in enumerate(zip(hits, chunks), 1):
        content = (c or {}).get("content", "")
        print(f"{rank}. [{h['id'][:22]}  score={h.get('distance', 0):.3f}  "
              f"doc={str((c or {}).get('full_doc_id',''))[:14]}]")
        print(f"   {_oneline(content)[:args.chars]}…\n")


def cmd_search(args):
    asyncio.run(_search(args))


def main():
    ap = argparse.ArgumentParser(prog="chunk_reader",
                                 description="Inspect the raw source-chunk store.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list namespaces + chunk counts").set_defaults(fn=cmd_list)

    p = sub.add_parser("stats", help="chunk/doc counts for a namespace")
    p.add_argument("--ns", required=True)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("grep", help="regex search over raw chunk text (no model)")
    p.add_argument("pattern")
    p.add_argument("--ns", required=True)
    p.add_argument("-i", "--ignore-case", action="store_true")
    p.add_argument("--context", type=int, default=120, help="chars of context around match")
    p.add_argument("--limit", type=int, default=20, help="max chunks to show")
    p.set_defaults(fn=cmd_grep)

    p = sub.add_parser("show", help="dump one chunk in full (id or prefix)")
    p.add_argument("id")
    p.add_argument("--ns", required=True)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("search", help="semantic top-k (what the grounding layer retrieves)")
    p.add_argument("query")
    p.add_argument("--ns", required=True)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--chars", type=int, default=300, help="preview chars per chunk")
    p.set_defaults(fn=cmd_search)

    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        ap.print_help()
        return
    args.fn(args)


if __name__ == "__main__":
    main()
