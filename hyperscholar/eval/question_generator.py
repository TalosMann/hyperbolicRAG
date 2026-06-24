r"""eval/question_generator.py

Generates evaluation questions in FIVE styles, each targeting a different
retrieval capability so the comparison isn't dominated by one question type:

  fact        - "what does this specific passage say" (random chunk → Q)
                Favors HyperRAG: direct entity/fact lookup.
  relational  - "how do X and Y relate" (sampled from a real hyperedge)
                Favors HyperRAG: hypergraph's signature n-ary capability.
  synthesis   - "how is topic X treated across the whole corpus"
                (sampled from a high-degree entity, i.e. a recurring topic)
                Favors HierarchicalRAG: collapsed-tree synthesis is built for this.
  overview    - fixed corpus-level prompts, not anchored to any one chunk/topic
                Favors HierarchicalRAG: narrative coherence at scale.
  negative    - asks for a specific detail that sounds plausible but is NOT
                stated in the source passage. Scored separately in judge.py
                by refusal rate, not the 5-metric rubric — this tests whether
                each backend admits "not in the corpus" rather than fabricating.
                CAVEAT: a non-refusal here is not proof of hallucination, only
                that the canned fail-marker wasn't triggered. Spot-check
                non-refused negative answers manually if you need a true
                hallucination rate.

Questions accumulate in questions.json — each call to generate_questions()
APPENDS new questions (with auto-incrementing ids) rather than overwriting,
so you can build a mixed-style set by calling this multiple times with
different --style values.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.question_generator --corpus neurology --style fact --n 15 --domain medicine
    python -m hyperscholar.eval.question_generator --corpus neurology --style relational --n 10 --domain medicine
    python -m hyperscholar.eval.question_generator --corpus neurology --style synthesis --n 10 --domain medicine
    python -m hyperscholar.eval.question_generator --corpus neurology --style overview --n 5 --domain medicine
    python -m hyperscholar.eval.question_generator --corpus neurology --style negative --n 5 --domain medicine

Prereqs:
  fact, negative      - corpus indexed (text_chunks store must exist)
  relational, synthesis - corpus_export.py --backend hyperrag already run
                          (reads corpus_hyperrag.json for entities/hyperedges)
  overview            - no prerequisite, pure templates
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

# ── prompts ───────────────────────────────────────────────────────────────────

FACT_PROMPT = """You are creating an exam question to test understanding of a specific passage from a {domain} corpus.

Read the passage below and write ONE clear, self-contained question whose answer is found in the passage. The question must:
- be answerable from the passage alone
- not reference "the passage" or "the text" (ask about the subject matter directly)
- require understanding, not just keyword matching
- be a single sentence

PASSAGE:
{passage}

Output ONLY the question, nothing else."""

RELATIONAL_PROMPT = """You are creating an exam question to test understanding of how specific concepts relate to each other within a {domain} corpus.

These concepts are linked by a documented relationship in the corpus: {entities}
Relationship as described in the source material: {description}

Write ONE clear question that asks how these concepts relate or interact. The question must:
- name the specific concepts involved
- require understanding of the connection between them, not just a definition of one
- be answerable using the relationship described above
- be a single sentence

Output ONLY the question, nothing else."""

SYNTHESIS_PROMPT = """You are creating an exam question that requires synthesizing information about a recurring topic across an ENTIRE {domain} corpus, not just one passage.

Topic that recurs throughout the corpus: {topic}

Write ONE question that asks for a synthesis or overview of how this topic is treated across the whole corpus — for example "what are the different ways X is discussed, categorized, or treated across this text?" The question must:
- require drawing together information from multiple places, not a single fact
- name the topic specifically
- be a single sentence

Output ONLY the question, nothing else."""

OVERVIEW_TEMPLATES = [
    "What is the overall structure and scope of this {domain} text?",
    "What are the major themes or categories covered across this {domain} corpus?",
    "How does this {domain} text approach the relationship between theory and practice?",
    "What recurring concerns or perspectives appear throughout this {domain} corpus?",
    "What would a reader unfamiliar with the subject learn about the field of {domain} from this corpus as a whole?",
    "How does this {domain} corpus organize its subject matter into subareas or specialties?",
    "What philosophy or approach toward the field of {domain} does this corpus seem to advocate?",
]

NEGATIVE_PROMPT = """You are creating a question to test whether an AI system correctly admits it doesn't know something, rather than making up an answer.

Below is a real passage from a {domain} corpus:

PASSAGE:
{passage}

Write ONE question that:
- sounds plausible and closely related to the passage's topic
- asks for a SPECIFIC fact, number, name, date, or detail that is plausible-sounding but is NOT actually stated anywhere in this passage — invent a specific but absent detail to ask about (an exact statistic, a specific date, a named study, a specific dosage, etc.)
- is a single sentence

Output ONLY the question, nothing else. Do not explain what's missing."""


# ── per-style generators (each returns a list of dicts, no "id" yet) ─────────

async def _generate_fact(llm, domain: str, n: int, seed: int,
                         results_dir: Path, corpus: str, namespace: str,
                         working_dir: str) -> list[dict]:
    from hyperrag.storage import JsonKVStorage

    workdir = os.path.join(working_dir, "hyperrag", namespace)
    gcfg = {"working_dir": workdir, "addon_params": {}, "embedding_batch_num": 8}
    chunks = JsonKVStorage(namespace="text_chunks", global_config=gcfg)
    chunk_ids = await chunks.all_keys()
    if not chunk_ids:
        raise RuntimeError(
            f"No indexed chunks for namespace '{namespace}'. Index the corpus first.")

    rng = random.Random(seed)
    sample_ids = rng.sample(chunk_ids, min(n, len(chunk_ids)))
    rows = await chunks.get_by_ids(sample_ids)

    out = []
    for cid, row in zip(sample_ids, rows):
        content = (row or {}).get("content", "")
        if not content.strip():
            continue
        q = await llm(FACT_PROMPT.format(domain=domain, passage=content[:2000]))
        q = (q or "").strip().strip('"')
        if q:
            out.append({
                "style": "fact",
                "question": q,
                "source_chunk_id": cid,
                "source_excerpt": content[:500],
            })
        print(f"  [fact {len(out)}/{n}] {q[:80]}")
    return out


def _load_hyperrag_export(results_dir: Path, corpus: str) -> dict:
    p = results_dir / corpus / "corpus_hyperrag.json"
    if not p.exists():
        raise RuntimeError(
            f"{p} not found. Run "
            f"`python -m hyperscholar.eval.corpus_export --corpus {corpus} "
            f"--backend hyperrag` first — relational/synthesis styles need it.")
    return json.loads(p.read_text(encoding="utf-8"))


async def _generate_relational(llm, domain: str, n: int, seed: int,
                               results_dir: Path, corpus: str) -> list[dict]:
    export = _load_hyperrag_export(results_dir, corpus)
    edges = [e for e in export.get("hyperedges", [])
             if e.get("description") and e.get("entity_set")]
    if not edges:
        print("  [relational] no hyperedges with descriptions found, skipping")
        return []

    rng = random.Random(seed + 1)
    sample = rng.sample(edges, min(n, len(edges)))

    out = []
    for e in sample:
        entities = e["entity_set"] if isinstance(e["entity_set"], list) else [str(e["entity_set"])]
        q = await llm(RELATIONAL_PROMPT.format(
            domain=domain, entities=", ".join(entities),
            description=e["description"][:600]))
        q = (q or "").strip().strip('"')
        if q:
            out.append({
                "style": "relational",
                "question": q,
                "source_hyperedge_id": e.get("id", ""),
                "source_entities": entities,
                "source_description": e["description"][:400],
            })
        print(f"  [relational {len(out)}/{n}] {q[:80]} "
              f"(entities: {', '.join(entities[:3])})")
    return out


async def _generate_synthesis(llm, domain: str, n: int, seed: int,
                              results_dir: Path, corpus: str) -> list[dict]:
    export = _load_hyperrag_export(results_dir, corpus)
    entities = export.get("entities", [])  # already sorted by degree desc
    if not entities:
        print("  [synthesis] no entities found, skipping")
        return []

    # Sample from the highest-degree pool — these are topics that recur
    # across many chunks, which is what makes a synthesis question meaningful.
    pool_size = max(1, min(50, len(entities)))
    pool = entities[:pool_size]

    rng = random.Random(seed + 2)
    sample = rng.sample(pool, min(n, len(pool)))

    out = []
    for ent in sample:
        topic = ent["id"]
        q = await llm(SYNTHESIS_PROMPT.format(domain=domain, topic=topic))
        q = (q or "").strip().strip('"')
        if q:
            out.append({
                "style": "synthesis",
                "question": q,
                "source_topic": topic,
                "source_degree": ent.get("degree", 0),
            })
        print(f"  [synthesis {len(out)}/{n}] {q[:80]} (topic: {topic})")
    return out


def _generate_overview(domain: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 3)
    templates = OVERVIEW_TEMPLATES.copy()
    rng.shuffle(templates)
    chosen = templates[:n] if n <= len(templates) else templates * (n // len(templates) + 1)
    chosen = chosen[:n]
    out = []
    for t in chosen:
        q = t.format(domain=domain)
        out.append({"style": "overview", "question": q, "source": "corpus-level"})
        print(f"  [overview {len(out)}/{n}] {q[:80]}")
    return out


async def _generate_negative(llm, domain: str, n: int, seed: int,
                             namespace: str, working_dir: str) -> list[dict]:
    from hyperrag.storage import JsonKVStorage

    workdir = os.path.join(working_dir, "hyperrag", namespace)
    gcfg = {"working_dir": workdir, "addon_params": {}, "embedding_batch_num": 8}
    chunks = JsonKVStorage(namespace="text_chunks", global_config=gcfg)
    chunk_ids = await chunks.all_keys()
    if not chunk_ids:
        raise RuntimeError(
            f"No indexed chunks for namespace '{namespace}'. Index the corpus first.")

    rng = random.Random(seed + 4)
    sample_ids = rng.sample(chunk_ids, min(n, len(chunk_ids)))
    rows = await chunks.get_by_ids(sample_ids)

    out = []
    for cid, row in zip(sample_ids, rows):
        content = (row or {}).get("content", "")
        if not content.strip():
            continue
        q = await llm(NEGATIVE_PROMPT.format(domain=domain, passage=content[:2000]))
        q = (q or "").strip().strip('"')
        if q:
            out.append({
                "style": "negative",
                "question": q,
                "source_chunk_id": cid,
                "source_excerpt": content[:500],
                "note": "Asked detail is intentionally absent from the source "
                        "passage — correct behavior is to decline, not fabricate.",
            })
        print(f"  [negative {len(out)}/{n}] {q[:80]}")
    return out


# ── merge + save (append-mode, auto-incrementing ids) ─────────────────────────

def _merge_and_save(results_dir: Path, corpus: str, namespace: str,
                    domain: str, new_items: list[dict]) -> tuple[Path, dict]:
    out_dir = results_dir / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "questions.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        questions = existing.get("questions", [])
    else:
        existing = {"corpus": corpus, "namespace": namespace, "domain": domain}
        questions = []

    next_id = max((q["id"] for q in questions), default=0) + 1
    for item in new_items:
        item["id"] = next_id
        next_id += 1
        questions.append(item)

    style_counts: dict = {}
    for q in questions:
        s = q.get("style", "fact")
        style_counts[s] = style_counts.get(s, 0) + 1

    existing["questions"] = questions
    existing["n_generated"] = len(questions)
    existing["style_counts"] = style_counts
    existing["domain"] = domain
    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path, style_counts


# ── public API ────────────────────────────────────────────────────────────────

async def generate_questions(corpus: str, n: int, namespace: str,
                             results_dir: Path, style: str = "fact",
                             domain: str = "academic", seed: int = 42) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.llm import build_llm_func

    cfg = load_config()
    llm = build_llm_func(cfg.llm)

    if style == "fact":
        new_items = await _generate_fact(llm, domain, n, seed, results_dir,
                                         corpus, namespace, cfg.working_dir)
    elif style == "relational":
        new_items = await _generate_relational(llm, domain, n, seed,
                                               results_dir, corpus)
    elif style == "synthesis":
        new_items = await _generate_synthesis(llm, domain, n, seed,
                                              results_dir, corpus)
    elif style == "overview":
        new_items = _generate_overview(domain, n, seed)
    elif style == "negative":
        new_items = await _generate_negative(llm, domain, n, seed,
                                             namespace, cfg.working_dir)
    else:
        raise ValueError(f"Unknown style: {style!r}")

    out_path, style_counts = _merge_and_save(results_dir, corpus, namespace,
                                             domain, new_items)
    print(f"\n✓ {len(new_items)} new '{style}' questions → {out_path}")
    print(f"  total by style: {style_counts}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--style", default="fact",
                    choices=["fact", "relational", "synthesis", "overview", "negative"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--domain", default="academic")
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(generate_questions(
        corpus=args.corpus, n=args.n, namespace=namespace,
        results_dir=Path(args.results_dir), style=args.style,
        domain=args.domain, seed=args.seed))


if __name__ == "__main__":
    main()
