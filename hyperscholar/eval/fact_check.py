r"""eval/fact_check.py

Ground-truth grounding check — independent of the LLM-as-judge's stylistic
5-metric scoring. For each question style, pulls the most appropriate
available ground-truth anchor and asks an LLM fact-checker whether each
backend's answer is SUPPORTED, CONTRADICTED, or UNVERIFIABLE against it.

Ground-truth anchor used per style:
  fact         FULL source chunk (raw, independent text — pulled fresh from
               disk, not the 500-char excerpt cached in questions.json)
  relational   the sampled hyperedge's own description + entity list, from
               corpus_hyperrag.json
  synthesis    the sampled entity's own description, from corpus_hyperrag.json
  overview     SKIPPED — no single ground-truth anchor exists for a
               corpus-level question; checking against nothing would be
               meaningless, not conservative
  negative     SKIPPED — already covered by judge.py's refusal check

IMPORTANT ASYMMETRY for relational/synthesis: the ground truth there is
HyperRAG's OWN extracted hyperedge/entity description, not independent raw
text. For HyperRAG, this checks internal consistency with its own hypergraph.
For HierarchicalRAG, it checks incidental agreement with structured data it
never retrieved at all. This is a different (weaker) kind of check than the
fact-style one, which compares both backends against truly independent
source text. Treat relational/synthesis verdicts accordingly — they are
useful, but not apples-to-apples with the fact-style numbers.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.fact_check --corpus neurology

Reads:  <corpus>/answers.json, <corpus>/corpus_hyperrag.json
Writes: <corpus>/fact_check.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

FACT_CHECK_PROMPT = """You are verifying whether an answer's key factual claim is supported by a specific piece of ground-truth text.

GROUND TRUTH ({ground_truth_label} — the ONLY source of truth; ignore any outside knowledge, even if you believe it to be true):
{excerpt}

QUESTION:
{question}

ANSWER TO CHECK:
{answer}

Determine the answer's key factual claim relative to the ground truth ONLY:
- SUPPORTED: the answer's key claim is directly stated in or clearly inferable from the ground truth
- CONTRADICTED: the answer states something that conflicts with what the ground truth actually says
- UNVERIFIABLE: the ground truth does not address this specific claim either way

Respond with ONLY a JSON object in this exact form, no other text:
{{"verdict": "SUPPORTED", "explanation": "one short sentence"}}
(verdict must be exactly one of: SUPPORTED, CONTRADICTED, UNVERIFIABLE)"""


def _parse_verdict(raw: str) -> dict | None:
    if not raw:
        return None
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    v = str(obj.get("verdict", "")).upper().strip()
    if v not in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
        return None
    return {"verdict": v, "explanation": obj.get("explanation", "")}


async def _full_chunk_content(namespace: str, working_dir: str, chunk_id: str) -> str:
    """Pull the FULL chunk text from the HyperRAG text_chunks store — not the
    truncated excerpt cached in questions.json."""
    from hyperrag.storage import JsonKVStorage
    workdir = os.path.join(working_dir, "hyperrag", namespace)
    gcfg = {"working_dir": workdir, "addon_params": {}, "embedding_batch_num": 8}
    chunks = JsonKVStorage(namespace="text_chunks", global_config=gcfg)
    row = await chunks.get_by_id(chunk_id)
    return (row or {}).get("content", "")


def _load_hyperrag_export(results_dir: Path, corpus: str) -> dict | None:
    p = results_dir / corpus / "corpus_hyperrag.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _ground_truth_for(item: dict, export: dict | None) -> tuple[str, str] | None:
    """Returns (ground_truth_text, ground_truth_label) or None if unavailable."""
    style = item.get("style", "fact")

    if style == "relational":
        if export is None:
            return None
        hid = item.get("source_hyperedge_id")
        edge = next((e for e in export.get("hyperedges", []) if e.get("id") == hid), None)
        if edge is None:
            return None
        entities = edge.get("entity_set", [])
        entities_str = ", ".join(entities) if isinstance(entities, list) else str(entities)
        text = f"Entities: {entities_str}\nRelationship description: {edge.get('description', '')}"
        return text, "extracted hyperedge description, from HyperRAG's own hypergraph"

    if style == "synthesis":
        if export is None:
            return None
        topic = item.get("source_topic")
        ent = next((e for e in export.get("entities", []) if e.get("id") == topic), None)
        if ent is None:
            return None
        text = f"Entity: {ent.get('id', '')}\nDescription: {ent.get('description', '')}"
        return text, "extracted entity description, from HyperRAG's own hypergraph"

    return None  # fact handled separately (needs async chunk fetch); overview/negative skipped


async def fact_check_corpus(corpus: str, namespace: str, results_dir: Path) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.llm import build_llm_func

    cfg = load_config()
    llm = build_llm_func(cfg.llm)

    ans_path = results_dir / corpus / "answers.json"
    if not ans_path.exists():
        raise FileNotFoundError(f"{ans_path} not found. Run runner first.")
    data = json.loads(ans_path.read_text(encoding="utf-8"))

    export = _load_hyperrag_export(results_dir, corpus)
    if export is None:
        print("  [note] corpus_hyperrag.json not found — relational/synthesis "
              "questions will be skipped. Run corpus_export --backend hyperrag first "
              "to enable them.")

    out_path = results_dir / corpus / "fact_check.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        checked = existing.get("checks", [])
        done_ids = {c["id"] for c in checked}
        print(f"  resuming — {len(done_ids)} already checked")
    else:
        checked = []
        done_ids = set()

    skipped_styles = {"overview": 0, "negative": 0}
    eligible = []
    for item in data["results"]:
        style = item.get("style", "fact")
        if style in ("overview", "negative"):
            skipped_styles[style] += 1
            continue
        eligible.append(item)

    print(f"  {len(eligible)} questions eligible for ground-truth check "
          f"({len(done_ids)} already checked)")
    if skipped_styles["overview"]:
        print(f"  [skip] {skipped_styles['overview']} overview-style questions "
              f"— no single ground-truth anchor exists for a corpus-level question")
    if skipped_styles["negative"]:
        print(f"  [skip] {skipped_styles['negative']} negative-style questions "
              f"— already covered by judge.py's refusal check")

    for item in eligible:
        qid = item["id"]
        if qid in done_ids:
            continue

        style = item.get("style", "fact")

        if style == "fact":
            chunk_id = item.get("source_chunk_id")
            if not chunk_id:
                print(f"  Q{qid}: no source_chunk_id, skipping")
                continue
            ground_truth = await _full_chunk_content(namespace, cfg.working_dir, chunk_id)
            label = "source passage, as indexed"
            if not ground_truth:
                print(f"  Q{qid}: source chunk '{chunk_id}' not found on disk, skipping")
                continue
        else:
            gt = _ground_truth_for(item, export)
            if gt is None:
                print(f"  Q{qid} [{style}]: ground truth unavailable, skipping")
                continue
            ground_truth, label = gt

        result = {"id": qid, "style": style, "question": item["question"]}

        for backend in ("hyperrag", "hierarchical"):
            answer = item[backend]["answer"]
            try:
                reply = await llm(FACT_CHECK_PROMPT.format(
                    ground_truth_label=label, excerpt=ground_truth[:2500],
                    question=item["question"], answer=answer[:1500]))
                verdict = _parse_verdict(reply)
            except Exception as e:
                verdict = None
                print(f"  Q{qid} [{backend}]: fact-check error ({e})")
            result[backend] = verdict or {"verdict": "UNVERIFIABLE",
                                          "explanation": "fact-check call failed or unparseable"}

        print(f"  Q{qid} [{style}]: hyperrag={result['hyperrag']['verdict']:13s}  "
              f"hierarchical={result['hierarchical']['verdict']}")

        checked.append(result)
        out_path.write_text(json.dumps({"corpus": corpus, "checks": checked},
                                       indent=2, ensure_ascii=False), encoding="utf-8")

    # summary, broken down by style
    summary_by_style: dict = {}
    for c in checked:
        s = c.get("style", "fact")
        bucket = summary_by_style.setdefault(s, {
            "hyperrag": {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIABLE": 0},
            "hierarchical": {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIABLE": 0},
        })
        for b in ("hyperrag", "hierarchical"):
            v = c.get(b, {}).get("verdict", "UNVERIFIABLE")
            bucket[b][v] = bucket[b].get(v, 0) + 1

    overall = {"hyperrag": {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIABLE": 0},
              "hierarchical": {"SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIABLE": 0}}
    for s, bucket in summary_by_style.items():
        for b in ("hyperrag", "hierarchical"):
            for v in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
                overall[b][v] += bucket[b][v]

    out_path.write_text(json.dumps({
        "corpus": corpus, "checks": checked,
        "summary": overall,
        "summary_by_style": summary_by_style,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ fact-checked {len(checked)} answers → {out_path}")
    print(f"  Overall  HyperRAG:      {overall['hyperrag']}")
    print(f"  Overall  Hierarchical:  {overall['hierarchical']}")
    for s, bucket in summary_by_style.items():
        print(f"  [{s}]  HyperRAG: {bucket['hyperrag']}   Hierarchical: {bucket['hierarchical']}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(fact_check_corpus(args.corpus, namespace, Path(args.results_dir)))


if __name__ == "__main__":
    main()
