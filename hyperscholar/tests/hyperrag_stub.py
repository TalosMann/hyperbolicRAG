"""HyperRAG-aware stub LLM.

Drives the *real* HyperRAG pipeline offline by recognising its prompt types
and producing structurally valid responses:

  entity extraction → canned Entity / Low-order / High-order Hyperedge records
                      using the live delimiters from hyperrag.prompt.PROMPTS
  keyword extraction → valid JSON with high/low-level keywords
  RAG answer        → grounded-looking answer that embeds a context fingerprint
  anything else     → generic summary text

This lets the conformance suite exercise chunking → extraction → hypergraph →
retrieval → answer with zero network calls.
"""
from __future__ import annotations

import json


class HyperRAGStubLLM:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompt, system_prompt=None, history_messages=[],
                       hashing_kv=None, **kwargs):
        self.calls.append(prompt[:80])
        from hyperrag.prompt import PROMPTS
        tup = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        rec = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        done = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        text = prompt.lower()

        # keyword extraction → strict JSON
        if "high_level_keywords" in text and "low_level_keywords" in text:
            return json.dumps({
                "high_level_keywords": ["Biology", "Energy conversion"],
                "low_level_keywords": ["photosynthesis", "chlorophyll",
                                        "mitochondria", "glucose"],
            })

        # entity extraction → records grounded in whichever chunk is present
        if "-goal-" in text and "identify all entities" in text:
            recs = []
            if "photosynthesis" in text or "chlorophyll" in text:
                recs += [
                    f'("Entity"{tup}PHOTOSYNTHESIS{tup}concept{tup}Process converting sunlight to chemical energy in plants{tup}domain: biology)',
                    f'("Entity"{tup}CHLOROPHYLL{tup}concept{tup}Pigment in chloroplasts that absorbs light{tup}domain: biology)',
                    f'("Entity"{tup}GLUCOSE{tup}concept{tup}Sugar produced by photosynthesis storing chemical energy{tup}domain: biology)',
                    f'("Low-order Hyperedge"{tup}PHOTOSYNTHESIS{tup}CHLOROPHYLL{tup}Chlorophyll drives photosynthesis by absorbing light{tup}light absorption{tup}9)',
                    f'("High-level keywords"{tup}energy conversion, plant biology)',
                    f'("High-order Hyperedge"{tup}PHOTOSYNTHESIS{tup}CHLOROPHYLL{tup}GLUCOSE{tup}Light absorbed by chlorophyll powers photosynthesis which produces glucose{tup}photosynthetic energy pathway{tup}energy flow{tup}9)',
                ]
            if "mitochondria" in text or "respiration" in text:
                recs += [
                    f'("Entity"{tup}CELLULAR RESPIRATION{tup}concept{tup}Process releasing energy stored in glucose to make ATP{tup}domain: biology)',
                    f'("Entity"{tup}MITOCHONDRIA{tup}organelle{tup}Organelle where cellular respiration occurs{tup}domain: biology)',
                    f'("Entity"{tup}ATP{tup}molecule{tup}Energy currency of the cell produced by respiration{tup}domain: biology)',
                    f'("Low-order Hyperedge"{tup}CELLULAR RESPIRATION{tup}MITOCHONDRIA{tup}Respiration takes place inside the mitochondria{tup}location{tup}9)',
                    f'("High-level keywords"{tup}cellular energy, metabolism)',
                    f'("High-order Hyperedge"{tup}CELLULAR RESPIRATION{tup}MITOCHONDRIA{tup}ATP{tup}Mitochondria host respiration which produces ATP{tup}cellular energy production{tup}metabolism{tup}9)',
                ]
            if not recs:
                recs = [f'("Entity"{tup}TOPIC{tup}concept{tup}A topic from the corpus{tup}n/a)']
            return rec.join(recs) + rec + done

        # gleaning continuation → nothing more to add
        if "many entities were missed" in text or "add them below" in text:
            return done

        # summarization prompts
        if "summar" in text and "passages" not in text:
            return "Combined description of the concept drawn from the corpus."

        # final RAG answer (context-bearing prompt)
        return "[grounded-answer] Plants convert sunlight via chlorophyll-driven photosynthesis; respiration in mitochondria yields ATP."
