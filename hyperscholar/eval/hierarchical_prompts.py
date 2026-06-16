"""eval/hierarchical_prompts.py

Mirror of the prompt constants in hierarchical_backend.py, imported by the
provenance wrapper so the answer-generation path stays identical to production.
Keep these in sync if the backend's prompts change.
"""

ANSWER_PROMPT = """Answer the question using ONLY the context below. If the context does not contain the answer, reply exactly: "{fail}"

CONTEXT:
{context}

QUESTION: {question}
"""

FAIL = "Sorry, I'm not able to provide an answer to that question."
