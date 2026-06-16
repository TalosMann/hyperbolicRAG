"""HyperScholar evaluation framework.

Compares HyperRAG vs HierarchicalRAG using the iMoonLab Hyper-RAG paper's
protocol: chunk-anchored question generation, dual-backend answering with
provenance capture, blind LLM-as-judge scoring on five metrics, and aggregated
reporting.

Modules:
    preindex            headless GPU indexing of a corpus into both backends
    corpus_export       full hypergraph / summary-tree structure dumps
    question_generator  N chunk-anchored questions per corpus
    runner              answers + provenance from both backends
    judge               LLM-as-judge, 5 metrics, blind + position-randomized
    report              markdown comparison tables
    run_all             one-command orchestrator (steps 2–5)
    provenance          query wrappers that capture retrieval provenance
"""
