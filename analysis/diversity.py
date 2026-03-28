# analysis/diversity.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def diversity_score(prompts: list[str]) -> float:
    """
    Mean pairwise cosine distance between prompt embeddings.
    0.0 = identical, 1.0 = maximally diverse.
    """
    if len(prompts) < 2:
        return 0.0

    embedder   = _get_embedder()
    embeddings = embedder.encode(prompts, show_progress_bar=False)
    distances  = cosine_distances(embeddings)

    n    = len(prompts)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(distances[mask].mean())