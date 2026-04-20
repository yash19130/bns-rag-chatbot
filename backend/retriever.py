"""
retriever.py — Query the FAISS index and return relevant chunks.
Handles the "I don't know" case via similarity threshold.
"""

import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL, FAISS_INDEX_DIR,
    TOP_K_RESULTS, MIN_SIMILARITY_SCORE
)
from backend.ingest import load_index

log = logging.getLogger(__name__)

# BGE models need this prefix on queries (not on passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self):
        self.model: SentenceTransformer = None
        self.index: faiss.Index = None
        self.chunks: list[dict] = None
        self._loaded = False

    def load(self, index_dir: Path = FAISS_INDEX_DIR):
        """Load embedding model and FAISS index. Call once at startup."""
        if self._loaded:
            return

        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        log.info(f"Loading FAISS index from: {index_dir}")
        self.index, self.chunks = load_index(index_dir)

        self._loaded = True
        log.info(f"Retriever ready. Index has {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Embed query, search FAISS, filter by similarity threshold.

        Returns list of dicts:
        {
            "text": str,
            "section": str,
            "title": str,
            "source": str,
            "score": float,
        }

        Returns empty list if no results pass the threshold
        (caller should handle this as "I don't know").
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call .load() first.")

        # BGE requires prefix on queries
        prefixed_query = BGE_QUERY_PREFIX + query

        query_vec = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < MIN_SIMILARITY_SCORE:
                continue  # below confidence threshold
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        log.debug(f"Query '{query[:60]}' → {len(results)} results above threshold")
        return results


# Singleton instance — shared across FastAPI and Streamlit
retriever = Retriever()