"""
retriever.py — Query the FAISS index and return relevant chunks.
Handles the "I don't know" case via similarity threshold.
Supports query decomposition for multi-section questions.
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

# Section 531 of BNSS is a giant schedule/table of offences — it scores
# highly on almost every query due to keyword overlap but contains no
# useful prose for the LLM to reason from. Exclude it from retrieval.
EXCLUDED_SECTIONS = {"Section 531"}

# Sub-queries for known complex question patterns.
DECOMPOSITION_MAP = [
    {
        "triggers": ["after", "procedure", "what happens", "process", "steps", "following"],
        "sub_queries": [
            "{original}",
            "arrest without warrant police officer cognizable",
            "information cognizable offence FIR police station",
            "bail non-bailable offence arrested",
            "police report charge sheet magistrate",
        ]
    },
    {
        "triggers": ["arrested", "arrest", "detained", "custody"],
        "sub_queries": [
            "{original}",
            "arrest without warrant police officer",
            "rights of arrested person informed grounds",
            "bail non-bailable offence",
            "information cognizable offence FIR police station",
        ]
    },
    {
        "triggers": ["fir", "first information report", "complaint", "filed", "lodge"],
        "sub_queries": [
            "{original}",
            "information cognizable offence officer police station",
            "investigation after information received",
            "police report charge sheet filing court",
        ]
    },
]


def decompose_query(query: str) -> list[str]:
    """
    Check if the query is a multi-section question.
    If so, return focused sub-queries. Otherwise return original as single item.
    """
    query_lower = query.lower()
    for pattern in DECOMPOSITION_MAP:
        if any(trigger in query_lower for trigger in pattern["triggers"]):
            sub_queries = [
                q.replace("{original}", query)
                for q in pattern["sub_queries"]
            ]
            log.debug(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
    return [query]


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

    def _search_single(self, query: str, top_k: int) -> list[dict]:
        """
        Embed a single query and return top_k results above threshold,
        excluding noisy schedule/table sections.
        Fetches extra candidates to account for filtered-out sections.
        """
        prefixed = BGE_QUERY_PREFIX + query
        query_vec = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Fetch more candidates than needed to absorb filtered exclusions
        scores, indices = self.index.search(query_vec, top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < MIN_SIMILARITY_SCORE:
                continue
            chunk = self.chunks[idx]
            # Skip noisy schedule/table sections
            if chunk.get("section") in EXCLUDED_SECTIONS:
                continue
            c = chunk.copy()
            c["score"] = float(score)
            results.append(c)
            if len(results) >= top_k:
                break

        return results

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        For simple queries: single FAISS search.
        For multi-section queries: decompose, search each sub-query,
        merge and deduplicate by best score.

        Returns empty list if nothing passes threshold → "I don't know".
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call .load() first.")

        sub_queries = decompose_query(query)

        if len(sub_queries) == 1:
            results = self._search_single(query, top_k)
        else:
            seen_ids = {}
            for sub_q in sub_queries:
                for chunk in self._search_single(sub_q, top_k=3):
                    cid = chunk["chunk_id"]
                    if cid not in seen_ids or chunk["score"] > seen_ids[cid]["score"]:
                        seen_ids[cid] = chunk

            results = sorted(
                seen_ids.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:top_k]

        log.debug(f"Query '{query[:60]}' → {len(results)} chunks retrieved")
        return results


# Singleton instance
retriever = Retriever()
