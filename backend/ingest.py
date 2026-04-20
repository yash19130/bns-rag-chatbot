"""
ingest.py — Parse BNS/BNSS PDFs, clean text, chunk by section (hybrid),
embed with BAAI/bge-base-en-v1.5, and store in a FAISS index.

Run directly:  python -m backend.ingest
"""

import re
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, FAISS_INDEX_DIR,
    MAX_CHUNK_WORDS, MIN_CHUNK_WORDS, CHUNK_OVERLAP_WORDS, BNS_FILES
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF using PyMuPDF, page by page."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# 2. Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """
    Remove gazette boilerplate that pollutes every page:
    - Form-feed characters (\x0c)
    - Gazette page headers (THE GAZETTE OF INDIA EXTRAORDINARY + underscores)
    - Sec. / 1] running headers
    - [Part II— running headers
    - Standalone underscore lines (decorative rules)
    - Devanagari/garbled Hindi lines (only in gazette front matter)
    - Excessive blank lines
    """
    text = raw

    # Form feeds
    text = text.replace("\x0c", "\n")

    # Full gazette header block that appears on every page:
    # "Sec.\n1]\nTHE ____...\nGAZETTE OF INDIA EXTRAORDINARY\n<page_num>\n___...\n"
    text = re.sub(
        r"Sec\.\s*\n\s*1\]\s*\n\s*THE _+\s*\nGAZETTE OF INDIA EXTRAORDINARY\s*\n\s*\d*\s*\n(_+\s*\n)+",
        "\n", text
    )

    # Alternate header pattern (page number first)
    text = re.sub(
        r"\n\s*\d{1,3}\s*\nTHE _+\s*\nGAZETTE OF INDIA EXTRAORDINARY\s*\n(_+\s*\n)+",
        "\n", text
    )

    # [Part II— lines
    text = re.sub(r"\[Part II[^\n]*\n", "", text)

    # Standalone underscore-only lines (decorative rules)
    text = re.sub(r"^_+\s*$", "", text, flags=re.MULTILINE)

    # Lines containing only Devanagari Unicode (U+0900–U+097F) — garbled Hindi
    text = re.sub(r"^[^\n]*[\u0900-\u097F][^\n]*$", "", text, flags=re.MULTILINE)

    # Gazette registration lines like "REGISTERED NO. DL—(N)04/..."
    text = re.sub(r"^REGISTERED NO\.[^\n]*\n", "", text, flags=re.MULTILINE)

    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# 3. Margin Note Extraction
# ---------------------------------------------------------------------------

def extract_margin_note(block: str) -> tuple[str, str]:
    """
    In BNS PDFs, each section is preceded by a short margin note
    (section title) printed in the right margin. After cleaning, these
    appear as 1–5 short lines immediately before the `N. (1) text...` line.

    Returns (margin_title, remaining_text_without_margin).

    Example input block:
        Commutation
        of sentence.

        5. The appropriate Government may...

    Returns: ("Commutation of sentence", "5. The appropriate Government may...")
    """
    lines = block.strip().split("\n")
    # Section content starts at the line matching ^<digits>.
    section_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\d+\.\s", line.strip()):
            section_start = i
            break

    if section_start is None:
        return ("", block)

    margin_lines = [l.strip() for l in lines[:section_start] if l.strip()]
    margin_title = " ".join(margin_lines).rstrip(".")
    content = "\n".join(lines[section_start:]).strip()
    return (margin_title, content)


# ---------------------------------------------------------------------------
# 4. Hybrid Chunking
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def split_large_section(section_text: str, section_num: str, max_words: int, overlap: int) -> list[str]:
    """
    Split an oversized section into overlapping sub-chunks.
    Tries to split at sentence boundaries.
    """
    sentences = re.split(r"(?<=[.;])\s+", section_text)
    chunks = []
    current_words = []
    current_count = 0

    for sent in sentences:
        sent_words = sent.split()
        if current_count + len(sent_words) > max_words and current_words:
            chunks.append(" ".join(current_words))
            # Overlap: keep last `overlap` words
            overlap_words = current_words[-overlap:] if len(current_words) > overlap else current_words
            current_words = overlap_words + sent_words
            current_count = len(current_words)
        else:
            current_words.extend(sent_words)
            current_count += len(sent_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def hybrid_chunk(text: str, source_name: str) -> list[dict]:
    """
    Hybrid chunking strategy:
    1. Split at section boundaries (^\d+\. pattern)
    2. If section > MAX_CHUNK_WORDS → sub-split with overlap
    3. If section < MIN_CHUNK_WORDS → merge with next section
    4. Attach metadata: section_number, margin_title, source

    Returns list of chunk dicts:
    {
        "text": str,          # text to embed
        "section": str,       # e.g. "Section 302"
        "title": str,         # margin note title
        "source": str,        # "BNS" or "BNSS"
        "chunk_id": int
    }
    """
    # Split on section boundaries — captures the section number
    raw_sections = re.split(r"(?=\n\d{1,3}\.\s)", "\n" + text)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    # Also capture CHAPTER headings as context anchors
    chapter_pattern = re.compile(r"^(CHAPTER\s+[IVXLC]+.*?)$", re.MULTILINE)

    chunks = []
    chunk_id = 0
    pending = ""  # for merging small sections
    pending_meta = {}

    def flush_pending(title, section_num):
        nonlocal pending, pending_meta, chunk_id
        if pending.strip():
            chunks.append({
                "text": pending.strip(),
                "section": section_num,
                "title": title,
                "source": source_name,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
        pending = ""
        pending_meta = {}

    current_chapter = ""

    for raw in raw_sections:
        # Track chapter headings embedded in section blocks
        chapter_match = chapter_pattern.search(raw)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()

        # Extract section number
        sec_match = re.match(r"^(\d{1,3})\.", raw.strip())
        section_num = f"Section {sec_match.group(1)}" if sec_match else "Preamble"

        # Extract margin note title
        margin_title, content = extract_margin_note(raw)
        display_title = margin_title if margin_title else section_num

        # Prepend chapter for context if available
        if current_chapter:
            content_with_ctx = f"{current_chapter}\n{content}"
        else:
            content_with_ctx = content

        wc = word_count(content_with_ctx)

        if wc < MIN_CHUNK_WORDS:
            # Too small: accumulate into pending
            pending += "\n\n" + content_with_ctx
            if not pending_meta:
                pending_meta = {"title": display_title, "section": section_num}
            # If still small after accumulation, keep accumulating
            if word_count(pending) >= MIN_CHUNK_WORDS:
                flush_pending(pending_meta["title"], pending_meta["section"])

        elif wc > MAX_CHUNK_WORDS:
            # Flush any pending first
            if pending:
                flush_pending(pending_meta.get("title", ""), pending_meta.get("section", ""))

            # Sub-split large section
            sub_chunks = split_large_section(content_with_ctx, section_num, MAX_CHUNK_WORDS, CHUNK_OVERLAP_WORDS)
            for i, sub in enumerate(sub_chunks):
                sub_title = f"{display_title} (part {i+1})" if len(sub_chunks) > 1 else display_title
                chunks.append({
                    "text": sub,
                    "section": section_num,
                    "title": sub_title,
                    "source": source_name,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

        else:
            # Flush pending, add this chunk normally
            if pending:
                flush_pending(pending_meta.get("title", ""), pending_meta.get("section", ""))
            chunks.append({
                "text": content_with_ctx,
                "section": section_num,
                "title": display_title,
                "source": source_name,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    # Flush any remaining pending
    if pending:
        flush_pending(pending_meta.get("title", ""), pending_meta.get("section", ""))

    return chunks


# ---------------------------------------------------------------------------
# 5. Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed each chunk. For BGE models, prefix query with 'Represent this sentence:'
    is for queries only. For passages, no prefix needed.
    We embed: title + text for richer representation.
    """
    texts = []
    for c in chunks:
        # Combine title + text so the section name is part of the vector
        combined = f"{c['title']}. {c['text']}" if c["title"] else c["text"]
        texts.append(combined)

    log.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for cosine similarity with FAISS IP
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. FAISS Index Building
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat inner-product index (cosine similarity since embeddings are normalized).
    Flat index = exact search, no approximation. Fine for ~1000 chunks.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine when normalized
    index.add(embeddings)
    log.info(f"FAISS index built with {index.ntotal} vectors (dim={dim})")
    return index


# ---------------------------------------------------------------------------
# 7. Save / Load
# ---------------------------------------------------------------------------

def save_index(index: faiss.Index, chunks: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    # Also save as JSON for inspection
    with open(output_dir / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    log.info(f"Saved index + {len(chunks)} chunks to {output_dir}")


def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# 8. Main Entry Point
# ---------------------------------------------------------------------------

def run_ingestion():
    log.info("=== BNS RAG Ingestion Pipeline ===")

    # Load embedding model
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []

    for source_name, pdf_path in BNS_FILES.items():
        if not pdf_path.exists():
            log.warning(f"PDF not found: {pdf_path} — skipping")
            continue

        log.info(f"Processing {source_name}: {pdf_path.name}")

        raw_text = extract_text_from_pdf(pdf_path)
        log.info(f"  Extracted {len(raw_text):,} chars")

        clean = clean_text(raw_text)
        log.info(f"  Cleaned: {len(clean):,} chars")

        chunks = hybrid_chunk(clean, source_name)
        log.info(f"  Chunked into {len(chunks)} chunks")

        # Log chunk size distribution
        sizes = [word_count(c["text"]) for c in chunks]
        log.info(f"  Chunk sizes — min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")

        all_chunks.extend(chunks)

    log.info(f"Total chunks across all documents: {len(all_chunks)}")

    # Re-assign chunk_ids sequentially across docs
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    # Embed
    embeddings = embed_chunks(all_chunks, model)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save
    save_index(index, all_chunks, FAISS_INDEX_DIR)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()