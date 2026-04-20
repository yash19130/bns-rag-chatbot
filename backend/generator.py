"""
generator.py — Build RAG prompt and call Groq (Llama 3.1 70B) for generation.
Handles "I don't know" when no relevant chunks are found.
"""

import logging
from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE

log = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a precise legal research assistant specializing in Indian criminal law.
You answer questions about both offences (BNS) and criminal procedure (BNSS).
You answer questions based ONLY on the provided excerpts from:
- Bharatiya Nyaya Sanhita, 2023 (BNS) — defines offences and punishments
- Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) — criminal procedure

Rules you must follow:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. Always cite the specific section number(s) you are drawing from.
3. If the context does not contain enough information to answer, say exactly:
   "I don't have enough information in the provided legal text to answer this question."
4. Be precise. Use legal language where appropriate.
5. For procedural questions (what happens after arrest, FIR procedure etc.), walk through the steps in order using the relevant sections.
6. If a question spans multiple sections or both BNS and BNSS, address each part clearly and in logical sequence.
6. Do not speculate or infer beyond what the text explicitly states."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, c in enumerate(chunks, 1):
        source_label = f"{c['source']} — {c['section']}"
        if c.get("title"):
            source_label += f" ({c['title']})"
        lines.append(f"[{i}] {source_label}\n{c['text']}")
    return "\n\n---\n\n".join(lines)


def format_citations(chunks: list[dict]) -> str:
    """Build a clean citation block to append after the answer."""
    seen = set()
    citations = []
    for c in chunks:
        key = (c["source"], c["section"])
        if key not in seen:
            seen.add(key)
            title = f" — {c['title']}" if c.get("title") else ""
            citations.append(f"• {c['source']}, {c['section']}{title}")
    return "\n".join(citations)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks.

    Returns:
    {
        "answer": str,
        "citations": str,
        "has_answer": bool,
    }
    """
    # No chunks above threshold → "I don't know"
    if not chunks:
        return {
            "answer": "I don't have enough information in the provided legal text to answer this question.",
            "citations": "",
            "has_answer": False,
        }

    context = format_context(chunks)

    user_message = f"""Context from Indian legal texts:

{context}

---

Question: {query}

Answer based strictly on the context above. Cite specific section numbers in your answer."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        answer = response.choices[0].message.content.strip()

        # Check if LLM itself said it doesn't know
        has_answer = "don't have enough information" not in answer.lower()

        return {
            "answer": answer,
            "citations": format_citations(chunks),
            "has_answer": has_answer,
        }

    except Exception as e:
        log.error(f"Groq API error: {e}")
        return {
            "answer": f"Error contacting the language model: {str(e)}",
            "citations": "",
            "has_answer": False,
        }
