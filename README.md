# ⚖️ BNS Legal Research RAG Chatbot ⚖️

A conversational AI assistant for researching India's new criminal laws — the **Bharatiya Nyaya Sanhita, 2023 (BNS)** and the **Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)**. Ask questions in plain English and get accurate, cited answers grounded in the actual legal text.

---

## What This Is (And What It Isn't)

This is a **Retrieval-Augmented Generation (RAG)** system — a specific architecture where an AI assistant answers questions by first searching a knowledge base, then using only what it finds to construct a response.

This means:
- ✅ Answers are grounded in the actual BNS/BNSS text
- ✅ Every answer cites the exact section(s) it draws from
- ✅ If the answer isn't in the law, the system says so — it does not guess
- ❌ It does not use the LLM's general knowledge about Indian law
- ❌ It does not browse the internet

---

## How It Works

Imagine you hired a law clerk who has read both the BNS and BNSS cover to cover and marked every section with a sticky note. When you ask a question, the clerk:

1. Understands the *meaning* of your question (not just keywords)
2. Flips to the most relevant sticky notes (retrieval)
3. Reads those sections carefully
4. Writes you an answer using only what those sections say
5. Tells you exactly which sections they used

That's RAG. The "sticky notes" are vector embeddings — a mathematical representation of meaning. Similar meanings cluster together, so *"can police detain someone without paperwork"* retrieves the same sections as *"arrest without warrant"*.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE (run once)                   │
│                                                         │
│  BNS PDF + BNSS PDF                                     │
│       ↓                                                 │
│  PyMuPDF extraction + data cleaning                     │
│       ↓                                                 │
│  Hybrid chunking (section-aware + size limits)          │
│       ↓                                                 │
│  BAAI/bge-base-en-v1.5 embedding model                  │
│       ↓                                                 │
│  FAISS vector index  ←── saved to disk                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  ONLINE (every query)                   │
│                                                         │
│  User question (Streamlit UI)                           │
│       ↓                                                 │
│  Embed question with same BGE model                     │
│       ↓                                                 │
│  FAISS similarity search → top 5 matching chunks        │
│       ↓                                                 │
│  Similarity score check                                 │
│    ├─ too low?  → "I don't know" (no LLM call)          │
│    └─ good?     → build prompt with chunks              │
│                          ↓                              │
│              Groq API → Llama 3.3 70B                   │
│                          ↓                              │
│          Answer + section citations → UI                │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| PDF Parsing | PyMuPDF (fitz) | Fast, accurate, gives page-level control for metadata |
| Embedding Model | BAAI/bge-base-en-v1.5 | Top-ranked English retrieval model; 768-dim vectors |
| Vector Store | FAISS (IndexFlatIP) | Runs locally, no server needed, exact search — right size for ~1000 chunks |
| LLM | Llama 3.3 70B via Groq | Open-weight model; Groq's LPU hardware makes it very fast |
| Backend | FastAPI | Clean REST API, async, easy to extend |
| Frontend | Streamlit | Python-native chat UI — keeps complexity in the pipeline, not the interface |

---

## Key Design Decisions

### 1. Data Cleaning — Why It Matters

The BNS and BNSS are Official Gazette of India publications. Every single page of the PDF has this printed at the top:

```
THE GAZETTE OF INDIA EXTRAORDINARY
____________________________________________________________
____________________________________________________________
```

Without cleaning, this boilerplate gets embedded into every chunk. The embedding model wastes capacity representing "GAZETTE OF INDIA" instead of legal meaning, degrading retrieval quality throughout.

**What was cleaned and why:**

| Noise Type | Count | Impact if Not Removed |
|---|---|---|
| Gazette page headers | 102 occurrences | Pollutes embeddings on every chunk |
| Decorative underscore lines | 500+ occurrences | Same issue |
| Garbled Hindi/Devanagari text | 567 lines | Encoding artifacts in gazette front matter |
| `[Part II—` running headers | 50 occurrences | Pure noise |
| Form-feed characters (`\x0c`) | 102 occurrences | Creates false paragraph breaks mid-sentence |

**Margin notes — turned into an asset:** The printed PDFs have section titles in the right margin (e.g. "Commutation of sentence", "Right of private defence"). In raw text extraction these appear as broken line fragments floating before the section text. Rather than discarding them, we extract them and attach them as a `title` field to each chunk. These titles are embedded alongside the chunk text — significantly improving retrieval for short, natural-language queries that match the section title rather than the body.

---

### 2. Chunking Strategy — The Most Important Decision

**Why chunking matters:** The retrieval quality of a RAG system is only as good as its chunks. A chunk that is too large contains too many topics — retrieved for one thing, it confuses the LLM with irrelevant text. A chunk that is too small lacks context — the answer is there but the LLM cannot use it properly.

**Why section-aware chunking is the right choice for legal text:**

BNS and BNSS are structured around numbered sections: `Section 1`, `Section 2`, ... `Section 358` (BNS) and up to `Section 531` (BNSS). This structure is not arbitrary — it is the unit of law itself. Lawyers cite sections. Courts reference sections. Users ask about sections. Splitting mid-section at a fixed word count would break the legal logic apart.

**The problem with pure section-aware chunking:**

After analysing the actual documents, the sections vary dramatically in size:

```
Shortest section:  20 words  (Section 20 — one sentence about children under 7)
Average section:  183 words  (fits comfortably)
Longest section: 2,256 words (Section 2 — a list of 40+ definitions)
```

A 2,256-word chunk is far too large. It would be retrieved for any definition query, flooding the LLM with 39 irrelevant definitions alongside the one it needs.

**The hybrid solution:**

```
For each section after cleaning:

  if section < 80 words:
      → merge with adjacent section
        (too small to carry useful context alone)

  if 80 ≤ section ≤ 400 words:
      → keep as a single chunk
        (the natural legal unit, retrievable and complete)

  if section > 400 words:
      → sub-split at sentence boundaries with 50-word overlap
        (preserves context across sub-chunk boundaries)
        (section number and title retained in metadata of every sub-chunk)
```

The 50-word overlap on large sections is important: if a sentence spans a sub-chunk boundary, both chunks contain it, so retrieval never misses it.

**Known limitation and conscious tradeoff:** Section 2 (definitions) sub-splits by size, but a sub-chunk may still contain 3–4 definitions together. A more granular approach would split at each individual definition clause (`(1)`, `(2)`, etc.). This was evaluated and deliberately not implemented — it would produce ~40 tiny chunks from Section 2 alone, each lacking context about what Act they belong to. The current approach produces better results for the majority of queries, and this tradeoff is documented here.

---

### 3. Retrieval Quality

**The embedding model choice:** `BAAI/bge-base-en-v1.5` consistently ranks at the top of the MTEB (Massive Text Embedding Benchmark) leaderboard for English retrieval tasks. For legal text it handles formal language, sub-clauses, and technical terminology well.

One important implementation detail: BGE requires a specific prefix on *queries* at search time:
```
"Represent this sentence for searching relevant passages: <your query>"
```
Passages (chunks) are embedded without this prefix. Skipping this prefix degrades retrieval noticeably and is a commonly missed step.

**FAISS IndexFlatIP (exact search):** For a dataset of ~1000 chunks across both documents, approximate nearest-neighbour search is unnecessary and adds complexity with no speed benefit. Exact search guarantees we never miss a relevant chunk due to index approximation.

**Similarity threshold:** A cosine similarity score below `0.25` means the query has no meaningful match in the legal text. At this point the system returns "I don't know" immediately — no LLM call is made, no hallucination is possible. This is a hard gate, not a soft warning.

---

### 4. Handling Questions That Span Multiple Sections

Many real legal questions cannot be answered from a single section. For example:

> *"What happens after someone is arrested for murder?"*

This requires:
- **BNS Section 101** — definition and punishment for murder
- **BNSS Section 35** — arrest without warrant
- **BNSS Section 173** — FIR filing procedure
- **BNSS Sections 478–480** — bail (murder is non-bailable)

The system handles this in two complementary ways:

**At retrieval — multi-chunk retrieval across both laws:**
Top-5 chunks are retrieved, not top-1. The FAISS index contains all chunks from both BNS and BNSS in a single index, with `source` metadata tagging each chunk. This means a single query can surface relevant sections from both documents simultaneously — the retriever does not need to know in advance which law to look in.

**At generation — structured multi-section prompting:**
All retrieved chunks are passed to the LLM with clear source labels:

```
[1] BNS — Section 101 (Murder)
<text>

[2] BNSS — Section 35 (Arrest without warrant)
<text>

[3] BNSS — Section 173 (Information in cognizable cases)
<text>
```

The system prompt instructs the LLM to address each relevant section separately and cite section numbers inline. This produces structured answers rather than a blended response that obscures which law says what.

**Configurable depth:** The `top_k` parameter in `config.py` controls how many chunks are retrieved. The default is 5. For complex multi-section queries, increasing this to 7–8 improves coverage at the cost of a slightly longer prompt.

---

### 5. "I Don't Know" — Two-Layer Hallucination Prevention

Hallucination — confidently stating something wrong — is the primary failure mode of LLM-based systems. For a legal research tool, a wrong answer is worse than no answer.

**Layer 1 — Retrieval gate (before the LLM):**
If all top-5 similarity scores fall below `0.25`, the query has no match in the legal corpus. A "not found" message is returned immediately. The LLM is never called. This handles questions completely outside the scope of BNS/BNSS (e.g. income tax, property registration, divorce law).

**Layer 2 — LLM instruction (during generation):**
The system prompt contains an explicit instruction:
> *"If the context does not contain enough information to answer, say exactly: 'I don't have enough information in the provided legal text to answer this question.'"*

The response is scanned for this phrase. If found, citations are hidden and a warning is shown in the UI instead. This catches cases where retrieval returned *something* (passed the threshold) but the LLM correctly identified it as insufficient for the specific question.

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A free Groq API key — get one at [console.groq.com](https://console.groq.com) (2 minutes, no credit card needed)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd bns-rag-chatbot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and add: GROQ_API_KEY=your_key_here
```

### Add the Source PDFs

Place both PDFs in the `data/` folder with these exact filenames:

```
data/
├── 250883_english_01042024.pdf   # Bharatiya Nyaya Sanhita (BNS)
└── 250884_2_english_01042024.pdf # Bharatiya Nagarik Suraksha Sanhita (BNSS)
```

### Run the Application

Open two terminals:

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --reload
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

Go to **http://localhost:8501** in your browser.

### Build the Index (First Run Only)

Click **"Build / Rebuild Index"** in the sidebar. This runs the full ingestion pipeline — parsing, cleaning, chunking, embedding, and indexing. Takes **2–5 minutes** and only needs to be done once. The index persists to disk at `faiss_index/` and is reloaded automatically on restart.

---

## Running Evaluations

```bash
python -m evals.test_retrieval
```

Runs ground-truth Q&A pairs and reports retrieval and answer accuracy.

| Category | Questions Tested |
|---|---|
| BNS offences | Murder, culpable homicide, theft, rape, dacoity, defamation |
| BNS general | Punishments, private defence, abetment, criminal conspiracy |
| BNSS procedure | Bail, FIR filing, arrest without warrant |
| Edge cases | Age of criminal responsibility (Sections 20–21) |


---

## Project Structure

```
bns-rag/
├── backend/
│   ├── config.py          # All settings — chunk sizes, thresholds, model names
│   ├── ingest.py          # PDF → clean → chunk → embed → FAISS index
│   ├── retriever.py       # FAISS search + similarity threshold + BGE query prefix
│   ├── generator.py       # Prompt construction, Groq API call, citation formatting
│   └── main.py            # FastAPI: /health, /ingest, /query
├── frontend/
│   └── app.py             # Streamlit chat UI
├── data/                  # Place PDFs here
├── evals/
│   └── test_retrieval.py  # ground-truth eval cases
├── faiss_index/           # Auto-generated after ingestion
├── .env.example
├── requirements.txt
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Backend status and chunk count |
| `POST` | `/ingest` | Run the full ingestion pipeline |
| `POST` | `/query` | `{question, top_k}` → `{answer, citations, has_answer, chunks_used}` |

---

## Example Queries

**Offences and punishments (BNS)**
- *"What is the punishment for murder?"*
- *"What is the difference between murder and culpable homicide?"*
- *"What constitutes theft under BNS?"*
- *"What does BNS say about the right of private defence?"*

**Procedure (BNSS)**
- *"How is an FIR filed?"*
- *"What are the bail provisions?"*
- *"Can police arrest without a warrant?"*

**Multi-section queries**
- *"What happens after someone is arrested for murder?"*
- *"What is the procedure after an FIR is filed for theft?"*

**Should return "I don't know"**
- *"What does BNS say about income tax evasion?"*
- *"What are the rules for property registration?"*
- *"How do I file for divorce?"*
