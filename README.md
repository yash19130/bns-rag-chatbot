# BNS Legal Research RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for researching the **Bharatiya Nyaya Sanhita, 2023 (BNS)** and **Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)** — India's new criminal law codes.

---

## Architecture

```
User Query (Streamlit)
        ↓
  FastAPI /query
        ↓
  Embed query (BAAI/bge-base-en-v1.5)
        ↓
  FAISS similarity search (top-5 chunks)
        ↓
  Similarity threshold check → "I don't know" if below 0.35
        ↓
  Build prompt with retrieved chunks
        ↓
  Groq API → Llama 3.3 70B (open-weight, hosted)
        ↓
  Answer + Section citations → Streamlit UI
```

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| PDF Parsing | PyMuPDF (fitz) | Fast, accurate text extraction, page-level control |
| Embedding Model | BAAI/bge-base-en-v1.5 | Best-in-class retrieval quality for English; 768-dim |
| Vector Store | FAISS (IndexFlatIP) | Local, no server needed, exact search for ~1000 chunks |
| LLM | Llama 3.3 70B via Groq | Open-weight model, Groq's LPU inference = very fast |
| Backend | FastAPI | Async, clean API, easy to extend |
| Frontend | Streamlit | Minimal Python-native chat UI; keeps focus on pipeline |

---

## Key Design Decisions

### Chunking Strategy: Hybrid (Section-aware + size limits)

BNS/BNSS are structured as numbered sections (`1.`, `2.`, ...). Each section is the natural legal unit — citations, queries, and answers all reference section numbers. We split at section boundaries first, then apply size rules:

- **< 80 words**: merge with the adjacent section (too small to be retrievable on its own)
- **80–400 words**: keep as-is (most sections fall here)
- **> 400 words**: sub-split with 50-word overlap at sentence boundaries (Section 2, the definitions section, is 2,200+ words)

Each chunk also carries the **margin note title** (e.g. "Commutation of sentence") extracted from the printed margin and embedded alongside the text. This significantly improves retrieval for short natural-language queries that match the section title rather than the section body.

### "I Don't Know" Handling

Two-layer approach:
1. **Retrieval layer**: if all top-5 FAISS scores fall below `MIN_SIMILARITY_SCORE = 0.35`, no context is passed to the LLM and a canned "not found" message is returned immediately — no wasted LLM call.
2. **LLM layer**: the system prompt instructs Llama to explicitly state when context is insufficient. The response is scanned for this phrase and flagged in the UI.

### Source Citations

Every retrieved chunk carries `source` ("BNS"/"BNSS") and `section` ("Section 101") metadata. After generation, the unique sections used are collected and displayed as a citation block below the answer. The LLM is also instructed to cite section numbers inline in its answer.

### Data Cleaning

The PDFs are Official Gazette publications with heavy boilerplate on every page:
- Gazette headers ("THE GAZETTE OF INDIA EXTRAORDINARY") repeated 102 times
- 500+ decorative underscore lines
- Garbled Devanagari Unicode in gazette front matter
- Margin notes (section titles) appearing as broken line fragments

All of this is stripped before chunking so it doesn't pollute embeddings or retrieved context.

---

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (takes 2 minutes)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd bns-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Add the PDF source documents

Place the BNS and BNSS PDFs in the `data/` directory:

```
data/
├── 250883_english_01042024.pdf   # Bharatiya Nyaya Sanhita (BNS)
└── 250884_2_english_01042024.pdf # Bharatiya Nagarik Suraksha Sanhita (BNSS)
```

### Running the application

**Terminal 1 — Start the FastAPI backend:**

```bash
uvicorn backend.main:app --reload
```

**Terminal 2 — Start the Streamlit frontend:**

```bash
streamlit run frontend/app.py
```

Open your browser at **http://localhost:8501**

### Build the index

On first run, click **"Build / Rebuild Index"** in the sidebar. This will:
1. Parse and clean both PDFs
2. Chunk the text using the hybrid strategy
3. Embed all chunks with BAAI/bge-base-en-v1.5
4. Build and save the FAISS index to `faiss_index/`

This takes approximately **2–5 minutes** and only needs to be done once. The index persists on disk.

You can also trigger ingestion directly via the API:

```bash
curl -X POST http://localhost:8000/ingest
```

---

## Running Evaluations

```bash
python -m evals.test_retrieval
```

This runs 15 ground-truth question/answer pairs covering:
- Murder, culpable homicide, theft, rape (BNS offences)
- Bail, FIR, arrest without warrant (BNSS procedure)
- Abetment, conspiracy, defamation, dacoity (various BNS)
- Age of criminal responsibility, right of private defence

Pass thresholds: **≥80% retrieval accuracy**, **≥70% answer accuracy**

---

## Project Structure

```
bns-rag/
├── backend/
│   ├── __init__.py
│   ├── config.py       # All settings (chunk sizes, model names, thresholds)
│   ├── ingest.py       # PDF parsing → cleaning → chunking → embedding → FAISS
│   ├── retriever.py    # FAISS query + similarity threshold
│   ├── generator.py    # Groq LLM call + prompt + citations
│   └── main.py         # FastAPI app (3 endpoints)
├── frontend/
│   └── app.py          # Streamlit chat UI
├── data/
│   └── (PDFs go here)
├── evals/
│   ├── __init__.py
│   └── test_retrieval.py  # 15 ground-truth eval cases
├── faiss_index/           # Auto-generated after ingestion
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check backend + index status |
| POST | `/ingest` | Trigger full ingestion pipeline |
| POST | `/query` | RAG query → `{answer, citations, has_answer, chunks_used}` |

---

## Example Queries

- *"What is the punishment for murder under BNS?"*
- *"What are the bail provisions in BNSS?"*
- *"When can police arrest without a warrant?"*
- *"What constitutes abetment under BNS?"*
- *"What does BNS say about the right of private defence?"*
- *"What is the difference between murder and culpable homicide?"*
