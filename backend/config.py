import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# Embedding
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# Chunking
MAX_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 80
CHUNK_OVERLAP_WORDS = 50

# Retrieval
TOP_K_RESULTS = 5
MIN_SIMILARITY_SCORE = 0.25  # below this = "I don't know"

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.1  # low = factual, consistent

# Source documents
BNS_FILES = {
    "BNS": DATA_DIR / "250883_english_01042024.pdf",    # Bharatiya Nyaya Sanhita
    "BNSS": DATA_DIR / "250884_2_english_01042024.pdf", # Bharatiya Nagarik Suraksha Sanhita
}