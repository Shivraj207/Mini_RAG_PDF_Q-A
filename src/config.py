import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "mini_rag_docs")

# Retrieval Settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 8))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 4))

# Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Embedding dimension
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1024))