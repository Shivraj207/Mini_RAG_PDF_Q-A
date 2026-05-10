from src.embedder import embed_query
from src.qdrant_store import search_similar
from src.config import TOP_K_RETRIEVAL


def retrieve_chunks(query: str):
    """
    Convert query to embedding and fetch top-k relevant chunks from Qdrant.
    """
    query_embedding = embed_query(query)
    results = search_similar(query_embedding, top_k=TOP_K_RETRIEVAL)
    return results