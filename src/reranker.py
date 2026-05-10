import requests
from src.config import JINA_API_KEY, TOP_K_RERANK

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


def rerank_chunks(query, chunks):
    """
    Rerank retrieved chunks using Jina reranker.
    """
    documents = [chunk["text"] for chunk in chunks]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": documents,
        "top_n": min(TOP_K_RERANK, len(documents))
    }

    response = requests.post(
        JINA_RERANK_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    rerank_results = response.json()["results"]

    final_chunks = []
    for item in rerank_results:
        idx = item["index"]
        chunk = chunks[idx].copy()
        chunk["rerank_score"] = item["relevance_score"]
        final_chunks.append(chunk)

    return final_chunks