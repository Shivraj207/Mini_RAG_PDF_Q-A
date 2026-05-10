import requests
from src.config import JINA_API_KEY


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"


def embed_texts(texts):
    """
    Generate embeddings for a list of texts using Jina API.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    payload = {
        "model": "jina-embeddings-v3",
        "input": texts
    }

    response = requests.post(
        JINA_EMBED_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    data = response.json()["data"]
    embeddings = [item["embedding"] for item in data]
    return embeddings


def embed_query(query):
    """
    Generate embedding for a single query.
    """
    return embed_texts([query])[0]