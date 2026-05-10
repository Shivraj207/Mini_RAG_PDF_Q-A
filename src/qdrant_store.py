import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)


def create_collection_if_not_exists():
    existing_collections = client.get_collections().collections
    existing_names = [c.name for c in existing_collections]

    if QDRANT_COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )


def upsert_chunks(chunks, embeddings, batch_size=64):
    """
    Upload points to Qdrant in small batches to avoid write timeouts.
    """
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=chunk
            )
        )

    # Send in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=batch,
            wait=True
        )


def search_similar(query_embedding, top_k=8):
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    )

    return [point.payload for point in results.points]