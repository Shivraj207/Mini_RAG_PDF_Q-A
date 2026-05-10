import os
import sys
import time
import math
import streamlit as st
from qdrant_client import QdrantClient

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_text
from src.embedder import embed_texts
from src.qdrant_store import create_collection_if_not_exists, upsert_chunks
from src.retriever import retrieve_chunks
from src.reranker import rerank_chunks
from src.llm_answer import generate_answer
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME
)


st.set_page_config(page_title="Mini RAG Track B", layout="wide")

st.title("Mini RAG System")
st.caption("Upload a PDF, ask questions, and get grounded answers with citations.")

# ---------------------------
# Session State
# ---------------------------
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_processing_time" not in st.session_state:
    st.session_state.last_processing_time = None

if "last_chunk_count" not in st.session_state:
    st.session_state.last_chunk_count = 0

if "last_page_count" not in st.session_state:
    st.session_state.last_page_count = 0


# ---------------------------
# Helper Functions
# ---------------------------
def estimate_tokens(text: str) -> int:
    """
    Rough token estimate.
    A quick approximation: 1 token ≈ 4 characters for English text.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def clear_qdrant_collection():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False
    )

    existing_collections = client.get_collections().collections
    existing_names = [c.name for c in existing_collections]

    if QDRANT_COLLECTION_NAME in existing_names:
        client.delete_collection(QDRANT_COLLECTION_NAME)


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Controls")

    if st.button("Clear Vector Database"):
        clear_qdrant_collection()
        st.session_state.document_uploaded = False
        st.session_state.uploaded_filename = None
        st.session_state.chat_history = []
        st.session_state.last_processing_time = None
        st.session_state.last_chunk_count = 0
        st.session_state.last_page_count = 0
        st.success("Vector database cleared.")

    st.markdown("---")
    st.subheader("Chunking Settings")
    st.write(f"Chunk Size: {CHUNK_SIZE}")
    st.write(f"Chunk Overlap: {CHUNK_OVERLAP}")

    st.markdown("---")
    st.subheader("Project Status")
    if st.session_state.document_uploaded:
        st.success("Document processed")
        st.write(f"File: {st.session_state.uploaded_filename}")
        st.write(f"Pages: {st.session_state.last_page_count}")
        st.write(f"Chunks: {st.session_state.last_chunk_count}")
    else:
        st.warning("No document processed yet")


# ---------------------------
# Upload Section
# ---------------------------
st.subheader("1. Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    save_dir = "data/uploads"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            start_time = time.time()

            create_collection_if_not_exists()

            pages = extract_text_from_pdf(save_path)
            chunks = chunk_text(
                pages,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )

            texts = [chunk["text"] for chunk in chunks]
            embeddings = embed_texts(texts)

            upsert_chunks(chunks, embeddings)

            end_time = time.time()

            st.session_state.document_uploaded = True
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.last_processing_time = round(end_time - start_time, 2)
            st.session_state.last_chunk_count = len(chunks)
            st.session_state.last_page_count = len(pages)

            st.success("Document processed successfully.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Pages", len(pages))
            col2.metric("Chunks", len(chunks))
            col3.metric("Processing Time (s)", round(end_time - start_time, 2))


# ---------------------------
# Query Section
# ---------------------------
st.subheader("2. Ask a Question")

query = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not st.session_state.document_uploaded:
        st.error("Please upload and process a document first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            start_time = time.time()

            retrieved = retrieve_chunks(query)
            reranked = rerank_chunks(query, retrieved)
            answer, sources = generate_answer(query, reranked)

            end_time = time.time()

            # Show only top 2 sources to reduce noise
            top_sources = sources[:2]

            input_text_for_estimate = query + " ".join([src["text"] for src in top_sources])
            output_text_for_estimate = answer

            estimated_input_tokens = estimate_tokens(input_text_for_estimate)
            estimated_output_tokens = estimate_tokens(output_text_for_estimate)
            total_estimated_tokens = estimated_input_tokens + estimated_output_tokens

            result = {
                "question": query,
                "answer": answer,
                "sources": top_sources,
                "response_time": round(end_time - start_time, 2),
                "estimated_tokens": total_estimated_tokens
            }

            st.session_state.chat_history.insert(0, result)

            st.subheader("Answer")
            st.write(answer)

            col1, col2 = st.columns(2)
            col1.metric("Response Time (s)", result["response_time"])
            col2.metric("Estimated Tokens", result["estimated_tokens"])

            st.subheader("Top Sources")
            for source in top_sources:
                with st.expander(f"{source['citation']} - Page {source['page']}"):
                    st.write(source["text"])


# ---------------------------
# Chat History
# ---------------------------
if st.session_state.chat_history:
    st.subheader("Previous Questions")

    for idx, item in enumerate(st.session_state.chat_history, start=1):
        with st.expander(f"{idx}. {item['question']}"):
            st.write("**Answer:**")
            st.write(item["answer"])
            st.write(f"**Response Time:** {item['response_time']} s")
            st.write(f"**Estimated Tokens:** {item['estimated_tokens']}")

            st.write("**Sources:**")
            for src in item["sources"]:
                st.markdown(f"- {src['citation']} Page {src['page']}")