# Mini RAG App

A document question-answering system built for Track B using Retrieval-Augmented Generation (RAG). Users can upload a PDF, process it into chunks, store embeddings in Qdrant, retrieve relevant chunks, rerank them, and generate grounded answers with citations.

---

## Features

- Upload PDF documents
- Extract text page by page
- Chunk text with overlap
- Generate embeddings using Jina AI
- Store vectors in Qdrant Cloud
- Retrieve relevant chunks semantically
- Rerank results using Jina Reranker
- Generate grounded answers using Gemini
- Show page-level citations
- Streamlit-based user interface
- Clear vector database from sidebar

---

## Tech Stack

- **Frontend:** Streamlit
- **Vector Database:** Qdrant Cloud
- **Embeddings:** Jina Embeddings
- **Reranker:** Jina Reranker
- **LLM:** Gemini
- **PDF Parsing:** PyMuPDF
- **Language:** Python

---

## Project Structure

```text
mini_rag_track_b/
│
├── app/
│   └── main.py
│
├── src/
│   ├── config.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── qdrant_store.py
│   ├── retriever.py
│   ├── reranker.py
│   └── llm_answer.py
│
├── data/
│   └── uploads/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt