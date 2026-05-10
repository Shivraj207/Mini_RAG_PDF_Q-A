from typing import List, Dict


def chunk_text(pages: List[Dict], chunk_size: int = 1000, overlap: int = 150):
    """
    Split extracted PDF text into overlapping chunks.

    Args:
        pages: list like [{"page": 1, "text": "..."}, ...]
        chunk_size: number of characters per chunk
        overlap: overlapping characters between chunks

    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "chunk_id": f"page_{page_num}_chunk_{chunk_index}",
                    "page": page_num,
                    "source": "uploaded_pdf",
                    "title": "User Uploaded PDF",
                    "section": f"Page {page_num}",
                    "position": chunk_index,
                    "text": chunk
                })

            start += chunk_size - overlap
            chunk_index += 1

    return chunks