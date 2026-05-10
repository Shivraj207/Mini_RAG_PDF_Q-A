import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from a PDF page by page.

    Returns:
        List of dictionaries like:
        [
            {"page": 1, "text": "some text"},
            {"page": 2, "text": "some text"}
        ]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page": page_num,
                "text": text
            })

    doc.close()
    return pages