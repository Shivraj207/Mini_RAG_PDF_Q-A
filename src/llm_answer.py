from google import genai
from src.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(query, reranked_chunks):
    if not reranked_chunks:
        return "I could not find the answer in the uploaded document.", []

    context_parts = []
    sources = []

    for idx, chunk in enumerate(reranked_chunks, start=1):
        context_parts.append(f"[{idx}] Page {chunk['page']}: {chunk['text']}")
        sources.append({
            "citation": f"[{idx}]",
            "page": chunk["page"],
            "text": chunk["text"]
        })

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say:
I could not find the answer in the uploaded document.

Use inline citations like [1], [2].

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = response.text.strip()
    return answer, sources