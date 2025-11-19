from typing import List

import google.generativeai as genai

from .config import CHAT_MODEL, GOOGLE_API_KEY
from .retriever import get_retriever


SYSTEM_PROMPT = """You are a professional SMC trading assistant.

You ONLY answer based on the provided context from SMC strategy documents 
(BOS, liquidity grabs, FVGs, etc).

Rules:
- If the context is not enough, say you are not sure and ask for more details.
- Do NOT invent new strategies outside the given context.
- Explain in simple, practical language.
- Focus on structure, liquidity, and risk management.
"""

USER_TEMPLATE = """User trade idea / question:
{question}

Use the context to:
- Identify which SMC concepts are relevant (BOS, liquidity grab, FVG, etc).
- Explain how they apply.
- Mention risk considerations (SL placement, over-risking, etc).

Answer clearly in 1–3 short paragraphs.

Context:
{context}
"""


genai.configure(api_key=GOOGLE_API_KEY)


def _format_context(chunks) -> str:
    if not chunks:
        return "No relevant context found in the knowledge base."
    return "\n\n---\n\n".join(doc.page_content for doc in chunks)


def _format_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT.strip()}\n\n{USER_TEMPLATE.format(question=question, context=context)}"


def _call_llm(prompt: str) -> str:
    model = genai.GenerativeModel(model_name=CHAT_MODEL)
    response = model.generate_content(prompt)
    if not response or not response.candidates:
        raise RuntimeError("Gemini returned no response.")
    text = getattr(response, "text", None)
    if not text:
        # Fallback: concatenate output parts
        parts: List[str] = []
        for candidate in response.candidates:
            for part in getattr(candidate, "content", []).parts:
                if getattr(part, "text", None):
                    parts.append(part.text)
        text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("Gemini response did not include any text output.")
    return text


def _retrieve_context(question: str):
    retriever = get_retriever(k=4)

    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)

    return retriever.invoke(question)


def ask_rag(question: str) -> str:
    chunks = _retrieve_context(question)
    context = _format_context(chunks)
    prompt = _format_prompt(question, context)
    return _call_llm(prompt)
