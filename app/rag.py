from __future__ import annotations

import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient, models

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
QDRANT_INFERENCE_MODEL = os.getenv(
    "QDRANT_INFERENCE_MODEL",
    "sentence-transformers/all-minilm-l6-v2",
)
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "electric_motors")
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
NO_ANSWER = "I couldn't find that in the electric motor reference material."
SYSTEM_PROMPT = """
You are an intelligent AI assistant powered by a
Retrieval-Augmented Generation (RAG) system.

YOUR MAIN RESPONSIBILITY:
Understand the user's actual intention and provide
accurate, helpful, natural, and easy-to-understand answers.

1. UNDERSTANDING USER QUESTIONS
- Understand simple, casual and informal language.
- Do not require users to use technical terminology.
- Understand spelling mistakes and grammatical errors.
- Understand incomplete and poorly structured questions.
- Identify the actual meaning behind the question.
- Understand synonyms and alternative expressions.
- Do not depend on exact keyword matching.
- Interpret what the user means, not just what they type.

2. QUERY INTERPRETATION
- Convert unclear questions into meaningful search queries.
- Identify important concepts and related terminology.
- Consider different possible meanings of a question.
- Search for relevant information even when the user's
  wording differs from the document's wording.
- Never change the original intent of the question.

3. CONTEXT AWARENESS
- Remember previous questions in the conversation.
- Understand follow-up questions such as:
  "Explain more", "Why?", "How does it work?",
  "Give an example", and "What about its types?"
- Connect follow-up questions to the previous topic.
- Do not ask users to repeat information unnecessarily.

4. DOCUMENT RETRIEVAL
- Use the retrieved document context carefully.
- Consider all relevant retrieved passages.
- Combine information from multiple relevant passages.
- Do not select information based only on keyword matches.
- Ignore irrelevant retrieved content.
- If information is missing, clearly acknowledge it.

5. ANSWER GENERATION
- Answer the exact question asked.
- Use simple and understandable English.
- Explain technical terms in ordinary language.
- Prefer clear and direct answers.
- Break complicated concepts into smaller parts.
- Provide examples wherever useful.
- Use bullet points and headings when appropriate.
- Avoid unnecessary technical jargon.
- Do not give unnecessarily long answers.

6. ACCURACY AND HONESTY
- Never invent facts, figures, or references.
- Do not assume information that is not provided.
- Distinguish document-based facts from general knowledge.
- If the retrieved context does not contain the answer,
  clearly say that the documents do not provide it.
- Never pretend to have found information that is missing.

7. CONVERSATIONAL BEHAVIOUR
- Be friendly, natural and helpful.
- Respond like a human tutor.
- Do not sound robotic.
- Handle greetings and casual conversation naturally.
- Ask a clarification question only when necessary.
- If the question is understandable, answer directly.

8. EXPLANATION STYLE
When explaining a topic:
- Start with a simple definition.
- Explain how it works.
- Give a practical example.
- Explain important types or applications if relevant.
- Provide a short summary when useful.

9. OUTPUT QUALITY
- Prioritize relevance over answer length.
- Do not repeat the same information.
- Use examples related to the question.
- Make answers understandable even to beginners.

IMPORTANT:
Your primary purpose is to help users understand
information, not simply repeat retrieved document text.

MATHEMATICAL FORMATTING:
- Put every inline mathematical expression inside $...$.
- Put each standalone equation on its own line inside $$...$$.
- Use valid LaTeX with balanced braces; for example, write $E_a = V_t - I_a(R_a + R_{se})$.
- Do not output LaTeX commands or equation syntax as unformatted plain text.
- Do not use Markdown emphasis markers inside math delimiters.
"""


def _format_bare_math(text: str) -> str:
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "$$\n" + match.group(1).strip() + "\n$$",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: "$" + match.group(1).strip() + "$",
        text,
        flags=re.DOTALL,
    )

    def wrap_formula(match: re.Match[str]) -> str:
        formula = match.group(1).strip().rstrip(" ;,.")
        return f"${formula}$"

    text = re.sub(
        r"(?<!\\)\[\s*([^\[\]\n]*?(?:\\[A-Za-z]+|=)[^\[\]\n]*?)\s*(?<!\\)\]",
        wrap_formula,
        text,
    )

    def wrap_standalone_equation(line: str) -> str:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(("$", "#", "- ", "* ", ">", "1.", "2.", "3."))
            or "$" in stripped
        ):
            return line

        equation = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9{}_^]*)\s*=\s*(.+)", stripped
        )
        if not equation:
            return line

        right_hand_side = equation.group(2)
        if not re.search(r"\\[A-Za-z]+|[_^]|[+*/]|[×·]", right_hand_side):
            return line
        return f"$$\n{stripped}\n$$"

    return "\n".join(wrap_standalone_equation(line) for line in text.splitlines())


class RagService:
    """Queries the shared Qdrant collection and answers user questions."""

    def __init__(self) -> None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be configured")

        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._client = None
        self._client_lock = Lock()
        self.collection_name = QDRANT_COLLECTION

    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = QdrantClient(
                        url=self._qdrant_url,
                        api_key=self._qdrant_api_key,
                        cloud_inference=True,
                        check_compatibility=False,
                    )
        return self._client

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def ask(
        self,
        question: str,
        top_k: int = 5,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        collection_count = self.count()
        if collection_count == 0:
            return {"answer": NO_ANSWER, "sources": []}

        history = (history or [])[-8:]
        api_key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
        llm = None
        retrieval_question = question
        if history and api_key:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=DEFAULT_GROQ_MODEL,
                temperature=0.1,
                max_tokens=2048,
            )
            conversation = "\n".join(
                f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
            )
            try:
                rewritten = llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Rewrite the latest user question as a standalone search query, "
                                "using the conversation only to resolve references such as 'it' "
                                "or 'that motor'. Preserve the user's intent. Return only the "
                                "query; do not answer it. If it is already standalone, return it unchanged."
                            )
                        ),
                        HumanMessage(
                            content=f"Conversation:\n{conversation}\n\nLatest question: {question}"
                        ),
                    ]
                )
                rewritten_content = rewritten.content
                if isinstance(rewritten_content, str) and rewritten_content.strip():
                    retrieval_question = rewritten_content.strip().strip('"')
            except Exception:
                # Keep the original query if clarification fails; normal RAG can still proceed.
                pass

        matches = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=retrieval_question,
                model=QDRANT_INFERENCE_MODEL,
            ),
            limit=min(top_k, collection_count),
            with_payload=True,
        )
        sources = []
        for point in matches.points:
            payload = point.payload or {}
            similarity = float(point.score or 0.0)
            if similarity < MIN_SIMILARITY:
                continue
            sources.append(
                {
                    "source": Path(str(payload.get("source", "document"))).name,
                    "page": int(payload.get("page", 0)) + 1,
                    "content": str(payload.get("text", "")),
                    "similarity": round(similarity, 4),
                }
            )

        if not sources:
            return {"answer": NO_ANSWER, "sources": []}

        if not api_key:
            raise RuntimeError("GROQ_API is not configured")
        if llm is None:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=DEFAULT_GROQ_MODEL,
                temperature=0.1,
                max_tokens=2048,
            )

        labeled_context = "\n\n".join(
            f"[{index}] {source['content']}"
            for index, source in enumerate(sources, start=1)
        )
        conversation_context = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
        )
        history_section = (
            f"Recent conversation (use only to understand references in the latest question):\n{conversation_context}\n\n"
            if conversation_context
            else ""
        )
        prompt = f"""Use the following retrieved passages to answer the question.
If none of the passages contains the answer, reply exactly: {NO_ANSWER}
Include the supporting passage number, such as [1].
Write valid equations in LaTeX using $...$ inline or $$...$$ on separate lines. Balance all braces and delimiters. Put each standalone equation on its own line. Do not output raw LaTeX commands or bare square brackets as math delimiters.

{history_section}Passages:
{labeled_context}

Question: {question}
Answer:"""
        for attempt in range(3):
            try:
                response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        answer = response.content
        if not isinstance(answer, str):
            answer = "".join(
                part.get("text", "") for part in answer if isinstance(part, dict)
            )
        answer = answer.strip()
        answer = _format_bare_math(answer)
        return {"answer": answer, "sources": sources}
