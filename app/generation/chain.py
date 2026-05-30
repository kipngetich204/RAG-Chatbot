from typing import Generator

from app.generation.prompts import (
    RAG_PROMPT_TEMPLATE,
)


class RAGChain:

    def __init__(
        self,
        retriever,
        llm,
        memory,
    ):
        self.retriever = retriever
        self.llm = llm
        self.memory = memory

    def _prepare_context(
        self,
        question: str,
    ):

        memory_context = (
            self.memory.get_context()
        )

        if memory_context.strip():

            rewrite_prompt = f"""
                    Conversation History:
                    {memory_context}

                    Current Question:
                    {question}

                    Rewrite the current question into a
                    clear standalone question.

                    Only return the rewritten question.
                    """

            standalone_question = (
                self.llm.generate(
                    rewrite_prompt
                ).strip()
            )

        else:
            standalone_question = question

        retrieved = self.retriever.retrieve(
            standalone_question
        )

        context = retrieved.get(
            "context",
            ""
        )

        #print(f"context: {context}")

        sources = retrieved.get(
            "sources",
            []
        )

        full_context = f"""
Conversation History:
{memory_context}

Retrieved Context:
{context}
"""

        prompt = (
            RAG_PROMPT_TEMPLATE.format(
                context=full_context,
                question=standalone_question,
            )
        )

        return prompt, sources

    def run(
        self,
        question: str,
    ) -> dict:

        prompt, sources = (
            self._prepare_context(
                question
            )
        )

        answer = self.llm.generate(
            prompt
        )

        self.memory.add(
            "User",
            question,
        )

        self.memory.add(
            "Assistant",
            answer,
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    def stream_run(
        self,
        question: str,
    ) -> Generator[dict, None, None]:

        prompt, sources = (
            self._prepare_context(
                question
            )
        )

        answer_parts = []

        for token in self.llm.stream_generate(
            prompt
        ):

            answer_parts.append(
                token
            )

            yield {
                "type": "token",
                "text": token,
            }

        answer = "".join(
            answer_parts
        )

        self.memory.add(
            "User",
            question,
        )

        self.memory.add(
            "Assistant",
            answer,
        )

        yield {
            "type": "end",
            "sources": sources,
        }