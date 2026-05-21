from app.generation.prompts import (
    RAG_PROMPT_TEMPLATE
)


class RAGChain:

    def __init__(
        self,
        retriever,
        llm
    ):
        self.retriever = retriever
        self.llm = llm

    def run(self, question: str):
        context = self.retriever.retrieve(question)

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        answer = self.llm.generate(prompt)

        return answer