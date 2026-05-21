RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the context does not contain the answer,
say explicitly:
"I could not find the answer in the provided context."

<context>
{context}
</context>

<question>
{question}
</question>

Answer:
"""