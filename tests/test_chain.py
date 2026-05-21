from app.generation.prompts import (
    RAG_PROMPT_TEMPLATE
)


def test_prompt():
    prompt = RAG_PROMPT_TEMPLATE.format(
        context="AI is artificial intelligence.",
        question="What is AI?"
    )

    assert "AI" in prompt