from app.ingestion.embedder import (
    EmbeddingModel
)


def test_embeddings():
    embedder = EmbeddingModel()

    vec = embedder.embed_query(
        "What is RAG?"
    )

    assert len(vec) > 0