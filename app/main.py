from app.config import settings

from app.ingestion.loader import (
    DocumentLoader
)

from app.ingestion.splitter import (
    RecursiveTextSplitter
)

from app.ingestion.embedder import (
    EmbeddingModel
)

from app.retrieval.vector_store import (
    FAISSVectorStore
)

from app.retrieval.retriever import (
    RAGRetriever
)

from app.generation.llm import (
    OllamaLLM
)

from app.generation.chain import (
    RAGChain
)


def build_index_from_dir():

    print("Loading documents...")

    docs = DocumentLoader.load_directory(
        settings.RAW_DATA_DIR
    )

    splitter = RecursiveTextSplitter(
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP
    )

    chunks = splitter.split(docs)

    print(f"Created {len(chunks)} chunks")

    embedder = EmbeddingModel(
        settings.EMBEDDING_MODEL
    )

    embeddings = embedder.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    vector_store = FAISSVectorStore()

    vector_store.build_index(
        embeddings,
        chunks
    )

    vector_store.save_index(
        settings.INDEX_DIR
    )

    print("Index saved successfully")


def chat():

    embedder = EmbeddingModel(
        settings.EMBEDDING_MODEL
    )

    vector_store = FAISSVectorStore()

    vector_store.load_index(
        settings.INDEX_DIR
    )

    retriever = RAGRetriever(
        embedder,
        vector_store,
        settings.TOP_K
    )

    llm = OllamaLLM(
        settings.LLM_MODEL,
        settings.OLLAMA_BASE_URL
    )

    rag_chain = RAGChain(
        retriever,
        llm
    )

    print("\nRAG Chatbot Ready")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Ask any question relating to document context: ")

        if question.lower() == "exit":
            break

        answer = rag_chain.run(question)

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":

    build_index_from_dir()

    chat()