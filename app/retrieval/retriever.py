class RAGRetriever:

    def __init__(self, embedder, vector_store, k=4):
        self.embedder = embedder
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, question: str):
        try:
            query_vector = self.embedder.embed_query(question)

            docs = self.vector_store.search(
                query_vector,
                self.k
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            sources = []


            for doc in docs:
                source = doc.metadata.get("source","Unknown")
                page= doc.metadata.get("page", "N/A")
                sources.append(f"{source} (page {page})")



            return {
                "context": context,
                "sources": sources
            }
        
        except Exception as exc:
            raise RuntimeError(
                f"Error retrieving from vector store: {exc}") from exc
