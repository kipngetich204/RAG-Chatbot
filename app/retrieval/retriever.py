class RAGRetriever:

    def __init__(self, embedder, vector_store, k=4):
        self.embedder = embedder
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, question: str):
        query_vector = self.embedder.embed_query(question)

        docs = self.vector_store.search(
            query_vector,
            self.k
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        return context