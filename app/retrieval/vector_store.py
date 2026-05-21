import faiss
import pickle
import numpy as np
from pathlib import Path


class FAISSVectorStore:

    def __init__(self):
        self.index = None
        self.documents = []

    def build_index(self, embeddings, documents):
        dimension = len(embeddings[0])

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings).astype("float32"))

        self.documents = documents

    def save_index(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            f"{path}/faiss.index"
        )

        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self, path):
        self.index = faiss.read_index(
            f"{path}/faiss.index"
        )

        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query_vector, k=4):
        distances, indices = self.index.search(
            np.array([query_vector]).astype("float32"),
            k
        )

        return [self.documents[i] for i in indices[0]]