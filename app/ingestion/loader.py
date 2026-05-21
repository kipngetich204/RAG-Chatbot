from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)


class DocumentLoader:

    @staticmethod
    def load_directory(path: str):
        documents = []

        for file in Path(path).glob("*"):
            if file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix == ".txt":
                loader = TextLoader(str(file))
            else:
                continue

            documents.extend(loader.load())
           

        return documents