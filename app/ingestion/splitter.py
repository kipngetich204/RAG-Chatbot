from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveTextSplitter:

    def __init__(self, chunk_size=512, overlap=64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def split(self, docs):
        return self.splitter.split_documents(docs)