import faiss


class KnowledgeBase:
    def __init__(self, faiss_path) -> None:
        self.vector_base = faiss.read_index(faiss_path)

    def search_by_embedding(self, embedding, k):
        _, indexes = self.vector_base.search(embedding, k)
        return indexes
