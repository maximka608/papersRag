import faiss, json
from app.script.preprocessing_text import Preprocessor
from rank_bm25 import BM25Okapi
import numpy as np

class KnowledgeBase:
    def __init__(self, faiss_path, preprocessing_path) -> None:
        self.BM25_model = BM25Okapi(self._load(preprocessing_path))
        self.vector_base = faiss.read_index(faiss_path)

    def _load(self, path):
        with open(path, 'rb') as file:
            data = json.load(file)
            return data

    def search_by_BM25(self, query, k=5):
        preprocessor = Preprocessor()
        prep_query = preprocessor.preprocessing_text(query)
        doc_scores = self.BM25_model.get_scores(prep_query)
        sorted_docs = np.argsort(-doc_scores)
        return sorted_docs[:k].tolist()

    def search_by_embedding(self, embedding, k):
        _, indexes = self.vector_base.search(embedding, k)
        return indexes


