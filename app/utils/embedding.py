from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, revision="main")

    def get_query_embedding(self, query):
        query_embed = self.model.encode([query], normalize_embeddings=True)
        return query_embed

    def get_embeddings(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
