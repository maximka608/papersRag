import faiss, json
from datasets import load_dataset
from app.utils.embedding import Embeddings
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

def get_chunks(docs, max_tokens):
    chunked_texts, metadata = [], []

    for _, text in enumerate(docs):
        sentences = sent_tokenize(text['abstract'])
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_tokens:
                chunked_texts.append(" ".join(current_chunk))
                metadata.append({'title': text['title'], 'text': " ".join(current_chunk)})
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunked_texts.append(" ".join(current_chunk))
            metadata.append({'title': text['title'], 'text': " ".join(current_chunk)})

    return chunked_texts, metadata


def create_base(docs, model: Embeddings):
    chunks, metadata = get_chunks(docs, 256)
    dimension = 384
    embeddings = model.get_embeddings(chunks)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, metadata


def main():
    data = load_dataset("aalksii/ml-arxiv-papers")
    articles = data['train'].select(range(1000))
    embed_model = Embeddings()

    vector_base, metadata = create_base(articles, embed_model)
    faiss.write_index(vector_base, "faiss_index.faiss")

    with open("../metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    main()
