import faiss, json
from datasets import load_dataset
from app.utils.embedding import Embeddings

def get_chunkes(docs, size):
    chunked_texts, metadata= [], []

    for _, text in enumerate(docs):
        for i in range(0, len(text['abstract']), size):
            chunk = text['abstract'][i:i + size]

            chunked_texts.append(chunk)
            metadata.append({'title': text['title'], 'text': chunk})

    return chunked_texts, metadata


def create_base(docs, model: Embeddings):
    chunks, metadata = get_chunkes(docs, 256)
    dimension = 384
    embeddings = model.get_embeddings(chunks)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, metadata


def main():
    data = load_dataset("aalksii/ml-arxiv-papers")
    articles = data['train'].select(range(10000))
    embed_model = Embeddings()

    vector_base, metadata = create_base(articles, embed_model)
    faiss.write_index(vector_base, "faiss_index.faiss")

    with open("../metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    main()
