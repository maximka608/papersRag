from app.utils.vector_base import KnowledgeBase
from app.utils.embedding import Embeddings
from app.utils.llm import LLM
from app.config import config
import json


def get_emdedding_model():
    return Embeddings()


def get_llm():
    return LLM()


def get_metadata(path):
    titles, texts = [], []
    with open(path, 'rb') as file:
        metadata = json.load(file)
        for data in metadata:
            titles.append(data['title'])
            texts.append(data['text'])
    return texts, titles


def combine_docs(indexes, texts):
    print(indexes.tolist())
    result = ""
    for index in indexes:
        result += texts[index]
    return result


def create_prompt(query, docs):
    system_prompt = f""" You are a language model integrated into a search and
    generation system based on relevant documents (RAG system).
    Your task is to provide answers to the user's queries based on the provided
    documents. Respond only based on the provided documents. Do not make up
    information that is not in the sources. If you use data from a document,
    indicate the document number in square brackets. For example: "This term
    means such-and-such [1]." If there is no information in the documents,
    politely explain that the information is not available. Do not alter the
    content of the sources, convey the information accurately
    If the user requests a list of sources, provide it as a numbered list with
    a brief description of each document.
    Structure the text in a clear way whenever possible, even if formatting is
    limited.

    Example of working with sources:
    Query: "What does LGTM mean?"
    Answer: "LGTM stands for 'Looks Good To Me' [1]."
    List of sources:
    1. Document 1: Brief description of LGTM.
    2. Document 2: Another source mentioning LGTM.

    Follow this format in your responses. User query: {query}. Documents: {docs}
    """

    return system_prompt


def main():
    model = get_emdedding_model()
    llm = get_llm()
    texts, titles = get_metadata(config.PATH_METADATA)
    query = 'What is ChainGAN'
    embedding = model.get_query_embedding(query)
    knowledge_base = KnowledgeBase(config.PATH_FAISS)
    indexes = knowledge_base.search_by_embedding(embedding, 5)[0]
    docs = combine_docs(indexes, texts)
    prompt = create_prompt(query, docs)
    response = llm.generate_response(prompt)
    print(response)


if __name__ == '__main__':
    main()
