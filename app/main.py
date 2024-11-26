from app.utils.vector_base import KnowledgeBase
from app.utils.embedding import Embeddings
from app.utils.llm import LLM
from app.config import config
import gradio as gr
import json


def get_emdedding_model():
    return Embeddings()


def get_llm(api_key):
    return LLM(api_key)


def get_metadata(path):
    titles, texts = [], []
    with open(path, 'rb') as file:
        metadata = json.load(file)
        for data in metadata:
            titles.append(data['title'])
            texts.append(data['text'])
    return texts, titles

def combine_docs(indexes, texts):
    result = ""
    for i, index in enumerate(indexes):
        result += " [" + str(i + 1) + "] " + texts[index]
    return result


def print_docs(indexes, texts):
    for i, index in enumerate(indexes):
        print(" [" + str(i + 1) + "] " + texts[index])


def create_prompt(query, docs):
    system_prompt = f"""You are a language model integrated into a search and generation system based on relevant documents (RAG system).
    Your task is to provide answers to the user's queries based solely on the provided documents.
    If the information required to answer the user's question is available in the documents, use it, and refer to the document from which it was sourced by indicating its number in square brackets. For example: 
    "This term means such-and-such [1]."
    Ensure that the citation clearly refers to the relevant document and is placed directly after the information from the source.

    If the information is not present in the documents, kindly explain that the information is not available, and do not speculate or make up information.

    Do not alter the content or meaning of the sources. Convey the information accurately and structure your response clearly, even if the formatting options are limited.

    User query: {query}
    Documents:
    {docs}
    """
    return system_prompt


def main(query, search_types, llm_api_key):
    model, llm = get_emdedding_model(), get_llm(llm_api_key)
    texts, titles = get_metadata(config.PATH_METADATA)
    embedding = model.get_query_embedding(query)

    knowledge_base = KnowledgeBase(config.PATH_FAISS, config.PATH_PREPROCESSING_TEXT)
    vector_search = []
    bm25_search = []

    if "Vector" in search_types:
        vector_search = knowledge_base.search_by_embedding(embedding, 5)[0].tolist()
    if "BM25" in search_types:
        bm25_search = knowledge_base.search_by_BM25(query, 5)

    docs = combine_docs(vector_search + bm25_search, texts)
    prompt = create_prompt(query, docs)

    response = llm.generate_response(prompt)
    return response, docs


if __name__ == '__main__':
    demo = gr.Interface(
        fn=main,
        inputs=[
            gr.Textbox(label="Enter your query"),
            gr.CheckboxGroup(
                choices=["Vector", "BM25"],
                label="Search Types",
                value=["Vector", "BM25"]
            ),
            gr.Textbox(label="LLM API Key", placeholder="Enter LLM API Key", type="password")
        ],
        outputs=[
            gr.Textbox(label="LLM Response"),
            gr.Textbox(label="Combined Documents")
        ],
        title="PaperRAG",
        description="RAG system for scientific papers with selectable search types"
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)
