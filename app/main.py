from app.utils.vector_base import KnowledgeBase
from app.utils.embedding import Embeddings
from app.utils.llm import LLM
from app.config import config
import gradio as gr
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
    result = ""
    for i, index in enumerate(indexes):
        result += " [" + str(i + 1) + "] " + texts[index]
    return result


def print_docs(indexes, texts):
    for i, index in enumerate(indexes):
        print(" [" + str(i + 1) + "] " + texts[index])


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
    Structure the text in a clear way whenever possible, even if formatting is
    limited.
    For example: 
    User query: ML.
    Documents:  
    [1] es of ML models.  
    [2] The rapid escalation of applying Machine Learning (ML) in various domains has led to paying more attention to the quality of ML components. There is then a growth of techniques and tools aiming at improving the quality of ML components and integrating them.  
    
    Machine Learning (ML) is increasingly applied across various domains, leading to a focus on the quality of ML components and the development of techniques to improve and integrate them [2]
   
    Follow this format in your responses and print all documents. User query: {query}. Documents: {docs}
    """

    return system_prompt


def main(query, search_types):
    model, llm = get_emdedding_model(), get_llm()
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
    return response


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
        ],
        outputs="text",
        title="PaperRAG",
        description="RAG system for scientific papers with selectable search types"
    )
    demo.launch()