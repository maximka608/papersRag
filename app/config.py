from pathlib import Path


class Config:
    PATH_FAISS = str(Path(__file__).parent / 'faiss_index.faiss')
    PATH_METADATA = str(Path(__file__).parent / 'metadata.json')
    PATH_PREPROCESSING_TEXT = str(Path(__file__).parent / 'preprocessing_text.json')


config = Config()
