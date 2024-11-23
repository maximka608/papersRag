import nltk, json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# from app.main import get_metadata
from app.config import config


class Preprocessor:
    def _tokenize(self, text):
        text = text.lower().split(' ')
        return text

    def preprocessing_text(self, doc):
        tokens = self._tokenize(doc)

        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if not token in stop_words]

        stemmer = PorterStemmer()
        stemmed_tokes = [stemmer.stem(filtered_token) for filtered_token in filtered_tokens]
        preprocess_text =  " ".join(stemmed_tokes)
        return preprocess_text

    def _save(self, docs):
        with open("../preprocessing_text.json", "w") as f:
            json.dump(docs, f, indent=4)

    def preprocessing(self, docs):
        preprocessed_docs = [self.preprocessing_text(doc) for doc in docs]
        self._save(preprocessed_docs)

if __name__ == '__main__':
    texts, _ = get_metadata(config.PATH_METADATA)
    preprocessor = Preprocessor()
    preprocessor.preprocessing(texts)
