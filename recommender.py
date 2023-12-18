from sentence_transformers import SentenceTransformer

class Recommender:
    def __init__(self, model_path = "paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path)
    def recommend(self, tweets):
        return ""