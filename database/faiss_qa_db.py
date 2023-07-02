import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FaissQAIndex:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data = []

    def add_data(self, data):
        self.data.append(data)

    def build_index(self):
        question_vectors = np.array(
            [self.model.encode(question) for question, _ in self.data])
        self.index = faiss.IndexFlatL2(question_vectors.shape[1])
        self.index.add(question_vectors)

    def search(self, input_question, k=1):
        input_question_vector = self.model.encode(input_question)
        input_question_vector = np.array([input_question_vector])
        distances, indices = self.index.search(input_question_vector, k)
        results = []
        for i in range(k):
            result = {
                "question": self.data[indices[0][i]][0],
                "answer": self.data[indices[0][i]][1],
                "distance": distances[0][i]
            }
            results.append(result)
        return results

    def delete_data(self, question):
        self.data = [(q, a) for q, a in self.data if q != question]
        self.build_index()

    def update_data(self, question, new_answer):
        for i, (q, a) in enumerate(self.data):
            if q == question:
                self.data[i] = (question, new_answer)
        self.build_index()
