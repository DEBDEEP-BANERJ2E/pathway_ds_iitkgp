from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ConferenceRecommender:
    def __init__(self, vector_store):
        self.vector_store = vector_store  # Pathway VectorStore instance

    def recommend_conferences(self, paper_embeddings):
        """Recommend conferences based on cosine similarity."""
        recommendations = []
        for paper_embedding in paper_embeddings:
            similar_vectors = self.vector_store.query(paper_embedding, top_k=3)
            recommendations.append(similar_vectors)
        return recommendations
