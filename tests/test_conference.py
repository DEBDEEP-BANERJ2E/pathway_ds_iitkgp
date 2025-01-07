import pytest
from src.backend.conference import ConferenceRecommender

def test_conference_recommender():
    vector_store = None  # Use an appropriate mock or instance of your vector store
    recommender = ConferenceRecommender(vector_store)
    paper_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Example paper embeddings
    
    recommendations = recommender.recommend_conferences(paper_embeddings)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) == 2
