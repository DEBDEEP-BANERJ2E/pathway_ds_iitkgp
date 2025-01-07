import pytest
from src.backend.rag_pipeline import RagPipeline

def test_rag_pipeline():
    pipeline = RagPipeline()
    query = "Sample query"
    
    result = pipeline.run(query)
    
    assert isinstance(result, str)
    assert len(result) > 0
