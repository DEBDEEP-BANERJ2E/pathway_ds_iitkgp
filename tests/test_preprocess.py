import pytest
from src.backend.preprocess import preprocess_paper_data

def test_preprocess_paper_data():
    raw_data = [
        {"title": "Paper A", "author": "Author 1", "abstract": "Abstract A"},
        {"title": "Paper B", "author": "Author 2", "abstract": "Abstract B"}
    ]
    processed_data = preprocess_paper_data(raw_data)
    assert isinstance(processed_data, list)
    assert len(processed_data) == 2
    assert "title" in processed_data[0]
    assert "author" in processed_data[0]
