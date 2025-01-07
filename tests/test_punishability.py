import pytest
from src.backend.publishability import PublishabilityClassifier

def test_publishability_classifier():
    classifier = PublishabilityClassifier()
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Example embeddings
    labels = [1, 0]  # Example labels (1 = publishable, 0 = not publishable)
    
    # Train the model
    model = classifier.train_model(embeddings, labels)
    
    # Test predictions
    predictions = classifier.predict(embeddings)
    assert len(predictions) == 2
    assert predictions[0] in [0, 1]
    assert predictions[1] in [0, 1]
