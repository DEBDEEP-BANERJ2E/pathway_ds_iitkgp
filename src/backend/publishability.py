from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

class PublishabilityClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, embeddings, labels):
        """Train the publishability classifier."""
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("F1 Score:", f1_score(y_test, predictions))
        print("Classification Report:\n", classification_report(y_test, predictions))
        return self.model

    def save_model(self, filepath):
        """Save the trained model to disk."""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load a trained model from disk."""
        self.model = joblib.load(filepath)

    def predict(self, embeddings):
        """Predict publishability for new data."""
        return self.model.predict(embeddings)
