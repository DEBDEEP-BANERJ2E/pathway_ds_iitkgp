from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from backend.publishability import PublishabilityClassifier
from backend.conference import ConferenceRecommender
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="src/frontend/templates")

# Serve static files (CSS, JavaScript, images)
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")

# Initialize the necessary components
# Example: You will need to replace these with actual embeddings and vector store
publishability_classifier = PublishabilityClassifier()
conference_recommender = ConferenceRecommender(vector_store=None)  # Initialize with the actual vector store

@app.get("/")
def index(request: Request):
    """
    Serve the landing page (index.html)
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results")
def results(request: Request):
    """
    Serve the results page with publishability and conference recommendations
    """
    # Example paper text (replace with actual text or embeddings from your preprocessing step)
    paper_text = "Sample research paper text"

    # Simulate getting embeddings for the paper
    paper_embeddings = np.random.rand(1, 300)  # Example: Replace with actual embeddings

    # Predict publishability
    publishability = publishability_classifier.predict(paper_embeddings)

    # Get conference recommendations
    recommended_conferences = conference_recommender.recommend_conferences(paper_embeddings)

    # Flatten the list of conference recommendations for easier rendering
    # You may need to adjust the format based on your vector store results
    flattened_conferences = [item for sublist in recommended_conferences for item in sublist]

    # Filter out empty strings and None from the conferences list
    filtered_conferences = [conf for conf in flattened_conferences if conf]

    # Return the results to the frontend template
    return templates.TemplateResponse("results.html", {
        "request": request,
        "publishability": publishability[0],  # Assuming the model returns an array-like structure
        "conferences": filtered_conferences
    })

# You can define other routes and logic here as needed
