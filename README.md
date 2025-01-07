# Conference Paper Recommendation System

This project aims to classify the publishability of research papers and recommend conferences based on the paper's embeddings. It uses machine learning models for publishability classification and cosine similarity for conference recommendations.

## Project Structure

- **data/**: Contains the dataset and benchmark reference files.
  - `papers.csv`: Main dataset with research papers.
  - `reference_papers.csv`: Benchmark reference papers for conference recommendations.
  - `embeddings/`: Precomputed embeddings if needed.

- **src/**: Contains the backend logic and frontend application.
  - **backend/**: The backend logic for preprocessing, publishability classification, and conference recommendation.
  - **frontend/**: The frontend application with HTML templates and static files.

- **tests/**: Unit tests for preprocessing, publishability classification, and conference recommendation.
- **results/**: Output results for publishability and conference recommendations.
- **notebooks/**: Jupyter notebooks for generating embeddings and testing integration.

## How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the FastAPI server:
    ```bash
    uvicorn src.app:app --reload
    ```

3. Access the frontend:
    Open `http://localhost:8000` in your browser.

## Running Tests

To run the tests, use `pytest`:

```bash
pytest
# pathway_ds_iitkgp
