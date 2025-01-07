import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class Preprocessor:
    def __init__(self):
        # Load SciBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    def clean_text(self, text):
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        return text.lower().strip()

    def generate_embeddings(self, texts):
        """Generate SciBERT embeddings for a list of texts."""
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def preprocess_papers(self, papers_df):
        """Preprocess papers and generate embeddings."""
        papers_df['cleaned_abstract'] = papers_df['abstract'].apply(self.clean_text)
        papers_df['embeddings'] = list(self.generate_embeddings(papers_df['cleaned_abstract'].tolist()))
        return papers_df
