import pandas as pd

def load_papers(filepath):
    """Load papers data from a CSV file."""
    return pd.read_csv(filepath)

def save_results(data, filepath):
    """Save results to a CSV file."""
    data.to_csv(filepath, index=False)
