from pathway_integration import PathwayIntegration

def main():
    try:
        # Initialize PathwayIntegration
        integration = PathwayIntegration(embedder_model="allenai/scibert_scivocab_uncased")

        # Test stream_data (ensure it connects to Google Drive properly)
        print("Testing data streaming...")
        data_stream = integration.stream_data()
        print("Streamed Data:", data_stream)

        # Test query_statistics (retrieve document stats)
        print("Testing statistics query...")
        stats = integration.query_statistics()
        print("Statistics Query Result:", stats)

        # Test a simple embedding query
        print("Testing similar embedding query...")
        query_embedding = [0.1] * 768  # Dummy embedding
        similar_docs = integration.query_similar_embeddings(query_embedding)
        print("Similar Documents:", similar_docs)
        
        print("All tests passed successfully!")

    except Exception as e:
        print("An error occurred during testing:", e)

if __name__ == "__main__":
    main()
