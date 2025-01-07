from transformers import pipeline

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store  # Pathway VectorStore instance
        self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

    def retrieve_documents(self, query_embedding):
        """Retrieve documents from the VectorStore."""
        return self.vector_store.query(query_embedding, top_k=5)

    def generate_justification(self, paper, retrieved_docs):
        """Generate justifications using the RAG pipeline."""
        input_text = f"Paper: {paper} \n Retrieved Docs: {retrieved_docs}"
        output = self.generator(input_text, max_length=150, do_sample=False)
        return output[0]['generated_text']
