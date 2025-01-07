import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from transformers import pipeline
from typing import List, Tuple

class PathwayIntegration:
    def __init__(self, embedder_model="allenai/scibert_scivocab_uncased"):
        # Initialize connectors
        self.google_drive_connector = pw.io.gdrive("https://drive.google.com/drive/folders/1Z8z4craj36ighb8hzUzeM76OOgpUdsKr")
        
        # Set up embedding pipeline (using SciBERT or any other model)
        self.embedder = pipeline("feature-extraction", model=embedder_model, tokenizer=embedder_model)

        # Initialize VectorStoreServer
        self.vector_store_server = VectorStoreServer(
            docs=self.stream_data(),
            embedder=self._embedding_udf,
            parser=self._basic_parser,
        )

    def stream_data(self):
        """Stream papers data using Pathway connectors."""
        return self.google_drive_connector.read()

    def _embedding_udf(self, text: str) -> List[float]:
        """Embed text using the embedding pipeline."""
        embeddings = self.embedder(text)
        return embeddings[0][0]  # Flatten the output

    def _basic_parser(self, content: bytes) -> List[Tuple[str, dict]]:
        """Basic parser to handle file content."""
        text = content.decode("utf-8")
        return [(text, {})]

    def query_statistics(self):
        """Query document statistics."""
        query_table = pw.Table.from_dicts([{"id": 1}])  # Example query schema
        result_table = self.vector_store_server.statistics_query(query_table)
        return result_table

    def query_similar_embeddings(self, query_embedding: List[float]):
        """Query nearest neighbors for an embedding."""
        return self.vector_store_server._graph["knn_index"].query(query_embedding, top_k=3)

    def query_inputs(self, metadata_filter=None, filepath_globpattern=None):
        """Query inputs using metadata filtering."""
        query_table = pw.Table.from_dicts([
            {"id": 1, "metadata_filter": metadata_filter, "filepath_globpattern": filepath_globpattern}
        ])
        result_table = self.vector_store_server.inputs_query(query_table)
        return result_table
