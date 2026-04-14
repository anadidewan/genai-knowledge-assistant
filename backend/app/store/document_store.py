import os
import json
import faiss

from app.utils.custom_logger import get_logger
logger = get_logger(__name__)

class DocumentStore:
    def __init__(self):
        self.stored_chunks = []
        self.stored_index = None
        self.graph_data = []

        self.data_dir = "data"
        self.index_path = os.path.join(self.data_dir, "vector_index.faiss")
        self.chunks_path = os.path.join(self.data_dir, "chunks.json")
        self.graph_path = os.path.join(self.data_dir, "graph_data.json")

    def save_to_disk(self):
        os.makedirs(self.data_dir, exist_ok=True)

        if self.stored_index is not None:
            faiss.write_index(self.stored_index, self.index_path)
            logger.info("FAISS index written to %s", self.index_path)


        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.stored_chunks, f, ensure_ascii=False, indent=2)
            logger.info("Chunks written to %s | count=%d", self.chunks_path, len(self.stored_chunks))
        
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(self.graph_data, f, ensure_ascii=False, indent=2)
            logger.info("Graph data written to %s | records=%d", self.graph_path, len(self.graph_data))




    def load_from_disk(self):
        if os.path.exists(self.index_path):
            self.stored_index = faiss.read_index(self.index_path)
            logger.info("FAISS index loaded from %s", self.index_path)
        else:
            logger.info("No existing index file found at %s", self.index_path)


        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.stored_chunks = json.load(f)
            logger.info("Chunks loaded from %s | count=%d", self.chunks_path, len(self.stored_chunks))
        else:
            logger.info("No existing chunks file found at %s", self.chunks_path)

        if os.path.exists(self.graph_path):
            with open(self.graph_path, "r", encoding="utf-8") as f:
                self.graph_data = json.load(f)
            logger.info("Graph data loaded from %s | records=%d", self.graph_path, len(self.graph_data))
        else:
            logger.info("No existing graph file found at %s", self.graph_path)


store = DocumentStore()