import os
import json
import faiss


class DocumentStore:
    def __init__(self):
        self.stored_chunks = []
        self.stored_index = None
        # self.graph_data = []

        self.data_dir = "data"
        self.index_path = os.path.join(self.data_dir, "vector_index.faiss")
        self.chunks_path = os.path.join(self.data_dir, "chunks.json")
        # self.graph_path = os.path.join(self.data_dir, "graph_data.json")

    def save_to_disk(self):
        os.makedirs(self.data_dir, exist_ok=True)

        if self.stored_index is not None:
            faiss.write_index(self.stored_index, self.index_path)

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.stored_chunks, f, ensure_ascii=False, indent=2)

        # with open(self.graph_path, "w", encoding="utf-8") as f:
        #     json.dump(self.graph_data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self):
        if os.path.exists(self.index_path):
            self.stored_index = faiss.read_index(self.index_path)

        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.stored_chunks = json.load(f)

        # if os.path.exists(self.graph_path):
        #     with open(self.graph_path, "r", encoding="utf-8") as f:
        #         self.graph_data = json.load(f)


store = DocumentStore()