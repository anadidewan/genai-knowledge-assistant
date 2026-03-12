class InMemoryStore:
    def __init__(self):
        self.stored_chunks = []
        self.stored_index = None


store = InMemoryStore()