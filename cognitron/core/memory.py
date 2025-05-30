class CognitiveMemory:
    def __init__(self):
        self.storage = {}

    def store(self, pattern, solution, uncertainty):
        """Menyimpan pola, solusi, dan ketidakpastian."""
        self.storage[pattern] = (solution, uncertainty)

    def retrieve(self, pattern):
        """Mengambil solusi berdasarkan pola."""
        return self.storage.get(pattern, (None, None))
