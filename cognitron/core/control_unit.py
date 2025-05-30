class MetaControlUnit:
    def __init__(self):
        pass

    def decide(self, options, utility_scores):
        """Memilih opsi terbaik berdasarkan skor utilitas."""
        return max(options, key=lambda x: utility_scores[x])
