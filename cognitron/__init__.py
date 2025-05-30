from .core.memory import CognitiveMemory
from .core.control_unit import MetaControlUnit
from .tasks import get_solver

class Cognitron:
    def __init__(self, task, memory=None, control_unit=None):
        self.task = task
        self.memory = memory if memory else CognitiveMemory()
        self.control_unit = control_unit if control_unit else MetaControlUnit()
        self.data = None

    def load_data(self, data_path, data_type='grid'):
        """Memuat data dari path (sederhana untuk contoh)."""
        # Implementasi nyata akan tergantung pada format data
        self.data = data_path  # Misalnya, ini bisa menjadi grid, teks, atau gambar

    def solve(self):
        """Menjalankan solver untuk tugas yang ditentukan."""
        solver = get_solver(self.task)
        return solver(self)
