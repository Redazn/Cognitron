solvers = {}

def register_solver(task_name):
    """Dekorator untuk mendaftarkan solver tugas."""
    def decorator(func):
        solvers[task_name] = func
        return func
    return decorator

def get_solver(task_name):
    """Mengambil solver berdasarkan nama tugas."""
    return solvers.get(task_name, lambda x: "Tugas tidak didukung")
