from . import register_solver

@register_solver('arc')
def solve_arc(agi):
    """Solver khusus untuk tugas ARC."""
    # Contoh implementasi sederhana
    data = agi.data
    pattern = str(data)  # Misalnya, data adalah grid
    solution, _ = agi.memory.retrieve(pattern)
    if solution is None:
        solution = "Solusi default untuk ARC"
        agi.memory.store(pattern, solution, 0.5)
    return solution
