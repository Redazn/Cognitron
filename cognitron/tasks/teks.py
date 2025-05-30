from . import register_solver

@register_solver('text_classification')
def solve_text_classification(agi):
    data = agi.data
    return f"Klasifikasi teks untuk: {data}"
