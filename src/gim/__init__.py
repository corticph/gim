"""
GIM: Gradient-based Interpretability Method for transformer models.

Provides gradient modifications that improve feature attribution quality
for transformer-based language models.
"""
from gim.context import GIM
from gim.explain import explain

__all__ = ["GIM", "explain"]
__version__ = "0.1.0"
