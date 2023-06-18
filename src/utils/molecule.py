import numpy as np
from src.data.features import get_eigenspectrum


def measure_distance(coulomb1: np.ndarray, coulomb2: np.ndarray) -> float:
    """
    Measure the distance between two molecules

    Args:
        coulomb1: coulomb matrix [n_atoms, n_atoms]
        coulomb2: coulomb matrix [n_atoms, n_atoms]

    Returns:
        distance between two molecules
    """
    spectrum1 = get_eigenspectrum(coulomb1)
    spectrum2 = get_eigenspectrum(coulomb2)
    return np.sqrt(np.sum((spectrum1 - spectrum2) ** 2))
