"""
Representations of molecules listed in the NeurIPS 2012 paper
`Learning Invariant Representations of Molecules for Atomization Energy Prediction`
"""

import numpy as np


def represent(
    X: np.ndarray,
    type: str = "coulomb",
    norm: bool = True
) -> np.ndarray:
    """
    Represent molecules as coulomb matrices

    Args:
        X: input data [n_samples, n_atoms, n_features]
        type: type of representation

    Returns:
        representation [n_samples, n_atoms, n_atoms]
    """
    if type == "coulomb":
        X_after = X
    elif type == "sorted":
        X_after = sort_coulomb(X)
    elif type == "eigen":
        X_after = get_eigenspectrum(X)
    elif type == "randomize":
        X_after, expand_ids = randomly_sort_coulomb(X)
    else:
        raise ValueError(f"Unknown representation type: {type}")

    if norm:
        X_after = normalize(X_after)
    if type == "randomize":
        return X_after, expand_ids
    return X_after


def get_eigenspectrum(coulomb: np.ndarray, sort: bool = False) -> np.ndarray:
    eigs = np.linalg.eigvalsh(coulomb)
    if sort:
        inds = np.argsort(np.abs(eigs))[::-1]
        n_samples = coulomb.shape[0]
        if eigs.ndim == 2:
            eigs = eigs[np.arange(n_samples)[:, None], inds]
        elif eigs.ndim == 1:
            eigs = eigs[inds]
    return eigs


def encode_atom_charge(atom_charge: np.ndarray):
    unique_atom = np.unique(atom_charge)    # include 0 (no atom)
    unique_atom = np.sort(unique_atom)
    atom_dict = {atom: i for i, atom in enumerate(unique_atom)}
    atom_types = []
    for charge in atom_charge:
        atype = np.array([atom_dict[atom] for atom in charge])
        atom_types.append(atype)
    return np.stack(atom_types)


def sort_coulomb(coulomb: np.ndarray) -> np.ndarray:
    """
    Sort coulomb matrices by descending order
    with respect to the norm-2 of their columns

    Args:
        coulomb: coulomb matrix [n_samples, n_atoms, n_atoms]

    Returns:
        sorted coulomb matrix [n_samples, n_atoms, n_atoms]
    """

    def sort_coulomb_single(coulomb: np.ndarray) -> np.ndarray:
        order = np.argsort(np.linalg.norm(coulomb, axis=0))[::-1]
        return coulomb[order, :][:, order]

    return np.array([sort_coulomb_single(c) for c in coulomb])


def normalize(X: np.ndarray):
    """
    Normalize X by mean and std

    Args:
        X: [n_samples. n_atoms, n_atoms]

    Returns:
        normalized X: [n_samples, n_atoms, n_atoms]
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def randomly_sort_coulomb(
    coulomb_matrix: np.ndarray,
    n_samples: int = 3000,
    n_augmented_samples: int = 4,
    noise_level: float = 1.0
) -> np.ndarray:
    """
    Sampling more coulomb matrices by randomly sorting

    Args:
        coulomb: coulomb matrix [n_samples, n_atoms, n_atoms]
        n_samples: number of samples to randomly sort
        n_augmented_samples: number of samples to augment from a original sample
        noise_level: noise level for sampling
    """
    rand_coulomb = []
    ids = []
    for _ in range(n_samples):
        idx = np.random.randint(low=0, high=coulomb_matrix.shape[0])
        coulomb = coulomb_matrix[idx]
        row_norms = np.asarray([np.linalg.norm(row) for row in coulomb], dtype=np.float32)
        for _ in range(n_augmented_samples):
            e = np.random.normal(loc=0, scale=noise_level, size=row_norms.shape)
            p = np.argsort(row_norms + e)
            new = coulomb[p, :][:, p]
            rand_coulomb.append(new)
        ids.extend([idx] * n_augmented_samples)

    new_coulomb = np.concatenate([coulomb_matrix, np.array(rand_coulomb)], axis=0)
    return new_coulomb, ids
