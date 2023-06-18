"""
Representations of molecules listed in the NeurIPS 2012 paper
`Learning Invariant Representations of Molecules for Atomization Energy Prediction`
"""

import numpy as np
import torch
from typing import Union


def represent(
    X: Union[np.ndarray, torch.Tensor],
    type: str = "coulomb",
    norm: bool = True
) -> Union[np.ndarray, torch.Tensor]:
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
        X_after = randomly_sort_coulomb(X)
    else:
        raise ValueError(f"Unknown representation type: {type}")

    if norm:
        X_after = normalize(X_after)
    return X_after


class Input:

    def __init__(self, X):
        self.step = 1.0
        self.noise = 1.0
        self.triuind = (np.arange(23)[:, np.newaxis]
                        <= np.arange(23)[np.newaxis, :]).flatten()
        self.max = 0
        for _ in range(10):
            self.max = np.maximum(self.max, self.realize(X).max(axis=0))
        X = self.expand(self.realize(X))
        self.nbout = X.shape[1]
        self.mean = X.mean(axis=0)
        self.std = (X - self.mean).std()

    def realize(self, X):

        def _realize_(x):
            inds = np.argsort(
                -(x**2).sum(axis=0)**.5 +
                np.random.normal(0, self.noise, x[0].shape))
            x = x[inds, :][:, inds] * 1
            x = x.flatten()[self.triuind]
            return x

        return np.array([_realize_(z) for z in X])

    def expand(self, X):
        Xexp = []
        for i in range(X.shape[1]):
            for k in np.arange(0, int(self.max[i]) + int(self.step), int(self.step)):
                Xexp += [np.tanh((X[:, i] - k) / self.step)]
        return np.array(Xexp).T

    def normalize(self, X):
        return (X - self.mean) / self.std

    def forward(self, X):
        print("0:", X.shape)
        X = self.realize(X)
        print("1:", X.shape)
        X = self.expand(X)
        print("2:", X.shape)
        X = self.normalize(X)
        print("3:", X.shape)
        return X.astype('float32')


def get_eigenspectrum(coulomb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    is_torch = False
    if isinstance(coulomb, torch.Tensor):
        is_torch = True
        coulomb = coulomb.numpy()
    values = np.linalg.eigvalsh(coulomb)
    values = np.sort(values)[::-1]
    if is_torch:
        return torch.from_numpy(values)
    return values


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
    n_samples: int = 1,
    n_augmented_samples: int = 1,
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
    for _ in range(n_samples):
        idx = np.random.randint(low=0, high=coulomb_matrix.shape[0])
        coulomb = coulomb_matrix[idx]
        row_norms = np.asarray([np.linalg.norm(row) for row in coulomb], dtype=np.float32)
        for _ in range(n_augmented_samples):
            e = np.random.normal(loc=0, scale=noise_level, size=row_norms.shape)
            p = np.argsort(row_norms + e)
            new = coulomb[p, :][:, p]
            rand_coulomb.append(new)
    # TODO: process y to match the number of samples of X

    return np.array(rand_coulomb)
