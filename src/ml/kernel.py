import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances


def gaussian_kernel(X, Y=None, sigma=41):
    """
    Compute the Gaussian kernel between X and Y

            K(x, y) = exp(- ||x - y||^2 / (2 * sigma^2))

    for each pair of rows x in X and y in Y.

    Args:
        X: array of shape [n_samples_X, n_features]
        Y: array of shape [n_samples_Y, n_features], default=None
            An optional second array of samples. If None, use X instead.
        sigma: float, default=None
            if None, defaults to 1.0 / n_features

    """
    X, Y = check_pairwise_arrays(X, Y)
    if sigma is None:
        sigma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K = -K / (2 * sigma**2)
    np.exp(K, K)  # exponentiate K in-place
    return K
