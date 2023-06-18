from multiprocessing import cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from src.ml.kernel import gaussian_kernel


def get_model(model_name,
              gauss_kernel: bool = False,
              verbose: bool = False,
              perform_grid_search: bool = False,
              represent_type: str = "eigen",
              **kwargs):
    """
    Get model from model name
    """
    if model_name == "linear":
        return LinearRegression()
    elif model_name == "knn":
        return KNeighborsRegressor(**kwargs)
    elif model_name == "svr":
        if gauss_kernel:
            kernel = gaussian_kernel
        else:
            kernel = "rbf"
        if perform_grid_search:
            svr = SVR(kernel=kernel, verbose=verbose)
            param_grid = {
                "gamma": [10**-4, 10**-3, 10**-2, 10**-1, 1, 5],
                "C": [10**-2, 1, 5, 10, 20]
            }
            return GridSearchCV(svr,
                                param_grid,
                                cv=5,
                                n_jobs=cpu_count() // 2,
                                verbose=verbose,
                                scoring="neg_mean_absolute_error")
        if represent_type == "eigen":
            C = 1000
            gamma = 0.001
        elif represent_type == "sorted":    # norm
            C = 1000
            gamma = 0.001
        elif represent_type == "randomize":
            C = 0.0001
            gamma = 0.001
        else:
            C = 0.0001
            gamma = 0.001
        return SVR(kernel=kernel, gamma=gamma, C=C, verbose=verbose, **kwargs)
    elif model_name == "kr":
        if gauss_kernel:
            kernel = gaussian_kernel
        else:
            kernel = "rbf"
        if perform_grid_search:
            kr = KernelRidge(kernel=kernel)
            param_grid = {
                "alpha": [10**-10, 10**-8, 10**-6, 10**-4, 10**-2, 1],
                "gamma": [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
            }
            return GridSearchCV(kr,
                                param_grid,
                                cv=5,
                                n_jobs=cpu_count() // 2,
                                verbose=verbose,
                                scoring="neg_mean_absolute_error")
        if represent_type == "eigen":
            alpha = 0.0001
            gamma = 0.001
        elif represent_type == "sorted":
            alpha = 0.0001
            gamma = 0.0001
        elif represent_type == "randomize":
            alpha = 0.0001
            gamma = 0.0001
        else:
            alpha = 0.0001
            gamma = 0.001
        return KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma, **kwargs)
    else:
        raise ValueError("Unknown model name")
