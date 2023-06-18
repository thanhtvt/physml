from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


def get_model(model_name, verbose: bool = False, **kwargs):
    """
    Get model from model name
    """
    if model_name == "linear":
        return LinearRegression()
    elif model_name == "svr":
        return SVR(kernel="poly", gamma="scale", verbose=verbose, **kwargs)
    elif model_name == "kr":
        return KernelRidge(kernel="rbf", **kwargs)
    elif model_name == "gp":
        return GaussianProcessRegressor(**kwargs)
    elif model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=[1024, 256, 128],
            activation="logistic",
            solver="adam",
            random_state=0,
            max_iter=10000,
            verbose=verbose,
            learning_rate="adaptive",
            learning_rate_init=0.01,
            early_stopping=True,
            tol=1e-6,
            **kwargs
        )
    else:
        raise ValueError("Unknown model name")
