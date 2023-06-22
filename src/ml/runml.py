import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error
from src.utils.qm7 import load_qm7
from src.data import features as f
from src.ml.model import get_model


def run_model(model_name: str,
              save_dir: str = "./data",
              represent: str = "coulomb",
              norm_input: bool = True,
              grid_search: bool = False,
              use_gauss_kernel: bool = False,
              shuffle: bool = False,
              verbose: bool = False,
              num_folds: int = 5):

    meas = []
    for fold in range(num_folds):
        data_train, data_test = load_qm7(save_dir=save_dir, fold=fold)
        # TODO: get input X, output T
        X_train, X_test = data_train["X"], data_test["X"]
        y_train, y_test = data_train["T"], data_test["T"]
        # TODO: preprocess X (src.data.features) if needed
        if represent == "randomize":
            X_train, train_expand_ids = f.represent(X_train, type=represent, norm=norm_input)
            y_train = np.concatenate([y_train, y_train[train_expand_ids]], axis=0)
            X_test, test_expand_ids = f.represent(X_test, type=represent, norm=norm_input)
            y_test = np.concatenate([y_test, y_test[test_expand_ids]], axis=0)
        else:
            X_train = f.represent(X_train, type=represent, norm=norm_input)
            X_test = f.represent(X_test, type=represent, norm=norm_input)
        if shuffle:
            inds = np.arange(X_train.shape[0])
            np.random.shuffle(inds)
            X_train = X_train[inds]
            y_train = y_train[inds]
        # X_train = f.Input(X_train).forward(X_train)
        # X_test = f.Input(X_train).forward(X_test)
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        # TODO: load model
        model = get_model(model_name,
                          gauss_kernel=use_gauss_kernel,
                          perform_grid_search=grid_search,
                          represent_type=represent,
                          verbose=verbose)

        # TODO: train model by 5 folds
        model.fit(X_train, y_train)
        if grid_search:
            print(f"Best parameters: {model.best_params_}")  # type: ignore
            print(f"Best score: {model.best_score_}")  # type: ignore
            break
        else:
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            meas.append(mae)
            print(f"Fold {fold}: MAE = {mae:.4f} kcal/mol")
    if not grid_search:
        meas = np.array(meas)
        print(f"\nAverage MAE = {meas.mean():.4f} kcal/mol")
        print(f"Standard deviation = {meas.std():.4f} kcal/mol")


def run(args):
    for model in args.models:
        run_model(model,
                  save_dir=args.data_dir,
                  represent=args.represent,
                  norm_input=not args.not_norm,
                  grid_search=args.grid_search,
                  use_gauss_kernel=args.gauss_kernel,
                  shuffle=args.shuffle,
                  verbose=args.verbose)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "svr", "kr", "mlp", "knn"],
        help="list of models to run",
    )
    parser.add_argument(
        "--represent",
        type=str,
        choices=["coulomb", "sorted", "eigen", "randomize"],
        default="sorted",
        help="representation of input data",
    )
    parser.add_argument(
        "--not-norm",
        action="store_true",
        help="normalize input data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print out training progress",
    )
    parser.add_argument(
        "--data-dir",
        default="../../data",
        help="directory to save data",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="perform grid search",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle data",
    )
    parser.add_argument(
        "--gauss-kernel",
        action="store_true",
        help="use gaussian kernel",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
