import argparse
from typing import List
from sklearn.metrics import mean_absolute_error
from src.utils.qm7 import load_qm7
from src.data import features as f
from src.ml.model import get_model


def run_model(model_name: str,
              save_dir: str = "./data",
              represent: str = "coulomb",
              norm_input: bool = True,
              verbose: bool = False,
              num_folds: int = 5):
    for fold in range(num_folds):
        data_train, data_test = load_qm7(save_dir=save_dir, fold=fold)
        # TODO: get input X, output T
        X_train, X_test = data_train["X"], data_test["X"]
        y_train, y_test = data_train["T"], data_test["T"]
        # TODO: preprocess X (src.data.features) if needed
        # X_train = f.represent(X_train, type=represent, norm=norm_input)
        # X_test = f.represent(X_test, type=represent, norm=norm_input)
        X_train = f.Input(X_train).forward(X_train)
        X_test = f.Input(X_train).forward(X_test)
        # X_train = X_train.reshape(X_train.shape[0], -1)
        # X_test = X_test.reshape(X_test.shape[0], -1)
        # TODO: load model
        model = get_model(model_name, verbose=verbose)
        # TODO: train model by 5 folds
        model.fit(X_train, y_train)
        # TODO: evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Fold {fold}: MAE = {mae:.4f} kcal/mol")


def run(args):
    for model in args.models:
        run_model(model,
                  save_dir=args.data_dir,
                  represent=args.represent,
                  norm_input=not args.not_norm,
                  verbose=args.verbose)


if __name__ == "__main__":
    import os
    import pyrootutils

    pyrootutils.setup_root(
        os.path.dirname(os.path.dirname(__file__)),
        indicator=".project_root",
        pythonpath=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "svr", "kr", "gp", "mlp"],
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
        default="./data",
        help="directory to save data",
    )
    args = parser.parse_args()
    run(args)
