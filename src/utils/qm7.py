import os
import numpy as np
import scipy.io
from urllib.request import urlopen
from typing import Dict, Tuple


def download_data(save_dir: str = "."):
    url = "http://www.quantum-machine.org/data/qm7.mat"
    print(f"Downloading QM7 dataset from {url}")
    response = urlopen(url)
    save_path = os.path.join(save_dir, "qm7.mat")
    with open(save_path, "wb") as f:
        f.write(response.read())
    print("Data downloaded!")


def get_fold_ids(P: np.ndarray, fold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Get the training and validation folds from the indices P."""
    train_ids = np.concatenate([P[:fold], P[fold + 1:]]).flatten()
    test_ids = P[fold]

    return train_ids, test_ids


def get_fold_data(data: Dict, fold: int = 0):
    train_ids, test_ids = get_fold_ids(data["P"], fold)
    if data["T"].shape[0] == 1:
        data["T"] = data["T"].squeeze()
    data_train = {
        "X": data["X"][train_ids],
        "T": data["T"][train_ids],
        "R": data["R"][train_ids],
        "Z": data["Z"][train_ids],
    }
    data_test = {
        "X": data["X"][test_ids],
        "T": data["T"][test_ids],
        "R": data["R"][test_ids],
        "Z": data["Z"][test_ids],
    }

    return data_train, data_test


def load_qm7_data(save_dir: str) -> Dict:
    """ Load the QM7 dataset."""
    filepath = os.path.join(save_dir, "qm7.mat")
    if not os.path.exists(filepath):
        download_data(save_dir)

    data = scipy.io.loadmat(filepath)
    return data


def load_qm7(save_dir: str, fold: int = 0):
    """Load the QM7 dataset."""
    data = load_qm7_data(save_dir)
    return get_fold_data(data, fold=fold)
