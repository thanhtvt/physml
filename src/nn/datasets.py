import scipy.io
import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from typing import Dict, Tuple
from src.utils.qm7 import get_fold_data
from src.data.features import sort_coulomb, get_eigenspectrum, encode_atom_charge


class NNBaseDataset(Dataset):
    def __init__(self, data: Dict):
        super().__init__()
        self.coulomb = torch.from_numpy(data["X"])
        self.atom_energy = torch.from_numpy(data["T"].squeeze())

    def __len__(self):
        return len(self.coulomb)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.coulomb[idx]
        y = self.atom_energy[idx]
        return self.preprocess(X), y

    def preprocess(self, X: torch.Tensor):
        raise NotImplementedError


class MLPDataset(NNBaseDataset):
    def __init__(
        self,
        data: Dict,
        data_mean: Tuple[float, float],
        data_std: Tuple[float, float],
        step: int = 1,
        feature_type: str = "sorted",
    ):
        super().__init__(data=data)
        self.step = step
        self.X_mean, self.energy_mean = data_mean
        self.X_std, self.energy_std = data_std

        num_atoms = self.coulomb.shape[-1]
        prune_idx = torch.triu(torch.ones(num_atoms, num_atoms)).bool().flatten()
        if feature_type == "sorted":
            self.features = torch.from_numpy(sort_coulomb(self.coulomb.numpy()))
            self.features = self.features.reshape(-1, num_atoms**2)[:, prune_idx]
        elif feature_type == "eigen":
            self.features = torch.from_numpy(get_eigenspectrum(self.coulomb.numpy()))
        else:
            self.features = self.coulomb.reshape(-1, num_atoms**2)[:, prune_idx]
        self.features = self.expand(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.features[idx]
        y = self.atom_energy[idx]
        X_norm = self.normalize(X, self.X_mean, self.X_std)  # type: ignore
        # y_norm = self.normalize(y, self.energy_mean, self.energy_std)
        return X_norm, y

    def get_target_stats(self):
        return self.energy_mean, self.energy_std

    def normalize(self, mat: torch.Tensor, mean, std):
        return (mat - mean) / std

    def expand(self, X):
        X_expand = []
        for i in range(X.shape[1]):
            for k in [-3, -2, -1, 0, 1, 2, 3]:
                X_expand += [torch.tanh((X[:, i] + k * self.step) / self.step)]
        return torch.stack(X_expand).T


class GraphDataset(InMemoryDataset):

    url = "http://www.quantum-machine.org/data/qm7.mat"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        data_dir=None,
        fold=0,
        train=True
    ):
        self.data_path = os.path.join(data_dir, "qm7.mat")
        self.train = train
        self.fold = fold
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["qm7.mat"]

    @property
    def processed_file_names(self):
        if self.train:
            return [f"train_fold_{self.fold}.pt"]
        else:
            return [f"test_fold_{self.fold}.pt"]

    def process(self):
        dataset = scipy.io.loadmat(self.data_path)
        database = get_fold_data(dataset, self.fold)
        dataset = database[0] if self.train else database[1]
        # coulomb = torch.from_numpy(sort_coulomb(dataset["X"]))
        coulomb = torch.from_numpy(dataset["X"])
        atom_energy = torch.from_numpy(dataset["T"].squeeze())
        atom_type = torch.from_numpy(encode_atom_charge(dataset["Z"]))
        # coordinates = torch.from_numpy(dataset["R"])

        data_list = []
        n_samples = len(coulomb)
        for i in range(n_samples):
            # Sort coulomb matrix
            mat = coulomb[i]
            norm = torch.linalg.norm(mat, dim=0)
            order = torch.argsort(norm, descending=True)
            mat = mat[order, :][:, order]

            # Get edge index and edge attribute
            edge_index = torch.nonzero(mat, as_tuple=False).t().contiguous()
            edge_attr = mat[edge_index[0], edge_index[1]].view(-1, 1)
            y = atom_energy[i].view(1, -1)

            # Process feature
            # 1. eigenvalues
            num_atoms = torch.sqrt(torch.tensor(edge_attr.shape[0])).int().item()
            coulomb_mat = edge_attr.view(num_atoms, num_atoms)
            eigs = torch.from_numpy(get_eigenspectrum(coulomb_mat.numpy()))
            eigs = eigs.view(-1, 1)

            # 2. Atom type (based on atom charge)
            # atype = atom_type[i][order]
            # atype = atype[atype != 0]
            # atype = atype.view(-1, 1)

            # 3. Concat
            # x = torch.cat([eigs, atype], dim=1)
            x = eigs

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1  # type: ignore
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
