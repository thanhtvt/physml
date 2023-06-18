import torch
from torch.utils.data import Dataset
from typing import Dict


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
        noise: float = 1.0,
        step: int = 1,
        max_value: torch.Tensor = None,
    ):
        super().__init__(data=data)
        self.noise = noise
        self.step = step

        num_atoms = self.coulomb.shape[-1]
        self.prune_idx = torch.triu(torch.ones(num_atoms, num_atoms)).bool().flatten()
        X_realized = self.realize(self.coulomb)
        if max_value is None:
            max_value = torch.max(X_realized, dim=0)[0]
            self.max = torch.where(max_value > 0, max_value, 0)
        else:
            self.max = max_value
        X = self.expand(X_realized)
        self.x_mean = torch.mean(X, axis=0)
        self.x_std = torch.std(X, axis=0)

    def get_max(self):
        return self.max

    def preprocess(self, X):
        # print("data-process 01:", X.shape, "\n", X)
        out = self.realize(X)
        # print("data-process 02:", out.shape, "\n", out)
        out = self.expand(out)
        # print("data-process 03:", out.shape, "\n", out)
        out = self.normalize(out)
        # print("data-process 04:", out.shape, "\n", out)
        return out

    def realize(self, X):

        def _realize(x):
            e = torch.normal(0, self.noise, x[0].shape)
            inds = torch.argsort(-(x**2).sum(axis=0)**.5 + e).flatten()
            x = x[inds, :][:, inds].flatten()
            return x

        if X.ndim == 2:
            return self.prune(_realize(X).unsqueeze(0))

        return self.prune(torch.stack([_realize(x) for x in X]))

    def expand(self, X):
        # TODO: check expand in sample code
        X_expand = []
        for i in range(X.shape[1]):
            for k in range(0, int(self.max[i]) + self.step, self.step):
                X_expand += [torch.tanh((X[:, i] - k) / self.step)]
        return torch.stack(X_expand).T

    def normalize(self, X):
        return (X - self.x_mean) / (self.x_std + 1e-6)

    def prune(self, X):
        return X[:, self.prune_idx]
