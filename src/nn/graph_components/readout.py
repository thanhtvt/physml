import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class MLPReadout(nn.Module):

    def __init__(
        self,
        node_input_dim: int,
        node_hidden_dim: int,
        node_output_dim: int,
        dropout_rate: float = 0.2,
        mode: str = "mean"
    ):
        super(MLPReadout, self).__init__()

        assert mode in ["sum", "mean", "max"], "mode must be sum, mean or max"
        self.project_node_features = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            nn.BatchNorm1d(node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(node_hidden_dim, node_output_dim)
        )
        if mode == "sum":
            self.pool = global_add_pool
        elif mode == "mean":
            self.pool = global_mean_pool
        else:
            self.pool = global_max_pool

    def forward(self, node_features, batch):
        node_feats = self.project_node_features(node_features)
        return self.pool(node_feats, batch)
