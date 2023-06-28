import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.nn_conv import NNConv


class MPNNGNN(nn.Module):
    """
    MPNN model is introduced in `Neural Message Passing for Quantum Chemistry`
    """

    def __init__(
        self,
        node_input_dim: int,
        node_output_dim: int,
        edge_input_dim: int,
        edge_hidden_dim: int,
        dropout_rate: float = 0.2,
        num_step_message_passing: int = 3,
    ):
        super(MPNNGNN, self).__init__()

        self.project_node_features = nn.Sequential(
            nn.Linear(node_input_dim, node_output_dim),
            nn.BatchNorm1d(node_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.BatchNorm1d(edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(edge_hidden_dim, node_output_dim * node_output_dim),
        )
        self.gnn = NNConv(
            in_channels=node_output_dim,
            out_channels=node_output_dim,
            nn=edge_network,
            aggr="add"
        )
        self.gru = nn.GRU(node_output_dim, node_output_dim)

    def reset_parameters(self):
        self.project_node_features[0].reset_parameters()
        self.gnn.reset_parameters()
        for layer in self.gnn.nn:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        node_feats = self.project_node_features(x)
        hidden_feats = node_feats.unsqueeze(0)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn(node_feats, edge_index, edge_attr))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats
