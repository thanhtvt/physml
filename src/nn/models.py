import torch.nn as nn
from typing import List

from src.nn.graph_components.readout import MLPReadout
from src.nn.graph_components.gnn import MPNNGNN


class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: List[int] = [400, 100],
                 dropout_rate: float = 0.5,
                 activation_type: str = "sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        # create linear layers
        self.layers = nn.Sequential()
        for i, h in enumerate(hidden_sizes):
            if i == 0:
                self.layers.add_module(f"linear_{i}", nn.Linear(input_size, h))
            else:
                self.layers.add_module(f"linear_{i}", nn.Linear(hidden_sizes[i - 1], h))
            self.set_activation(activation_type, index=i)
            self.layers.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
        self.layers.add_module(f"linear_{i + 1}", nn.Linear(hidden_sizes[-1], output_size))
        self.init_weight()

    def set_activation(self, activation_type: str, index: int):
        if activation_type == "sigmoid":
            self.layers.add_module(f"sigmoid_{index}", nn.Sigmoid())
        elif activation_type == "tanh":
            self.layers.add_module(f"tanh_{index}", nn.Tanh())
        elif activation_type == "relu":
            self.layers.add_module(f"relu_{index}", nn.ReLU())
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def init_weight(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1 / layer.weight.shape[1]**0.5)

    def forward(self, x, target_std, target_mean):
        out = self.layers(x)
        # TODO: process output as sample code
        out = out * target_std + target_mean
        return out.squeeze(1)


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)


class MPNN(nn.Module):
    """
    MPNN model is introduced in `Neural Message Passing for Quantum Chemistry`
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        gnn_node_output_dim: int = 8,
        gnn_edge_hidden_dim: int = 16,
        gnn_dropout_rate: float = 0.2,
        num_step_message_passing: int = 3,
        readout_node_hidden_dim: int = 8,
        readout_node_output_dim: int = 8,
        readout_dropout_rate: float = 0.2,
        pooling_mode: str = "mean",
        output_dim: int = 1,
    ):
        super(MPNN, self).__init__()

        self.gnn = MPNNGNN(
            node_input_dim,
            gnn_node_output_dim,
            edge_input_dim,
            gnn_edge_hidden_dim,
            gnn_dropout_rate,
            num_step_message_passing,
        )
        self.readout = MLPReadout(
            gnn_node_output_dim,
            readout_node_hidden_dim,
            readout_node_output_dim,
            readout_dropout_rate,
            pooling_mode
        )
        self.predict = nn.Linear(readout_node_output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        node_feats = self.gnn(x, edge_index, edge_attr)
        # print("node_feats.shape:", node_feats.shape)
        # print("batch.shape:", batch.shape)

        graph_feats = self.readout(node_feats, batch)
        return self.predict(graph_feats)
