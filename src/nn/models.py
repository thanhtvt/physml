import torch.nn as nn
from typing import List


class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: List[int] = [400, 100],
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
            self.layers.add_module(f"sigmoid_{i}", nn.Sigmoid())
        self.layers.add_module(f"linear_{i + 1}", nn.Linear(hidden_sizes[-1], output_size))
        self.init_weight()

    def init_weight(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1 / layer.weight.shape[1]**0.5)

    def forward(self, x):
        x = self.layers(x)
        # TODO: process output as sample code
        return x.squeeze(1)


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
