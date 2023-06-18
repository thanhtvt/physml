import torch
import torch.nn as nn
from typing import List


class MultiLayerPerceptron2(nn.Module):
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
        self.lin1 = nn.Linear(input_size, hidden_sizes[0])
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], output_size)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        # for i, h in enumerate(hidden_sizes):
        #     if i == 0:
        #         self.layers.add_module(f"linear_{i}", nn.Linear(input_size, h))
        #     else:
        #         self.layers.add_module(f"linear_{i}", nn.Linear(hidden_sizes[i - 1], h))
        #     self.layers.add_module(f"sigmoid_{i}", nn.Sigmoid())
        # self.layers.add_module(f"linear_{i + 1}", nn.Linear(hidden_sizes[-1], output_size))
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.lin1.weight, mean=0, std=1 / self.lin1.weight.shape[1]**0.5)
        nn.init.normal_(self.lin2.weight, mean=0, std=1 / self.lin2.weight.shape[1]**0.5)
        nn.init.normal_(self.lin3.weight, mean=0, std=1 / self.lin3.weight.shape[1]**0.5)

    def forward(self, x):
        # print("00:", x.shape, "\n", x)
        out = self.lin1(x)
        # print("01:", out.shape, "\n", x)
        out = self.sig1(out)
        # print("02:", out.shape, "\n", x)
        out = self.lin2(out)
        # print("03:", out.shape, "\n", x)
        out = self.sig2(out)
        # print("04:", out.shape, "\n", x)
        out = self.lin3(out)
        # print("05:", out.shape, "\n", x)
        return out.squeeze(1)
