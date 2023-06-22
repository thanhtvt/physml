<div align="center">

<h2> Deep Learning for Atomization Energies Prediction </h2>

_Suggestions are always welcome!_

</div>

## Introduction
This folder contains the recipe for utilizing MLP and graph neural networks to predict atomization energies of small molecules in [`QM7`](http://quantum-machine.org/datasets/) dataset.

The GNN is based on the `MPNN` model presented in the paper [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212).

## Prequisites
We use [`torch==1.12.1`](https://pytorch.org/get-started/previous-versions/#linux-and-windows-6) and [`torch_geometric==2.3.0`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to implement the model. It is recommended to have `conda` environment. Go to the root folder and run:
```bash
pip install -e .
```

## Usage
Currently, we provide two models: `MLP` with some Linear layers and `GNN` with `MPNN` as the message passing part and MLP as the readout part. To train the MLP:
```bash
python mlp_runner.py [-h] [--devices DEVICES]
                     [--fold {0,1,2,3,4}]
                     [--feature-type {coulomb,sorted,eigen}]
                     [--new-wandb]

options:
  -h, --help            show this help message and exit
  --devices DEVICES     Separated by comma, e.g. 0,1,2,3
  --fold {0,1,2,3,4}    Fold to use for training
  --feature-type {coulomb,sorted,eigen}
                        Type of feature to use
  --new-wandb           Create a new wandb run
```

or to train the GNN:
```bash
python graph_runner.py [-h] [--devices DEVICES]
                       [--fold {0,1,2,3,4}]
                       [--resume-training]
                       [--new-wandb]

options:
  -h, --help          show this help message and exit
  --devices DEVICES   Separated by comma, e.g. 0,1,2,3
  --fold {0,1,2,3,4}  Fold to use for training
  --new-wandb         Create a new wandb run
  --resume-training   Continue training from a checkpoint
```

## Results
Please check the report for more details.

## References
1. [Learning Invariant Representations of Molecules for Atomization Energy Prediction](https://proceedings.neurips.cc/paper_files/paper/2012/file/115f89503138416a242f40fb7d7f338e-Paper.pdf)
2. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)
3. [Gated Graph Sequence Neural Networks](https://arxiv.org/pdf/1511.05493.pdf)