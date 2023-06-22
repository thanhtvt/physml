<div align="center">

<h2> Atomization Energies Prediction with QM7 Dataset </h2>

_Suggestions are always welcome!_

</div>

## Introduction
This repository contains the recipe for predicting atomization energies of small molecules in [`QM7`](http://quantum-machine.org/datasets/) dataset, using various machine learning algorithms and deep learning models, which are MLPs and GNNs.

Details about those algorithms and models can be found in the report.

## Prequisites
To run the code, you first need to setup the repo by running at the root folder:
```bash
pip install -e .
```

## Usage
Details about how to train and evaluate the models can be found in 2 folders:
- [`ml`](./src/ml/): for machine learning algorithms.
- [`nn`](./src/nn/): for deep learning models.

## Results
Please check the report for more details.

## References
1. [Learning Invariant Representations of Molecules for Atomization Energy Prediction](https://proceedings.neurips.cc/paper_files/paper/2012/file/115f89503138416a242f40fb7d7f338e-Paper.pdf)
2. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)
3. [Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies](https://pubs.acs.org/doi/10.1021/ct400195d)
4. [Gated Graph Sequence Neural Networks](https://arxiv.org/pdf/1511.05493.pdf)
