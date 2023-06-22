<div align="center">

<h2> Machine Learning for Atomization Energies Prediction </h2>

_Suggestions are always welcome!_

</div>

## Introduction
This folder contains the recipe for utilizing different machine learning approaches to predict atomization energies of small molecules in [`QM7`](http://quantum-machine.org/datasets/) dataset. List of models that are currently available:
- [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [`K-nearest Neighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [`Kernel Ridge Regression`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
- [`Support Vector Machines`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

## Prequisites
Make sure you have Python 3.8+ with `numpy` and `sklearn` installed in your machine.

## Usage
With a simple command, users can train any of the above models:
```bash
python runml.py [-h] [--models MODELS [MODELS ...]]
                [--represent {coulomb,sorted,eigen,randomize}]
                [--not-norm] [--verbose] [--data-dir DATA_DIR]
                [--grid-search] [--shuffle] [--gauss-kernel]

options:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        list of models to run
  --represent {coulomb,sorted,eigen,randomize}
                        representation of input data
  --not-norm            not to normalize input data
  --verbose             print out training progress
  --data-dir DATA_DIR   directory to save data
  --grid-search         whether to perform grid search
  --shuffle             whether to shuffle data
  --gauss-kernel        use gaussian kernel
```

For example, to train a `Kernel Ridge Regression` model with the feature type of `eigen` and DON'T normalize data, run:
```bash
python runml.py --models kr --represent eigen --not-norm
```

## Results
Please check the report for more details.

## References
1. [Learning Invariant Representations of Molecules for Atomization Energy Prediction](https://proceedings.neurips.cc/paper_files/paper/2012/file/115f89503138416a242f40fb7d7f338e-Paper.pdf)
2. [Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies](https://pubs.acs.org/doi/10.1021/ct400195d)
