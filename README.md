# Ensemble Learning

## 1. Description
The project explore the ensemble learning method based on a set of divergence. These divergence are calculated on each pair of a dataset of SAR images.
The final dataset is composed of vectors of divergence associated to a label. The label is either "Forest", "Pasture" or "Different" (see `data/divergence.txt`).

6 different neural networks architectures are tested. A full report is created to compare the performance of each architecture. (`ensemble_learning/explore_learning.py`)
A selection of the 3 best architectures is used to observe the weight learned by the neural network (see `fig/weight.pdf` and `ensemble_learning/evaluate_model.py`).


## 2. Structure of the project:
``` bash
Ensemble_Learning
├── README.md
├── data
│   ├── divergence_process.h5
│   └── divergence.txt
├── ensemble_learning
│   ├── __init__.py
│   ├── utils.py
│   ├── figure.py
│   ├── architecture.py
│   ├── data_preparation.py
│   ├── evaluate_model.py
│   └── explore_learning.py
└── fig
    └── weight.pdf
```
