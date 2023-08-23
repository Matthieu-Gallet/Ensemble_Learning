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

### Acknowledgements

The authors would like to thank the *Spanish Instituto Nacional de Tecnica Aerospacial* (INTA) for the PAZ images (Project AO-001-051) .



Feel free to ask if any question.

If you use this work in your research and find it useful, please cite using the following bibtex reference:

```
@inproceedings{gallet:hal-04184390,
  TITLE = {{Apprentissage explicable d'un ensemble de divergences pour la similarit{\'e} inter-classe de donn{\'e}es SAR}},
  AUTHOR = {Gallet, Matthieu and Atto, Abdourrahmane and Trouv{\'e}, Emmanuel and Karbou, Fatima},
  URL = {https://hal.science/hal-04184390},
  BOOKTITLE = {{GRETSI, XXIX{\`e}me Colloque Francophone de Traitement du Signal et des Images}},
  ADDRESS = {Grenoble, France},
  ORGANIZATION = {{GRETSI}},
  YEAR = {2023},
  MONTH = Aug,
  PDF = {https://hal.science/hal-04184390/file/GRETSI_DIV23_version2.pdf},
  HAL_ID = {hal-04184390},
  HAL_VERSION = {v1},
}
```

