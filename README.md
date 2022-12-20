
# Overview
This repository complements the paper [Multiple Imputation with Neural Network Gaussian Process for High-dimensional Incomplete Data](https://arxiv.org/abs/2211.13297) (Zongyu Dai, Zhiqi Bu, Qi Long):

* `minngp.py` contains the main imputation functions
* `MI-NNGP experiments.ipynb` contains an example to use it

Note: The MI-NNGP functions are a convenient tool for imputing both low and high-dimensional datasets. It was originally developed for imputing multi-omics data in biological and medical research, where the number of patterns in the data is not overly large. However, if the number of patterns is very large, using MI-NNGP may be time-consuming.

# reference
Zongyu Dai, Zhiqi Bu, Qi Long: [Multiple Imputation with Neural Network Gaussian Process for High-dimensional Incomplete Data](https://arxiv.org/abs/2211.13297)
```
@article{dai2022multiple,
  title={Multiple Imputation with Neural Network Gaussian Process for High-dimensional Incomplete Data},
  author={Dai, Zongyu and Bu, Zhiqi and Long, Qi},
  journal={arXiv preprint arXiv:2211.13297},
  year={2022}
}
```

# Dependencies
* Python 3+
* neural_tangents
* jax
