<!--
 * @Author: ----
 * @Date: 2022-04-09 11:57:47
 * @LastEditors: GhMa
 * @LastEditTime: 2023-05-02 19:35:49
-->
<img src="https://github.com/genema/Noisy-Spiking-Neuron-Nets/raw/master/proj_logo.jpg" width="200px" align="left">

# Noisy Spiking Neural Networks 

![Python](https://img.shields.io/badge/Python-3.8.16-brightgreen)

![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-brightgreen)

![PROJASGARD](https://img.shields.io/badge/Project-ASGARD-orange)



 [`Here`](https://arxiv.org/abs/2305.16044)  you can find/get: 
 - A theoretical frame for SNNs, induced by re-introducing noisy neuronal dynamics;
 - A biophysical/mathematical rationale for surrogate gradients (pseudo derivative, derivative approximation);
 - Implicit regularization in SNNs by incorporating internal noise; 
 - Improved robustness of SNNs by incorporating internal noise; 
 - Computational account for the "variability-reliability" in biological neural computation.


## Repo arch
- models
  - neuron models (LIF, Noisy LIF with Gaussian, logistic, triangular, uniform noises)
  - networks (residual, vgg, cifarnet ...)
- crits
  - objective functions
- tools
  - helpers
- conf file of our software environment: `requirements.txt`.
- test scripts: `testing_(datasetname).py`.
- train scripts: `main_(datasetname).py`

## Usage

1. modify the dataset path in `dataset_utils.py`.
2. run example commands in `run_training.sh` script to train models. 

### Tips
* **It is rather easy to use NSNN by modifying your own code.** Since the inference (by forward) and learning (by backward) implementations are wrapped in the neuron modules, you may refer to (or directly use) our noisy neuron implementations in `models` folder to build your noisy networks.
* In addition to the Gaussian noise implementation in the text (`NILIF.py`), we provide implementations of discrete models (`NILIF_*.py`) with other random processes (with static increments) corresponding to rectangular, arctangent, and sigmoidal surrogate gradients.

## Acknowledgement
Special thanks to *Prof. Penghang Yin* (SUNY Albany) and *Dr. Seiya Tokui* (Preferred Networks, Inc.) for helpful discussions and suggestions.

## Citation info
A preprint version:
```bibtex
@article{ma2023exploiting,
  title={Exploiting Noise as a Resource for Computation and Learning in Spiking Neural Networks},
  author={Ma, Gehua and Yan, Rui and Tang, Huajin},
  journal={arXiv preprint arXiv:2305.16044},
  year={2023}
}
```
