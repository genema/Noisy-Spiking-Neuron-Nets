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


- A theoretical framework that subsumes conventional deterministic spiking neural networks and surrogate gradients

- Scalable spiking neural models that incorporate noisy neuronal dynamics for implicit regularization, improved robustness, and computational accounts of biological neural computation

## Summary
Networks of spiking neurons underpin the extraordinary information-processing capabilities of the brain and have emerged as pillar models in neuromorphic intelligence. Despite extensive research on spiking neural networks (SNNs), most are established on deterministic models. Integrating noise into SNNs leads to biophysically more realistic neural dynamics and may benefit model performance. This work presents the noisy spiking neural network (NSNN) and the noise-driven learning rule (NDL) by introducing a spiking neuron model incorporating noisy neuronal dynamics. Our approach shows how noise may serve as a resource for computation and learning and theoretically provides a framework for general SNNs. We show that our method exhibits competitive performance and improved robustness against challenging perturbations than deterministic SNNs and better reproduces probabilistic neural computation in neural coding. This study offers a powerful and easy-to-use tool for machine learning, neuromorphic intelligence practitioners, and computational neuroscience researchers.

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
A [`preprint version`](https://arxiv.org/abs/2305.16044):
```bibtex
@article{ma2023exploiting,
  title={Exploiting Noise as a Resource for Computation and Learning in Spiking Neural Networks},
  author={Ma, Gehua and Yan, Rui and Tang, Huajin},
  journal={arXiv preprint arXiv:2305.16044},
  year={2023}
}
```
