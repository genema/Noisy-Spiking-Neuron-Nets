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


- **News** 🥳🥳 👏👏🏼👏🏾: The model and method in this repo have been implemented and included in the [***snnTorch***](https://github.com/jeshraghian/snntorch) library!
  Currently, they reside in the [*noisy_leaky*](https://github.com/jeshraghian/snntorch/tree/noisy_leaky) branch and are awaiting subsequent merging into the master branch.



## Summary
Networks of spiking neurons underpin the extraordinary information-processing capabilities of the brain and have become pillar models in neuromorphic artificial intelligence. Despite extensive research on spiking neural networks (SNNs), most studies are established on deterministic models, overlooking the inherent non-deterministic, noisy nature of neural computations. This study introduces the noisy spiking neural network (NSNN) and the noise-driven learning rule (NDL) by incorporating noisy neuronal dynamics to exploit the computational advantages of noisy neural processing. NSNN provides a theoretical framework that yields scalable, flexible, and reliable computation and learning. We demonstrate that this framework leads to spiking neural models with competitive performance, improved robustness against challenging perturbations than deterministic SNNs, and better reproducing probabilistic neural computation in neural coding. Generally, this study offers a powerful and easy-to-use tool for machine learning, neuromorphic intelligence practitioners, and computational neuroscience researchers.

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
A [*preprint version*](https://arxiv.org/abs/2305.16044):
```bibtex
@article{ma2023exploiting,
  title={Exploiting Noise as a Resource for Computation and Learning in Spiking Neural Networks},
  author={Ma, Gehua and Yan, Rui and Tang, Huajin},
  journal={arXiv preprint arXiv:2305.16044},
  year={2023}
}
```
