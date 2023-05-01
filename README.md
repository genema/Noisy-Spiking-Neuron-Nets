<!--
 * @Author: ----
 * @Date: 2022-04-09 11:57:47
 * @LastEditors: GhMa
 * @LastEditTime: 2023-05-02 19:35:49
-->
# Repo. of Noisy Spiking Neural Network (NSNN)

Part of the *ASGARD* PROJECT.

![Porject ASGARD](https://github.com/genema/Noisy-Spiking-Neuron-Nets/raw/master/proj_logo.jpg)

## Repo arch
-- models
---- neuron models (LIF, Noisy LIF with gaussian, logistic, triangular, uniform noises), networks (residual, vgg, cifarnet)

## Usage

1. modify dataset path in `dataset_utils.py`.
2. run example commands in `run_training.sh` script to train models. 

### Tips
* **Since the inference (by forward) and learning (by backward) implementations are wrapped in the neuron modules, you may use noisy networks in your code by importing our  noisy neurons in `models` folder.**

* conf file of our software environment: `requirements.txt`.
* test scipts: `testing_dataset.py`.
* In addition to the Gaussian noise implementation in the text (`NILIF.py`), we provide implementations of discrete models (`NILIF_*.py`) with other random processes (with static increments) corresponding to rectangular, arctangent, and sigmoidal surrogate gradients.

## Ackowledgement

Special thanks to Prof. Penghang Yin (SUNY Albany) and Dr. Seiya Tokui (Preferred Networks, Inc.) for helpful discussions.

## Citation info


