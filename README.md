<!--
 * @Author: ----
 * @Date: 2022-04-09 11:57:47
 * @LastEditors: GhMa
 * @LastEditTime: 2022-10-02 19:35:49
-->
# Part of the Project *ASGARD*: A Spiking-neural-model GARDen
![Porject ASGARD](https://github.com/genema/Noisy-Spiking-Neuron-Nets/raw/master/proj_logo.jpg)
# Noisy Spiking Neural Networks
## Repo arch
* models: neuron models (LIF, Noisy LIF with gaussian, logistic, triangular, uniform noises), networks (residual, vgg, cifarnet)

## Usage

* modify dataset path in `dataset_utils.py`.
* to train a model, run example commands in `run_training.sh` script. 

### Tips
* **Since the inference (by forward) and learning (by backward) implementations are wrapped in the neuron modules, you may use noisy networks in your code by importing our  noisy neurons in `models` folder.**

* conf file of our software environment: `requirements.txt`.
* test scipts: `testing_dataset.py`.
* In addition to the Gaussian noise implementation in the text (`NILIF.py`), we provide implementations of discrete models (`NILIF_*.py`) with other random processes (with static increments) corresponding to rectangular, arctangent, and sigmoidal surrogate gradients.

## Ackowledgement
* Special thanks to Prof. Penghang Yin (SUNY Albany) and Dr. Seiya Tokui (Preferred Networks, Inc.) for helpful discussions.

## Citation info


