<!--
 * @Author: ----
 * @Date: 2022-04-09 11:57:47
 * @LastEditors: GhMa
 * @LastEditTime: 2022-09-08 15:31:00
-->
# Repo of Noisy Spiking Neural Networks.
## Usage

* modify dataset path in `dataset_utils.py`.
* to train a model, run commands in `run_training.sh` script. 

### Tips

* Conf file of our software environment: `requirements.txt`.
* test scipts: `testing_dataset.py`.
* In addition to the Gaussian noise implementation in the text (`NILIF.py`), we provide implementations of discrete models (`NILIF_*.py`) with other random processes (with static increments) corresponding to rectangular, arctangent, and sigmoidal surrogate gradients.

### Disclaimer
* Part of our implementations references those in **"STCA: Spatio-temporal credit assignment with delayed feedback in deep spiking neural networks." IJCAI. 2019**.


