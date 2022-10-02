#!/bin/bash
###
 # @Author: ----
 # @Date: 2022-04-06 15:47:35
 # @LastEditors: GhMa
 # @LastEditTime: 2022-10-02 19:34:49
###
# ----server

MY_PYTHON="python"  # default python exe
# See main_*.py for argument details!
# use *debug.py files for developing, they are slightly slower than those without debug


$MY_PYTHON main_cifar10.py --neuron NILIF --reg no --model sresnet18 --epochs 300 --minibatch 256 --timestep 2 --seed 1 --lr 0.003 --sigma 0.3
$MY_PYTHON main_dvs-cifar10.py --neuron NILIF --reg no --model vgg --epochs 210 --minibatch 32 --timestep 10 --seed 1 --lr 0.0003 --sigma 0.2 
