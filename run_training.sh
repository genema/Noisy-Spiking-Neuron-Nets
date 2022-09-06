#!/bin/bash
###
 # @Author: ----
 # @Date: 2022-04-06 15:47:35
 # @LastEditors: ----
 # @LastEditTime: 2022-09-06 15:19:01
###
# ----server

MY_PYTHON="python"  # default python exe
# See main_*.py for argument details!
# use *debug.py files for developing, they are slightly slower than those without debug


#220820
$MY_PYTHON main_cifar10.py --neuron NILIF --reg no --model sresnet19 --epochs 300 --minibatch 256 --timestep 2 --seed 2 --lr 0.003 --sigma 0.3
