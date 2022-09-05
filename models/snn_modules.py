'''
Author: ----
Date: 2022-04-06 10:57:29
LastEditors: ----
LastEditTime: 2022-08-02 14:11:31
'''
import torch
import torch.nn as nn
import numpy as np
import math


class Seq2ANN(nn.Module):
    r"""
    Forked from spikingjelly
    """
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = Seq2ANN(layer)
        self.bn = bn
    
    def forward(self, x):
        x = self.layer(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class tdBN(nn.Module):
    def __init__(self, c_out):
        super(tdBN, self).__init__()
        self.tdbn = Seq2ANN(nn.BatchNorm2d(c_out))

    def forward(self, x):
        return self.tdbn(x)
