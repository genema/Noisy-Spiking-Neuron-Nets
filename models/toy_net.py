'''
Author: ----
Date: 2022-04-08 11:09:07
LastEditors: ----
LastEditTime: 2022-08-30 14:39:48
'''
import torch
import torch.nn as nn
from .snn_modules import tdLayer, tdBN


class ToyNet(nn.Module):
    r"""
    """
    def __init__(
        self, 
        num_classes = 10, 
        hidden = 64,
        norm_layer = None,
        spiking_neuron: callable = None,
        n_input = [1, 28 * 28],
        **kwargs
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = tdBN

        self.layer = tdLayer(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden, bias=False),
        )
        self.fc = tdLayer(
            nn.Linear(hidden, num_classes, bias=False)
        )
        self.sn = spiking_neuron(**kwargs)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.sn(x)
        x = self.fc(x)
        return x


def toy_net(
    spiking_neuron: callable = None, 
    hidden = 64, 
    n_input = [1, 28 ** 2],
    n_output = 10,
    **kwargs
):
    return ToyNet( 
        num_classes = n_output,
        hidden = hidden,
        spiking_neuron = spiking_neuron,
        n_input = n_input,
        **kwargs
    )


