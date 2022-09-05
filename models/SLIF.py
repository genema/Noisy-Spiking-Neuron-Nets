'''
Author: ----
Date: 2022-04-29 09:44:29
LastEditors: ----
LastEditTime: 2022-04-29 09:45:40
'''
import torch
import torch.nn as nn
import numpy as np
import math
        

class Stochastic(torch.autograd.Function):
    r"""
    default type is gaussian noise. 
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) 
        ctx.o = (input > 0).float()
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temp = 1 / (2 * math.sqrt(2*math.pi)) * torch.exp(-1/2 * input.pow_(2))
        grad_input = grad_output.clone()
        return grad_input * temp
        
        
slif_act_fun = Stochastic.apply


class Neuron(nn.Module):
    r"""
    Iterative Stochastic LIF neuron model.
    """
    def __init__(
        self,
        threshold=1.0,
        decay_input=False,
        tau=2.0,
        alpha=1.,
        device='cuda'
    ) -> None:
        super().__init__()
        self.tau = 1. / tau
        self.decay_input = decay_input
        self.threshold = threshold
        self.device = device
        self.alpha = alpha
        self.u = None
        self.o = None
             
    def _update(self, x):
        timestep = x.size(1)
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            # additive noise
            if self.training:
                self.u += -torch.normal(torch.zeros_like(self.u), 1)  
            self.o[:, t, ...] = slif_act_fun(self.u - self.threshold)
            self.u = self.u * (1 - self.o[:, t, ...])
    
    def _silent_update(self, x):
        timestep = x.size(1)
        output = []
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            output.append(self.u)
        return torch.stack(output, dim=1)
    
    def _reset(self):
        self.u = None
        self.o = None
    
    def _states(self):
        return self.u, self.o
    
    def forward(self, x, out_u=False):
        self.u = 0
        self.o = torch.zeros_like(x) 
        if out_u:
            return self._silent_update(x)
        else:
            self._update(x)
            return self.o
