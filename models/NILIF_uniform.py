'''
Author: ----
Date: 2022-06-30 11:56:44
LastEditors: ----
LastEditTime: 2022-09-06 15:22:15
'''
import torch
import torch.nn as nn
import torch.distributions as distrib
import numpy as np
import math
        

def heaviside(x: torch.Tensor):
    return (x >= 0.0).to(x)


class StochasticST(torch.autograd.Function):
    r"""
    Stochastic straight-through gradient estimator
    The noise type here is uniform noise. 
    """
    @staticmethod
    def forward(ctx, input, a):
        ctx.save_for_backward(input) 
        ctx.a = a
        
        ctx.p_spike = ((input - -ctx.a) / (a - -ctx.a)).clamp(0, 1)
        return torch.bernoulli(ctx.p_spike)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temp = ((input >= -ctx.a).int() & (input <= ctx.a).int()) * (
            1 / (ctx.a - -ctx.a)
        )
        grad_input = grad_output.clone()
        return grad_input * temp, None, 
        
        
nilif_act_fun = StochasticST.apply


class Neuron(nn.Module):
    r"""
    Noise injected LIF neuron, uniform noise Uniform(-a, a)
    """
    def __init__(
        self,
        threshold=1.0,
        decay_input=False,
        tau=2.0,
        alpha=1.,
        mu=0,
        sigma=0.4,
        n_ensembles=4,
        device='cuda'
    ) -> None:
        super().__init__()
        self.tau = 1. / tau
        self.decay_input = decay_input
        self.threshold = threshold
        self.device = device
        self.alpha = alpha
        self.mu = mu
        self.a = sigma
        self.n_ensembles = n_ensembles
        self.u = None
        self.o = None
             
    def _update(self, x):
        timestep = x.size(1)
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            u_ = self.u
            self.u += -torch.zeros_like(self.u).uniform_(-self.a, self.a)

            self.o[:, t, ...] = nilif_act_fun(self.u - self.threshold, self.a)

            self.u = self.u * (1 - self.o[:, t, ...]) 
    
    def _silent_update(self, x):
        timestep = x.size(1)
        output = []
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            output.append(self.u)
        return torch.stack(output, dim=1)
    
    def _reset(self):
        self.u  = None
        self.o  = None
    
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
