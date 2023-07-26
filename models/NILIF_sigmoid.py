'''
Author: ----
Date: 2022-06-30 11:22:52
LastEditors: GhMa
LastEditTime: 2022-09-20 14:35:24
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
    The noise type here is sigmoid noise. 
    """
    @staticmethod
    def forward(ctx, input, mu, scale):
        ctx.save_for_backward(input) 
        ctx.mu = mu
        ctx.scale = scale
        ctx.p_spike = torch.special.expit((input - ctx.mu) / (ctx.scale + 1e-8)).nan_to_num_()
        return torch.bernoulli(ctx.p_spike)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temp = torch.exp(
            -(input - ctx.mu) / ctx.scale
        ) / ctx.scale / (1 + torch.exp(-(input - ctx.mu) / ctx.scale)).pow_(2)
        grad_input = grad_output.clone()
        return grad_input * temp, None, None
        
        
nilif_act_fun = StochasticST.apply


class Neuron(nn.Module):
    r"""
    Noise injected LIF neuron, sigmoid noise Logistic(mu, scale)
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
        self.scale = sigma
        self.n_ensembles = n_ensembles
        self.u = None
        self.o = None
             
    def _update(self, x):
        timestep = x.size(1)
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            u_ = self.u
            noise = torch.zeros_like(self.u).uniform_(0, 1)
            noise = self.mu + self.scale * (
                torch.log((noise+1e-7) / (1-noise+1e-7)))
            self.u += -noise

            self.o[:, t, ...] = nilif_act_fun(
                self.u - self.threshold, 
                self.mu, self.scale
            )
            
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
