'''
Author: ----
Date: 2022-04-29 09:43:08
LastEditors: ----
LastEditTime: 2022-09-03 13:12:09
'''
import torch
import torch.nn as nn
import numpy as np
import math
        

def heaviside(x: torch.Tensor):
    return (x >= 0.0).to(x)


class StochasticST(torch.autograd.Function):
    r"""
    Stochastic straight-through gradient estimator
    default noise type is gaussian noise, since the original form is derived from
    Ito SDE. 
    """
    @staticmethod
    def forward(ctx, input, mu, sigma):
        ctx.save_for_backward(input) 
        ctx.mu = mu
        ctx.sigma = sigma
        ctx.p_spike = 1/2 * (
            1 + torch.erf((input - mu) / (sigma * math.sqrt(2)))
        )
        return torch.bernoulli(ctx.p_spike)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temp = (
            1 / (math.sqrt(2*math.pi) * ctx.sigma)
        ) * torch.exp(
            -0.5 * ((input - ctx.mu) / ctx.sigma).pow_(2)
        )
        grad_input = grad_output.clone()
        return grad_input * temp, None, None
        
        
nilif_act_fun = StochasticST.apply


class Neuron(nn.Module):
    r"""
    Noise injected LIF neuron, Normal(mu, sigma^2)
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
        self.sigma = sigma
        self.n_ensembles = n_ensembles
        self.u = None
        self.o = None
             
    def _update(self, x):
        timestep = x.size(1)
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            u_ = self.u
            # additive noise
            self.u += -torch.normal(
                torch.ones_like(self.u) * self.mu, self.sigma
            )  

            # if self.training:
            self.o[:, t, ...] = nilif_act_fun(
                self.u - self.threshold, 
                self.mu, self.sigma
            )
            # else:
            #    self.o[:, t, ...] = heaviside(self.u - self.threshold)

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
        # self.o_ = None
    
    def _states(self):
        return self.u, self.o
    
    def forward(self, x, out_u=False):
        self.u = 0
        self.o = torch.zeros_like(x) 
        # self.o_ = torch.zeros_like(x)
        if out_u:
            return self._silent_update(x)
        else:
            self._update(x)
            # if self.training:  # principled stochastic inference 
            return self.o
            # else:  # this corresponds to deterministic inference with noise 
            #     return self.o_
