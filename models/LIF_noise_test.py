'''
Author: ----
Date: 2022-07-20 16:43:38
LastEditors: ----
LastEditTime: 2022-07-20 21:02:53
'''
import torch
import torch.nn as nn
import numpy as np
import math


def heaviside(x: torch.Tensor):
    return (x >= 0.0).to(x)


class SigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, alpha, threshold
    ):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return heaviside(input-threshold)
    
    @staticmethod
    def backward(
        ctx, 
        grad_output
    ):
        grad_ = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1.0 - sgax) * sgax * ctx.alpha
        return grad_x, None, None


class AtanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (
                1 + math.pi / 2 * ctx.alpha * ctx.saved_tensors[0].pow_(2)
            ) * grad_output
        return grad_x, None


class RectangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=0.8
    """
    @staticmethod
    def forward(ctx, input, threshold, alpha):
        ctx.save_for_backward(input) 
        ctx.threshold = threshold
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (2*abs(input-ctx.threshold) < ctx.alpha) * 1. / ctx.alpha
        return grad_input * temp, None, None


class TriangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=1.0
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input) 
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (1 / ctx.alpha) * (1 / ctx.alpha) * (
            (ctx.alpha - input.abs()).clamp(min=0)
        )
        return grad_input * temp, None


act_fun = TriangularSurrogate.apply


class Neuron(nn.Module):
    r"""
    Iterative LIF neuron model.
    """
    def __init__(
        self,
        threshold=1.0,
        decay_input=False,
        tau=2.0,
        alpha=1.,
        sigma=0.5,
        device='cuda'
    ) -> None:
        super().__init__()
        self.tau = 1. / tau
        self.decay_input = decay_input
        self.threshold = threshold
        self.device = device
        self.alpha = alpha
        self.sigma = sigma
        self.u = torch.zeros(1, requires_grad=True).to(device)
        self.o = torch.zeros(1, requires_grad=True).to(device)
             
    def _update(self, x):
        timestep = x.size(1)
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            # membrane potential-level background noise, the idle parameter sigma 
            # is used as the std of the noise. 
            self.u += -torch.normal(
                torch.ones_like(self.u) * 0, self.sigma
            ) 
            self.o[:, t, ...] = act_fun(self.u - self.threshold, self.alpha)
            self.u = self.u * (1 - self.o[:, t, ...])
    
    def _silent_update(self, x):
        timestep = x.size(1)
        output = []
        for t in range(timestep):
            self.u = self.tau * self.u + x[:, t, ...]
            output.append(self.u)
        return torch.stack(output, dim=1)
    
    def _reset(self):
        self.u = torch.zeros(1, requires_grad=True).to(self.device)
        self.o = torch.zeros(1, requires_grad=True).to(self.device)
    
    def _states(self):
        return self.u, self.o
    
    def forward(self, x, out_u=False):
        self.u = torch.zeros(1, requires_grad=True).to(self.device)
        self.o = torch.zeros_like(x) 
        if out_u:
            return self._silent_update(x)
        else:
            self._update(x)
            return self.o
