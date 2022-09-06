'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    r'''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    r'''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_grad_norm(model_params, norm_type='fro', step_size=None):
    norm = 0
    for p in model_params:
        norm += torch.linalg.norm(p.grad.detach().data)
    return norm


def visualize_grad_norms(model, writer, idx, norm_type=2, step_size=None):
    conv_counter = 0
    fc_counter = 0
    n_conv = 0
    n_fc = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d): 
            n_conv += 1
        elif isinstance(m, nn.Linear):
            n_fc += 1

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d): 
            if conv_counter == 0 or conv_counter == n_conv-1:
                for n, p in m.named_parameters():
                    writer.add_histogram('grad / '+name+n, p.grad, idx)
            conv_counter += 1
        elif isinstance(m, nn.Linear):
            if fc_counter == 0 or fc_counter == n_fc-1:
                for n, p in m.named_parameters():
                    writer.add_histogram('grad / '+name+n, p.grad, idx)
            fc_counter += 1


def get_weight_norm(model_parameters, norm_type='fro'):
    r"""
    Return avg. weight norm (default 2-norm, F-norm for matrices) per layer
    """
    norm = 0
    counter = 0
    for p in model_parameters:
        counter += 1
        norm += torch.linalg.norm(p.data)
    return norm / counter


def visualize_weight_norms(model, writer, idx, norm_type='fro'):    
    conv_counter = 0
    fc_counter = 0
    n_conv = 0
    n_fc = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d): 
            n_conv += 1
        elif isinstance(m, nn.Linear):
            n_fc += 1

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d): 
            if conv_counter == 0 or conv_counter == n_conv-1:
                for n, p in m.named_parameters():
                    writer.add_histogram('param / '+name+n, p, idx)
            conv_counter += 1
        elif isinstance(m, nn.Linear):
            if fc_counter == 0 or fc_counter == n_fc-1:
                for n, p in m.named_parameters():
                    writer.add_histogram('param / '+name+n, p, idx)
            fc_counter += 1


class Monitor:
    r"""
    Forked from Spikingjelly
    """
    def __init__(self, model):
        self.model = model
        self.modules = {}
        for n, m in self.model.named_modules():
            if 'sn' in n:
                self.modules[n] = m
        self.device = 'cpu'
    
    def _modules(self):
        return self.modules
    
    def reset(self):
        for name, module in self.modules.items():
            setattr(module, 'firing_time', 0)
            setattr(module, 'cnt', 0)
            
    def enable(self,):
        self.handle = dict.fromkeys(self.modules, None)
        self.neuron_cnt = dict.fromkeys(self.modules, None)
        for name, module in self.modules.items():
            setattr(module, 'neuron_cnt', self.neuron_cnt[name])
            self.handle[name] = module.register_forward_hook(self.hook)   
        self.reset()
    
    def disable(self):
        for name, module in self.modules.items():
            delattr(module, 'neuron_cnt')
            delattr(module, 'firing_time')
            delattr(module, 'cnt')
            self.handle[name].remove()
    
    @torch.no_grad()    
    def hook(self, module, input, output):
        output_shape = output.shape
        data = output.view([-1, ] + list(output_shape[2:])).clone() 
        data = data.to(self.device)
        if module.neuron_cnt is not None:
            module.neuron_cnt = data[0].numel()
        module.firing_time += torch.sum(data)
        module.cnt += data.numel()
        
    def get_avg_firing_rate(
            self, all: bool = True, module_name: str = None
    ) -> torch.Tensor or float:
        if module_name is not None:
            all = False

        if all:
            ttl_firing_time = 0
            ttl_cnt = 0
            for name, module in self.modules.items():
                ttl_firing_time += module.firing_time
                ttl_cnt += module.cnt
            return ttl_firing_time / (ttl_cnt + 1e-6)
        else:
            if module_name not in self.modules.keys():
                raise ValueError(f'Invalid module_name \'{module_name}\'')
            module = self.modules[module_name]
            return module.firing_time / (module.cnt + 1e-6)


def visualize_spiking_rates(writer, monitor, idx, srtype='train'):
    layer_counter = 0
    for m in monitor._modules().keys():
        layer_counter += 1
        sr = monitor.get_avg_firing_rate(all=False, module_name=m)
        writer.add_scalar(
            'Spiking rate {} / Layer {}'.format(srtype, layer_counter),
            sr, idx
        )
        

def TET_loss(outputs, labels, criterion, means, lamb):
    r"""
    Forked from https://github.com/Gus-Lab/temporal_efficient_training
    lambda = 0.05 for CIFAR
    lambda = 0.001 for imagenet
    """
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0

    return (1 - lamb) * Loss_es + lamb * Loss_mmd 
