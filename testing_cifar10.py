'''
Author: ----
Date: 2022-06-14 19:48:48
LastEditors: ----
LastEditTime: 2022-09-05 15:10:42
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import importlib
import geotorch
import uuid
import random

from models import *
from crits import *
from utils import progress_bar
from utils import Monitor
from dataset_utils import prepare_cifar10
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='SNN Exps.')
parser.add_argument('--model', default='sresnet19', type=str, help='arch')

parser.add_argument('--minibatch', default=256, type=int, 
                    help='mini-batch size')
# FOR NEURON
parser.add_argument('--neuron', default='LIF', type=str, help='neuron type')
parser.add_argument('--timestep', default=2, type=int, help='timestep')
parser.add_argument('--threshold', default=1.0, type=float, 
                    help='spiking thresh')
parser.add_argument('--tau', default=2.0, type=float, help='initial tau')
parser.add_argument('--sigma', default=0.4, type=float, help='std of p_epsilon')
# FOR SURROGATE GRAD FUNC
parser.add_argument('--alpha', default=1.0, type=float, 
                    help='surrogate grad func hyperparam')

# Other settings
parser.add_argument('--seed', default=1000, type=int, help='random seed')
parser.add_argument('--workers', default=8, type=int, help='#threads')

# FOR TESTING
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot path')                    
parser.add_argument('--n_ensembles', default=1, type=int,
                    help='testing ensemble number')
parser.add_argument('--perturbation', default='None', type=str, 
                    help='type of test sample perturbation,'
                    '[None, add_gaussian, multi_gaussian, sp, adversarial]'
                    'for clean, additive white noise, multiplicative white noise,'
                    'salt&pepper noise and adversatial perturbation'
                    '(adversarial attack)')

parser.add_argument('--gaussian_sigma', default=0.1, type=float, 
                    help='std (sqrt-var) for gaussian noise')
parser.add_argument('--sp_alpha', default=0.03, type=float, 
                    help='alpha for salt&pepper noise')
parser.add_argument('--adv_gamma', default=0.1, type=float, 
                    help='gamma for restrict 1-norm of entries adversarial attack')

parser.add_argument('--range', action='store_true', help='if range, run test on a range')
parser.add_argument('--left', type=float, help='range left')
parser.add_argument('--right', type=float, help='range left')
parser.add_argument('--step', type=float, help='range left')
args = parser.parse_args()

basic_neuron = importlib.import_module('models.' + args.neuron).Neuron
    
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
spn_p = 0.1


def net_init(args, ):
    if args.model == 'sresnet19':
        net = spiking_resnet19(
            spiking_neuron=basic_neuron,
            n_input=[3, 32, 32],
            n_output=10,
            decay_input=False,  # for neuron
            threshold=args.threshold,  # for neuron
            tau=args.tau,  # for neuron 
            alpha=args.alpha,  # for neuron
            sigma=args.sigma,  # for Noisy neuron
        )
    elif args.model == 'sresnet18':
        net = spiking_resnet18(
            spiking_neuron=basic_neuron,
            n_input=[3, 32, 32],
            n_output=10,
            decay_input=False,  # for neuron
            threshold=args.threshold,  # for neuron
            tau=args.tau,  # for neuron 
            alpha=args.alpha,  # for neuron
            sigma=args.sigma,  # for Noisy neuron
        )
    elif args.model == 'vgg':
        net = VGGSNN(
            spiking_neuron=basic_neuron,
            n_input=[3, 32, 32],
            n_output=10,
            decay_input=False,  # for neuron
            threshold=args.threshold,  # for neuron
            tau=args.tau,  # for neuron 
            alpha=args.alpha,  # for neuron
            sigma=args.sigma,  # for Noisy neuron
        )
    elif args.model == 'cifarnet':
        net = CIFARNet(
            spiking_neuron=basic_neuron,
            n_input=[3, 32, 32],
            n_output=10,
            decay_input=False,  # for neuron
            threshold=args.threshold,  # for neuron
            tau=args.tau,  # for neuron 
            alpha=args.alpha,  # for neuron
            sigma=args.sigma,  # for Noisy neuron
        )

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.snapshot)
    net.load_state_dict(checkpoint['net'])
    net = net.module
    return net, checkpoint


def spike_noise_hook(m, input, output):
    global spn_p
    bsz, ts = output.size(0), output.size(1)
    out_ = output.view(bsz, -1)
    for b in range(bsz):
        events = out_[b].nonzero(as_tuple=True) 
        nothing = (out_[b] == 0).nonzero(as_tuple=True)

        n_events = events[0].size(0)
        n_nothing = nothing[0].size(0)

        """ drop_events = torch.LongTensor(
            random.sample(range(0, n_events), int(spn_p * n_events))
        ) """
        drop_events = torch.randperm(n_events).long()[: int(spn_p * n_events)]
        drop_events_idx = (
            (torch.zeros_like(events[0][drop_events]) + b).long(),
            events[0][drop_events],
        )
        out_ = out_.index_put(
            drop_events_idx, 
            torch.Tensor([0]).to(output.device)
        )

        """ add_events = torch.LongTensor(
            random.sample(range(0, n_nothing), int(spn_p * n_nothing))
        ) """
        add_events = torch.randperm(n_nothing).long()[: int(spn_p * n_nothing)]
        add_events_idx = (
            (torch.zeros_like(nothing[0][add_events]) + b).long(),
            nothing[0][add_events],
        )
        out_ = out_.index_put(
            add_events_idx, 
            torch.Tensor([1]).to(output.device)
        )
        
    return out_.view(output.size())


class Attacker:
    r"""
    A serious class for gradient-based input-level attack! 
    """
    def __init__(
        self, 
        target, steps, eps, 
        type='do',
    ):
        self.target = target
        self.n_steps = steps
        self.eps = eps
        self.delta = None
        self.opt = None
        self.type = type  # valid options: [fgsm, do] for 
                          # fast gradient sign method, direct optimzation 

    def inp_grad_calc(x, labels, model, loss_func):
        r"""
        Forked from https://github.com/ssharmin/spikingNN-adversarial-attack

        Calculates the gradient of error with respect to input (del_loss/del_input)
        :param x: input
        :param labels: The labels used to calculate the loss or error
        :param model: Source model
        :loss_func: Loss function used to calculate the error or loss
        """
        inputs = x.clone().detach()
        inputs.requires_grad_(True)
        inputs.grad = None       
        out_channel = model.module.features[0].weight.size()[0]  
        in_channel = model.module.features[0].weight.size()[1]
        weight_new = np.zeros((model.module.features[0].weight.size()))
        for i in range(out_channel):
            for j in range(in_channel):
                weight_new[i, j, 0, 0:] = np.flip(
                    model.module.features[0].weight.detach().cpu(
                    ).numpy()[i, j, 2, 0:]
                )
                weight_new[i, j, 1, 0:] = np.flip(
                    model.module.features[0].weight.detach().cpu(
                    ).numpy()[i, j, 1, 0:]
                )
                weight_new[i, j, 2, 0:] = np.flip(
                    model.module.features[0].weight.detach().cpu(
                    ).numpy()[i, j, 0, 0:]
                )
        weight_new = np.transpose(weight_new, (1, 0, 2, 3))
        weight_rotate = torch.from_numpy(weight_new).float().cuda()
        
        output, input_spike_count, grad_mem_conv1 = model(inputs, 0, True)
        output = output/model.module.timesteps
            
        error = loss_func(output, labels)
        error.backward()        
        grad_mem_conv1 = model.module.grad_mem1
        inp_grad = F.conv2d(grad_mem_conv1, weight_rotate, padding=1)
        # reset        
        inputs.requires_grad_(False)
        inputs.grad = None
        model.module.zero_grad()
        return inp_grad
    
    def fgsm_attack(self, image, epsilon, data_grad):
        r"""
        forked from PyTorch official doc:
        https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        # perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # we need to remove this as our data transformations are not end with 
        # ToTensor (which results in [0, 1]) but end with z-score normalisation.

        # Return the perturbed image
        return perturbed_image

    def make_adversarial_example(self, x, label):
        if self.type == 'fgsm':  # Fast gradient sigh method
            self.target.eval()
            self.target.T = args.timestep
            x = x.unsqueeze_(1).repeat(1, args.timestep, 1, 1, 1)
            x.requires_grad = True
            y = self.target(x).mean(1)
            loss = F.cross_entropy(y, label)
            self.target.zero_grad()
            loss.backward()
            data_grad = x.grad.data
            x_perturb = self.fgsm_attack(x, self.eps, data_grad)
            # print(x.size(), x_perturb.size())
            return x_perturb

        else:  # direct optimize method
            # this method explicitely solve the constrained-optimization problem 
            # on a sphere manifold, which is much slower than the FGSM. 
            self.target.eval()
            bsz = x.size(0)
            dim = x.view(bsz, -1).size(-1)
            self.delta = nn.ModuleList()
            for i in range(bsz):
                self.delta.append(nn.Linear(1, dim).to(x.device))
                nn.init.zeros_(self.delta[i].weight)
                self.delta[i].weight.requires_grad = False
                nn.init.zeros_(self.delta[i].bias)
                geotorch.sphere(self.delta[i], 'bias', radius=self.eps)

            self.opt = torch.optim.Adam(self.delta.parameters(), lr=0.002)

            for _ in (range(self.n_steps)):
                self.opt.zero_grad()
                self.target.T = args.timestep
                x_perturb = x.clone()
                for i in range(bsz):
                    x_perturb[i] = (x.view(bsz, -1)[i] + self.delta[i].bias).view(
                        x[i].size()
                    )
                
                x_perturb = x_perturb.unsqueeze_(1).repeat(
                    1, args.timestep, 1, 1, 1)
                y = self.target(x_perturb).mean(1)
                loss = -F.cross_entropy(y, label)
                loss.backward()
                self.opt.step()

            for i in range(bsz):
                x[i] = (x.view(bsz, -1)[i] + self.delta[i].bias).view(x[i].size())

            return x


def test():
    global best_acc
    global spn_p
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('test start')
        
    if args.perturbation == 'None':
        print('clean input')
    elif args.perturbation == 'add_gaussian':
        print('additive white noise ', args.gaussian_sigma)
    elif args.perturbation == 'multi_gaussian':
        print('multiplicative white noise ', args.gaussian_sigma)
    elif args.perturbation == 'sp':
        print('pepper noise ', args.sp_alpha)
    elif args.perturbation == 'adversarial':
        print('adversarial attack ', args.adv_gamma)
        ################################################
        # if adversarial attack, initialize an attacker!
        # attacker = Attacker(net, steps=30, eps=args.adv_gamma, type='fgsm')
        attacker = Attacker(net, steps=30, eps=args.adv_gamma, type='do')
    elif args.perturbation == 'spike_level':
        print('spike-level perturbation', spn_p)
        
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
            
        if args.perturbation == 'adversarial' and args.adv_gamma > 0:
            inputs = attacker.make_adversarial_example(inputs, targets)

        net.T = args.timestep
        if args.perturbation == 'adversarial' and attacker.type == 'fgsm':
            inputs = inputs 
        else:    
            inputs = inputs.unsqueeze_(1).repeat(1, args.timestep, 1, 1, 1)
        if args.neuron == 'NILIF':
            outputs = 0
            n_ensembles = args.n_ensembles  
            for _ in range(n_ensembles):
                outputs += net(inputs).mean(1)
            outputs /= n_ensembles
        else:
            outputs = net(inputs).mean(1)
        
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(
            batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )
            
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    return test_loss/(batch_idx+1), 100.*correct/total


_, testloader = prepare_cifar10(args)
print('==> Building model..')

net, checkpoint = net_init(args)
best_acc, start_epoch = checkpoint['acc'], checkpoint['epoch']
print('>> ckpt Acc: {} Epoch: {}'.format(best_acc, start_epoch))
criterion = nn.CrossEntropyLoss()
    
    
if __name__ == '__main__':
    if args.range:
        uid = uuid.uuid4().hex
        resfname = 'CIFAR-10_{}_{}_{}_{}_{}_{}_{}.txt'.format(
            args.neuron, args.model, 
            args.perturbation, args.left, args.right, args.step, uid
        )
        resfpath = os.path.join('./results/test_results', resfname)
        with open(resfpath, 'w') as f:
            f.write(resfname + '\n')
        
        if args.perturbation == 'spike_level':
            if spn_p > 0:
                for n, m in net.named_modules():
                    if 'sn' in n:
                        m.register_forward_hook(spike_noise_hook)

        for ppp in tqdm(np.arange(args.left, args.right+1e-6, args.step)):
            args.gaussian_sigma = ppp 
            args.sp_alpha = ppp
            args.adv_gamma = ppp
            spn_p = ppp
            if args.neuron == 'LIF_noise_test':
                args.sigma = ppp
                print('membrane potential perturbation', ppp)
                net, _ = net_init(args)
            elif args.neuron == 'NILIF_noise_test':
                args.alpha = ppp 
                print('membrane potential perturbation', ppp)
                net, _ = net_init(args)

            if args.perturbation == 'adversarial':
                loss, acc = test()
            else:
                with torch.no_grad():
                    loss, acc = test()

            with open(resfpath, 'a') as f:
                f.write('{}, {}, {}\n'.format(ppp, loss, acc))
        print('done')
    else:        
        test()
        print('done')
