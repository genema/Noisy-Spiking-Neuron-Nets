'''
Author: ----
Date: 2022-06-29 14:32:02
LastEditors: ----
LastEditTime: 2022-08-01 16:07:15
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import importlib
import uuid
import random

from models import *
from crits import *
from utils import progress_bar
from utils import Monitor
from dataset_utils import prepare_dvs_cifar10
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='SNN Exps.')
parser.add_argument('--model', default='sresnet19', type=str, help='arch')

parser.add_argument('--minibatch', default=256, type=int, 
                    help='mini-batch size')
# FOR NEURON
parser.add_argument('--neuron', default='LIF', type=str, help='neuron type')
parser.add_argument('--timestep', default=10, type=int, help='timestep')
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
                    '[None, eventdrop]'
                    'for clean, EventDrop perturbation')

parser.add_argument('--drop_p', default=0.25, type=float, 
                    help='Event drop probability')

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
            n_input=[2, 48, 48],
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
            n_input=[2, 48, 48],
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
            n_input=[2, 48, 48],
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


def event_drop(inputs, args):
    r""""
    Random EventDrop, optimized re-implementation. 
    """
    bsz = inputs.size(0)
    p = args.drop_p
    for b in range(bsz):
        events = inputs[b].nonzero(as_tuple=True)
        n_events = events[0].size(0)
        """ drop_events = torch.LongTensor(
            random.sample(range(0, n_events), int(p*n_events))
        ) """
        drop_events = torch.randperm(n_events).long()[: int(p * n_events)]
        drop_events_idx = (
            (torch.zeros_like(events[0][drop_events]) + b).long(),
            events[0][drop_events], events[1][drop_events],
            events[2][drop_events], events[3][drop_events],
        )
        
        inputs = inputs.index_put(
            drop_events_idx, 
            torch.Tensor([0]).to(inputs.device)
        )
                            
    return inputs


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
    

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('test start')
    if args.perturbation == 'None':
        print('clean input')
    elif args.perturbation == 'eventdrop':
        print('EventDrop ', args.drop_p)
    elif args.perturbation == 'spike_level':
        print('spike-level perturbation', spn_p)
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if args.perturbation == 'eventdrop' and args.drop_p > 0:
                inputs = event_drop(inputs, args)

            net.T = 10
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


_, testloader = prepare_dvs_cifar10(args)
print('==> Building model..')

net, checkpoint = net_init(args)
best_acc, start_epoch = checkpoint['acc'], checkpoint['epoch']
print('>> ckpt Acc: {} Epoch: {}'.format(best_acc, start_epoch))

criterion = nn.CrossEntropyLoss()
    
    
if __name__ == '__main__':
    if args.range:
        uid = uuid.uuid4().hex
        resfname = 'DVSCIFAR_{}_{}_{}_{}_{}_{}_{}.txt'.format(
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
            args.drop_p = ppp 
            spn_p = ppp
            if args.neuron == 'LIF_noise_test':
                args.sigma = ppp
                print('membrane potential perturbation', ppp)
                net, _ = net_init(args)
            elif args.neuron == 'NILIF_noise_test':
                args.alpha = ppp 
                print('membrane potential perturbation', ppp)
                net, _ = net_init(args)
            
            loss, acc = test()
            with open(resfpath, 'a') as f:
                f.write('{}, {}, {}\n'.format(ppp, loss, acc))
        print('done')
    else:        
        test()
        print('done')
