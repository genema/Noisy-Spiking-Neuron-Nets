'''
Author: ----
Date: 2022-04-16 11:47:44
LastEditors: GhMa
LastEditTime: 2022-10-02 19:25:28
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import uuid
import argparse
import glob
import shutil
import importlib

from models import *
from crits import *
from utils import progress_bar, visualize_grad_norms, visualize_weight_norms
from utils import Monitor, visualize_spiking_rates
from utils import TET_loss
from dataset_utils import prepare_cifar10

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='SNN Exps.')
parser.add_argument('--model', default='sresnet19', type=str, help='arch')

parser.add_argument('--lr', default=0.004, type=float, help='learning rate')
parser.add_argument('--epochs', default=240, type=int, help='#epoch')
parser.add_argument('--minibatch', default=128, type=int, 
                    help='mini-batch size')
# FOR NEURON
parser.add_argument('--neuron', default='NILIF', type=str, help='neuron type')
parser.add_argument('--timestep', default=2, type=int, help='timestep')
parser.add_argument('--threshold', default=1.0, type=float, 
                    help='spiking thresh')
parser.add_argument('--tau', default=2.0, type=float, help='initial tau')
parser.add_argument('--sigma', default=0.3, type=float, help='std of p_epsilon')

# FOR SURROGATE GRAD FUNC
parser.add_argument('--alpha', default=1.0, type=float, 
                    help='surrogate grad func hyperparam')
# Vanilla regularizers
parser.add_argument('--weight_decay', default=0.0, type=float, 
                    help='lagrangian factor of L2 norm term')

# Other settings
parser.add_argument('--seed', default=1000, type=int, help='random seed')
parser.add_argument('--workers', default=8, type=int, help='#threads')
parser.add_argument('--optim', default='adam', type=str, help='optimizer')
parser.add_argument('--scheduler', default='cos', type=str, help='lr scheduler')

# Exps
# valid options: "gram", "srip"
parser.add_argument('--reg', default='gram', type=str, help='ortho regularizers')
parser.add_argument('--lambda1', default=0.0001, type=float, 
                    help='lag. factor of ortho regularizers')

# valid options: "uniform", "orthog"
parser.add_argument('--init', default='uniform', type=str, 
                    help="specified weight initialization method. ")                
parser.add_argument('--plot_spike_rate', default=False, type=bool, 
                    help="plot spike rate or not. ") 

# TET loss
parser.add_argument('--tet', action='store_true', help='use TET loss')
parser.add_argument('--lambda_tet', default=0.0, type=float, help='lambda in TET')

# 
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--debug', action='store_true',
                    help='if debug is true, formal log files will not be created')
parser.add_argument('--nobn', action='store_true', help='disable BN')
args = parser.parse_args()

args.reg = False if args.reg not in ['gram', 'srip'] else args.reg
args.lambda1 = 0 if not args.reg else args.lambda1
norm_layer_type = 'bn' if not args.nobn else None

basic_neuron = importlib.import_module('models.' + args.neuron).Neuron

os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.tet:
    print('==> Use TET loss')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

trainloader, testloader = prepare_cifar10(args)
print('==> Building model..')

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
        norm_layer=norm_layer_type,
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
        norm_layer=norm_layer_type,
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

if args.init == 'uniform':
    pass
elif args.init == 'orthog':
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)

criterion = nn.CrossEntropyLoss()

if args.optim == 'sgdm':
    optimizer = optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, 
        weight_decay=args.weight_decay
    )
elif args.optim == 'adam':
    optimizer = optim.Adam(
        net.parameters(), 
        lr=args.lr, weight_decay=args.weight_decay
    )
elif args.optim == 'adamw':
    optimizer = optim.AdamW(
        net.parameters(), 
        lr=args.lr, weight_decay=args.weight_decay
    )
elif args.optim == 'radam':
    optimizer = optim.RAdam(
        net.parameters(), 
        lr=args.lr, weight_decay=args.weight_decay
    )
    
if args.scheduler == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    ) 
else:
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=5e-3, max_lr=args.lr
    )


def train(epoch, writer, scheduler, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    other_loss = 0
    correct    = 0
    total      = 0
    if args.plot_spike_rate:
        monitor = Monitor(net)
        monitor.enable()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        net.T = args.timestep
        inputs = inputs.unsqueeze_(1).repeat(1, args.timestep, 1, 1, 1)
        out_ = net(inputs)
        outputs = out_.mean(1)
        
        if args.reg:  # explicit regularizer
            if args.reg == 'gram':
                loss = criterion(outputs, targets)
                gram_loss = Gramian()
                loss2 = gram_loss(net)
                loss_ = loss + args.lambda1 * loss2
                other_loss += loss2.item()
            elif args.reg == 'srip':
                loss = criterion(outputs, targets)
                srip_loss = SRIP()
                loss2 = srip_loss(net)
                loss_ = loss + args.lambda1 * loss2
                other_loss += loss2.item()
        else:  # w/o explicit regularization term 
            if args.tet:  # use tet loss 
                loss = TET_loss(out_, targets, criterion, 1.0, args.lambda_tet)
            else:
                loss = criterion(outputs, targets)
            loss_ = loss

        loss_.backward()
        # NOTE: only for epoch-wise LR schedulers.
        
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.plot_spike_rate:
            visualize_spiking_rates(writer, monitor, epoch, srtype='train')
            monitor.reset()

        progress_bar(
            batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss / (batch_idx+1), 100. * correct / total, correct, total)
        )

    visualize_grad_norms(net, writer, epoch)
    visualize_weight_norms(net, writer, epoch)
    writer.add_scalar('loss / train', train_loss / (batch_idx+1), epoch)
    if args.reg:
        writer.add_scalar(
            'loss / train {}'.format(args.reg), 
            other_loss / (batch_idx+1), epoch
        )
    writer.add_scalar('acc / train', 100. * correct / total, epoch)


def test(epoch, writer, path):
    global best_acc
    net.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    if args.plot_spike_rate:
        monitor = Monitor(net)
        monitor.enable()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            net.T = args.timestep
            inputs = inputs.unsqueeze_(1).repeat(1, args.timestep, 1, 1, 1)
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
    
    writer.add_scalar('loss / test', test_loss/(batch_idx+1), epoch)
    writer.add_scalar('acc / test', 100.*correct/total, epoch)
    if args.plot_spike_rate:
        visualize_spiking_rates(writer, monitor, epoch, srtype='test')
        monitor.reset()
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        torch.save(
            state, 
            os.path.join(path, 'best.pth')
        )
        best_acc = acc


if __name__ == '__main__':
    uid = uuid.uuid4().hex
    path = os.path.join(
        './results/logs/cifar10-neuron_{}-model_{}-optim_{}-scheduler_{}-lr_{}-batch_{}-t_{}-vth_{}-tau_{}-alpha{}-sigma_{}-init_{}-TET_{}'.format(
            str(args.neuron),
            str(args.model),
            str(args.optim),
            str(args.scheduler),
            float(args.lr), 
            int(args.minibatch),
            int(args.timestep), 
            float(args.threshold),
            float(args.tau),
            float(args.alpha),
            float(args.sigma),
            str(args.init),
            float(args.lambda_tet),
        ), 
        str(uid)
    )

    if args.debug:
        writer = SummaryWriter('./')
    else:
        os.makedirs(path)
        script_path = os.path.join(path, 'scripts')
        os.makedirs(script_path)
        writer = SummaryWriter(path)

    files = list(glob.iglob(os.path.join('./', '*.sh'))) + \
            list(glob.iglob(os.path.join('./', '*.py'))) + \
            list(glob.iglob(os.path.join('./models', '*.py')))
    for file in files:
        if not os.path.isfile(file):
            continue
        shutil.copy2(
            file, os.path.join(
                script_path, file.replace('models/', 'models_')
            )
        )

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch, writer, scheduler, args)
        
        test(epoch, writer, path)
        scheduler.step()
    
    print('done')
