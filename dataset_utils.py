'''
Author: ----
Date: 2022-04-07 20:48:43
LastEditors: ----
LastEditTime: 2022-08-26 19:18:28
'''
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import random 
import numpy as np

from tqdm import tqdm


class Cutout(object):
    r"""
    Forked from https://github.com/uoguelph-mlrg/Cutout
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def prepare_cifar10(args, autoaugment=True):
    print('==> Preparing CIFAR-10 data..')
    if autoaugment:
        print('>>> Auto Augmentation')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.AutoAugment(
                policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/home/----/exd1/data/cifar10', 
        train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.minibatch, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(
        root='/home/----/exd1/data/cifar10', 
        train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.minibatch, shuffle=False, num_workers=args.workers)
    return trainloader, testloader


def prepare_cifar100(args, autoaugment=True):
    print('==> Preparing CIFAR-100 data..')
    if autoaugment:
        print('>>> Auto Augmentation')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.AutoAugment(
                policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='/home/----/exd1/data/cifar100', 
        train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.minibatch, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR100(
        root='/home/----/exd1/data/cifar100', 
        train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.minibatch, shuffle=False, num_workers=args.workers)
    return trainloader, testloader


def prepare_imagenet(args, autoaugment=True):
    print('==> Preparing ILSVRC-12 data..')
    if autoaugment:
        print('>>> Auto Augmentation')
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.AutoAugment(
                policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.ImageFolder(
        root='/home/----/exd1/data/imagenet/train', 
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.minibatch, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.ImageFolder(
        root='/home/----/exd1/data/imagenet/val', 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.minibatch, shuffle=False, num_workers=args.workers)
    print('data prepared')
    return trainloader, testloader


class DVSCIFAR10(torch.utils.data.Dataset):
    r'''
    Forked from https://github.com/Gus-Lab/temporal_efficient_training
    '''
    def __init__(
        self, 
        root, 
        train=True, transform=False, target_transform=None
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
    def __getitem__(self, index):
        r"""
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        
        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))

        
def prepare_dvs_cifar10(args):
    print('==> Preparing DVS-CIFAR-10..')
    root = '/home/----/exd1/data/dvs-cifar10'
    train_path = os.path.join(root, 'train')
    val_path   = os.path.join(root, 'test')
    trainset   = DVSCIFAR10(root=train_path, transform=True)
    testset    = DVSCIFAR10(root=val_path)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.minibatch, shuffle=True, num_workers=args.workers)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.minibatch, shuffle=False, num_workers=args.workers)

    print('data loaded')
    return trainloader, testloader
