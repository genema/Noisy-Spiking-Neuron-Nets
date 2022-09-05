'''
Author: ----
Date: 2022-06-15 16:27:54
LastEditors: ----
LastEditTime: 2022-06-15 16:38:09
'''
import torch
import torchvision.transforms as transforms
import os
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(size=[48, 48]),
        transforms.ToTensor()
    ]
)

root = '/home/----/exd1/data/dvs-cifar10/'

for subset in ['train', 'test']: 
    files = os.listdir(
        os.path.join(root, subset)
    )
    for file in tqdm(files): 
        data, label = torch.load(
            os.path.join(root, subset, file)
        )
        temp = []
        for t in range(10):
            temp.append(transform(data[t, ...]))
        temp = torch.stack(temp, dim=0)
        
        torch.save(
            [temp, label], 
            os.path.join(root, subset, file)
        )
        
print('done')