'''
Author: ----
Date: 2022-04-09 14:24:09
LastEditors: ---a
LastEditTime: 2022-09-05 14:46:08
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gramian(nn.Module):
    r"""
    Enforcing weight matix to be orthogonal by restricting Gramians
    """
    def __init__(self) -> None:
        super(Gramian, self).__init__()
    
    def forward(self, model):
        loss = 0.
        for m in model.modules():
            if not hasattr(m, 'weight'):
                continue
            w = getattr(m, 'weight')
            
            if isinstance(m, nn.Conv2d):
                w = w.view(w.size(0), -1).T
            elif isinstance(m, nn.Linear):
                w = w.T
            elif isinstance(m, nn.BatchNorm2d):
                continue

            g = w.T.matmul(w)
            i = torch.eye(g.size(0)).to(g.device)
            loss += torch.linalg.norm(g-i, ord='fro')
        
        return loss


class SRIP(nn.Module):
    r"""
    Enforcing weight matix to be orthogonal by Resitricted Isometry Property
    """
    def __init__(self) -> None:
        super(SRIP, self).__init__()
    
    def forward(self, model):
        loss = 0.
        for m in model.modules():
            if not hasattr(m, 'weight'):
                continue
            w = getattr(m, 'weight')
            if isinstance(m, nn.Conv2d):
                rows = w.size(0)
                cols = w[0].numel()
                if rows > cols:
                    w = w.view(w.size(0), -1)
                else:
                    w = w.view(w.size(0), -1).T
            elif isinstance(m, nn.Linear):
                w = w.T
            elif isinstance(m, nn.BatchNorm2d):
                continue
            g = w.T.matmul(w)
            i = torch.eye(g.size(0)).to(g.device)
            w_tmp = g-i

            u = torch.rand(w_tmp.size(1), 1).to(g.device)
            v = torch.matmul(w_tmp, u)
            v = F.normalize(v, p=2, dim=0)
            v = torch.matmul(w_tmp, v)

            loss += torch.linalg.norm(v, ord=2) ** 2
        
        return loss
