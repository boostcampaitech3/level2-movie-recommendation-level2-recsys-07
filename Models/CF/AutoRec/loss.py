""" how to use in train.py
from loss import create_criterion
criterion = create_criterion(args.criterion) # default: bce_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_criterion_entrypoints = {
    'bce_loss': nn.BCELoss,
    'mse_loss': nn.MSELoss
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

class AutoRec_loss_fn(nn.Module):
    """
    AutoRec_loss_fn
    
    Args:
        - loss_fn: (nn.Module) 사용할 Loss Function
    Shape:
        - Input1: (torch.Tensor) Model의 예측 결과. Shape: (batch size,)
        - Input2: (torch.Tensor) 정답. Shape: (batch size,)
        - Output: (torch.Tensor) Observable한 데이터에 대해서만 계산한 Loss. Shape: ()
    """
    def __init__(self, loss_fn):
        super(AutoRec_loss_fn, self).__init__()
        self.loss_fn = loss_fn
    
    def forward(self, pred, y):
        y_for_compute = y.clone().to('cpu')
        index = np.where(y_for_compute != 0)
        loss = self.loss_fn(pred[index], y[index])
        return loss