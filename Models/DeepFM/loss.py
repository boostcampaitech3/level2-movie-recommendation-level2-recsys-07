""" how to use in train.py
from loss import create_criterion
criterion = create_criterion(args.criterion) # default: bce_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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