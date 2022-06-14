import numpy as np
import random
import pandas as pd
from datetime import datetime, date
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader


def activation_layer(activation_name='relu'):
    """
    Construct activation layers
    
    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation
    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
        elif activation_name.lower() == "identity" :
            activation = nn.Identity()
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation

    
class AutoRec(nn.Module) :

    def __init__(self, args, input_dims):
        super(AutoRec, self).__init__()
        
        self.args = args
        # initialize Class attributes
        self.input_dim = input_dims
        self.emb_dim = args.embedding_dim
        self.hidden_activation = args.hidden_activation
        self.out_activation = args.out_activation

        # define layers
        self.encoder = nn.Linear(self.input_dim, self.emb_dim)
        self.hidden_activation_function = activation_layer(self.args.hidden_activation)
        self.decoder = nn.Linear(self.emb_dim, self.input_dim)
        self.out_activation_function = activation_layer(self.out_activation)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    
    def forward(self, input_feature):
        h = self.encoder(input_feature)
        h = self.hidden_activation_function(h)
        
        output = self.decoder(h)
        output = self.out_activation_function(output)
        
        return output
    
