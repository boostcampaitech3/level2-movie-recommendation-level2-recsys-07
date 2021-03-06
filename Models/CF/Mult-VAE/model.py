import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from loss import *

class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)



class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims #[200, 600, 6807]
        if q_dims: # ?????? encoder ????????? decoder??? ????????? ???????????? ?????????, ??????????????? ??????. ?????? encoder??? output ????????? decoder??? input ????????? ???????????? ??????.
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1] # [6807, 600, 200] decoder??? ????????? ?????? ?????? -> ???????????? ???????????? ??????, ?????? ?????? ?????? 200?????? ??????. 

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2] # temp_q_dims = [6807, 600, 400] -> ????????? ????????? 2??? ????????? ?????? ????????? ????????? ?????? ????????? ?????? ????????????.
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]) # encoder : nn.Linear([6807, 600]), nn.Linear([600, 400])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]) # decoder : nn.Linear([200, 600]) , nn.Linear([600,6807])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input) # encoder??? ?????? ????????? ?????? ?????? ??????
        z = self.reparameterize(mu, logvar) # ????????? ???????????? sampling -> ?????? ?????? ???????????? 200?????? ?????? ?????????
        return self.decode(z), mu, logvar # sample?????? decoder??? ?????? ?????? ?????? ??? ?????? #R??????
    
    def encode(self, input):
        h = F.normalize(input) # ????????? ?????????
        h = self.drop(h) # dropout rate?????? ????????? ???????????? 0?????? ??????????????? -> ????????? item??? ?????? ???????????? ?????????, overfitting??? ???????????? ?????? ????????? 0?????? ????????????.
        
        for i, layer in enumerate(self.q_layers): 
            h = layer(h) 
            if i != len(self.q_layers) - 1: # ?????? ???????????? ????????????, tanh??? ????????????
                h = F.tanh(h)
            else: # ?????? ????????? layer??????
                mu = h[:, :self.q_dims[-1]] # ?????? ?????? [0 : 200]??? ??????
                logvar = h[:, self.q_dims[-1]:] # ?????? ?????? [200 : 400]??? ??????
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar) # e^(0.5*log(std^2)) = e^log(std) = std
            eps = torch.randn_like(std) # std??? ???????????? N(0,1) normal distribution?????? ??????
            return eps.mul(std).add_(mu) # eps x std + mu
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers): 
            h = layer(h)
            if i != len(self.p_layers) - 1: # ?????? ????????? layer??? ???????????? tanh??? ????????????
                h = F.tanh(h) # TODO : ??? ???????????? softmax??? ???????????? ??????????
                # softmax??? ????????? ?????? ?????? loss function?????? ???????????? ?????????

        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)