import numpy as np

import torch
import torch.nn as nn

class FFMLayer(nn.Module):
    def __init__(self, field_dims, factor_dim):
        '''
        Parameter
            field_dims: List of field dimensions 
                        The sum become the entire dimension of input (in sparse feature)
                        The length become the number of fields
            factor_dim: Factorization dimension
        '''
        super().__init__()
        self.num_fields = len(field_dims)
        self.input_dim = sum(field_dims) #.to(device)
        self.embedding = nn.ModuleList([
            # FILL HERE : Fill in the places `None` with                                      #
            #             either `factorization_dim`, `self.num_fields`, or `self.input_dim`. #
            nn.Embedding(
                self.input_dim, factor_dim
            ) for _ in range(self.num_fields)
        ])

    def forward(self, x):
        '''
        Parameter
            x: Long tensor of size "(batch_size, num_fields)"
               Each value of variable is an index calculated including the dimensions up to the previous variable.
               for instance, [gender:male, age:20, is_student:True] 
                             -> [1,0, 0,1,0,0,0,0, 0,1] in one-hot encoding
                             -> x = [0,3,9].
        Return
            y: Float tensor of size "(batch_size)"
        '''
        
        xv = [self.embedding[f](x) for f in range(self.num_fields)]
        
        y = list()
        for f in range(self.num_fields):
            for g in range(f + 1, self.num_fields):
                y.append(xv[f][:, g] *  xv[g][:, f])
        y = torch.stack(y, dim=1)
        
        return torch.sum(y, dim=(2,1))

class FieldAwareFM(nn.Module):
    def __init__(self, field_dims, factor_dim, device):
        '''
        Parameter
            field_dims: List of field dimensions
            factor_dim: Factorization dimension
        '''
        super().__init__()
        self.input_dim = sum(field_dims)
        self.encoding_dims = np.concatenate([[0], np.cumsum(field_dims)[:-1]])
        self.linear = nn.Linear(self.input_dim, 1, bias=True)
        self.ffm = FFMLayer(field_dims, factor_dim)
        self.device = device # torch.zero.to(device)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, FFMLayer):
                nn.init.normal_(m.v, 0, 0.01)

    def forward(self, x):
        '''
        Parameter
            x: Long tensor of size "(batch_size, num_fields)"
                x_multihot: Multi-hot coding of x. size "(batch_size, self.input_dim)"
        
        Return
            y: Float tensor of size "(batch_size)"
        '''
        x = x + x.new_tensor(self.encoding_dims).unsqueeze(0)
        x_multihot = torch.zeros(x.size(0), self.input_dim, device=self.device).scatter_(1, x, 1.)
        
        y = self.linear(x_multihot).squeeze(1) + self.ffm(x)

        return y