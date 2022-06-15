import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class GNN_ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.3, init_max=0.3):
        super(GNN_ConcreteDropout, self).__init__()

        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x, c_adjs1, layer):
        p = torch.sigmoid(self.p_logit)
        
        p = p.cuda()
        out = layer(self.concrete_dropout(x, p), c_adjs1)
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out
        
    def concrete_dropout(self, x , p):
        # eps = 1e-7
        eps = 1e-4
        temp = 0.1
        p = torch.sigmoid(self.p_logit)
        p=p.cuda()
        eps=torch.tensor(eps).cuda()
        temp=torch.tensor(temp).cuda()

        unif_noise = torch.rand_like(x)
        unif_noise = unif_noise.cuda()
        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x = torch.mul(x, random_tensor) / retain_prob
        
        return x