import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import torch.nn as nn
class gnn_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(gnn_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj ):
        h = self.W(x)   
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention*adj
        # h_prime = F.leaky_relu(torch.einsum('aij,ajk->aik',(attention, h)))
        h_prime = F.leaky_relu(torch.einsum('aij,ajk->aik',(adj, h)))       
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval



