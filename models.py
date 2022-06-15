from numpy import log, multiply
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import gnn_gate
from gnn_concrete_dropout import GNN_ConcreteDropout
from fc_concrete_dropout import FC_ConcreteDropout
N_atom_features=30
class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate
        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([gnn_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)])        
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1]+3, 512) if i==0 else
                                 nn.Linear(256, 128) if i==n_FC_layer-1 else
                                 nn.Linear(512, 256) for i in range(n_FC_layer)])
        self.linear_mu = nn.Linear(128, 1)
        self.linear_logvar = nn.Linear(128, 1)
        self.embede = nn.Linear(N_atom_features, d_graph_layer, bias = False)
        self.GNN_ConDrop = GNN_ConcreteDropout() 
        self.FC_ConDrop = FC_ConcreteDropout()
    def embede_graph(self, data):
        c_hs1,c_hs2, c_adjs1,c_adjs2,D1,D2 = data
        c_hs1 = self.embede(c_hs1) 
        c_hs2 = self.embede(c_hs2) 


        n1=c_hs1.shape[1]
        n2=c_hs2.shape[1]
        bn1 = nn.BatchNorm1d(n1)
        bn2 = nn.BatchNorm1d(n2)
        bn1.cuda()
        bn2.cuda()  


        for k in range(len(self.gconv1)):

            c_hs1 = self.GNN_ConDrop(c_hs1, c_adjs1, self.gconv1[k])
            c_hs2 = self.GNN_ConDrop(c_hs2, c_adjs2, self.gconv1[k])
            c_hs1 = bn1(c_hs1)
            c_hs2 = bn2(c_hs2)
        c_hs1=torch.cat((c_hs1,D1),2)
        c_hs2=torch.cat((c_hs2,D2),2)
        c_hs1 = c_hs1.sum(1)
        c_hs2 = c_hs2.sum(1)
        
        return c_hs1, c_hs2
    def fully_connected(self, c_hs1,c_hs2):
        regularization1 = torch.empty(5, device=c_hs1.device)
        n1=c_hs1.shape[0]
        fc_bn = nn.BatchNorm1d(n1)
        fc_bn.cuda()
        for k in range(len(self.FC)):
            if k<len(self.FC):                
                c_hs1, regularization1[k] = self.FC_ConDrop(c_hs1, self.FC[k])
                c_hs1=c_hs1.unsqueeze(0)
                c_hs1=fc_bn(c_hs1)
                c_hs1=c_hs1.squeeze(0)
                c_hs1 = F.leaky_relu(c_hs1)

        mean1, regularization1[3] = self.FC_ConDrop(c_hs1, self.linear_mu)
        log_var1, regularization1[4] = self.FC_ConDrop(c_hs1, self.linear_logvar)
        regularization1 = regularization1.sum()
        regularization2 = torch.empty(5, device=c_hs2.device)
        n2=c_hs2.shape[0]
        fc_bn = nn.BatchNorm1d(n2)
        fc_bn.cuda()
        for k in range(len(self.FC)):
            if k<len(self.FC):
                
                c_hs2, regularization2[k] = self.FC_ConDrop(c_hs2, self.FC[k])
                c_hs2=c_hs2.unsqueeze(0)
                c_hs2=fc_bn(c_hs2)
                c_hs2=c_hs2.squeeze(0)

                c_hs2 = F.leaky_relu(c_hs2)
        mean2, regularization2[3] = self.FC_ConDrop(c_hs2, self.linear_mu)
        log_var2, regularization2[4] = self.FC_ConDrop(c_hs2, self.linear_logvar)
        mean=mean1-mean2
        log_var=log_var1+log_var2
        regularization=regularization1.sum()+regularization2.sum()
        return  mean , log_var , regularization

    def train_model(self, data):
        c_hs1, c_hs2 = self.embede_graph(data)
        c_hs, log_var,regularization = self.fully_connected(c_hs1,c_hs2 )
        c_hs = c_hs.view(-1)

        return c_hs, log_var,regularization
    
    def test_model(self,data):
        c_hs1, c_hs2 = self.embede_graph(data)
        c_hs, log_var,regularization = self.fully_connected(c_hs1,c_hs2 )
        c_hs = c_hs.view(-1)

        return c_hs, log_var,regularization



