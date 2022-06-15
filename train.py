
import utils
import time
import argparse
import numpy as np
from dataset import Dataset, collate_fn
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import torch.nn as nn
from torch.utils.data import DataLoader 
from models import gnn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
if __name__ == "__main__":

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print (s)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help= "Random seed", type=int, default=42)   
    parser.add_argument('--epochs', help= "Number of epochs to train", type=int, default = 800)
    parser.add_argument('--weight_decay', help= "Weight decay (L2 loss on parameters)", type=float, default=5e-5)
    parser.add_argument("--ngpu", help= "number of gpu", type=int, default = 0)
    parser.add_argument("--train_keys", help= "train keys", type=str, default='data/S2648_keys/S2648_train_keys.pkl')
    parser.add_argument("--test_keys", help= "test keys", type=str, default='data/S2648_keys/S350_test_keys.pkl') 
    parser.add_argument("--data_fpath", help= "file path of data", type=str, default='data/S2648_mutation_pdb')
    parser.add_argument("--ddg_fpath", help="file path of ddg",type=str,default='data/S2648_ddg')
    parser.add_argument("--wild_pdb", help="file path of wild_pdb",type=str,default='data/S2648_wild_pdb/')
    parser.add_argument("--batch_size", help= "batch_size", type=int, default = 32)
    parser.add_argument("--num_workers", help= "number of workers", type=int, default = 0)  
    parser.add_argument("--lr", help= "learning rate", type=float, default = 0.0001) 
    parser.add_argument("--n_graph_layer", help= "number of GNN layer", type=int, default = 4)  
    parser.add_argument("--d_graph_layer", help= "dimension of GNN layer", type=int, default =1140) 
    parser.add_argument("--n_FC_layer", help= "number of FC layer", type=int, default = 3)
    parser.add_argument("--d_FC_layer", help= "dimension of FC layer", type=int, default =1129)

    parser.add_argument("--dropout_rate", help= "dropout_rate", type=float, default = 0.3)
    args = parser.parse_args()
    torch.cuda.empty_cache()

    lr = args.lr
    batch_size = args.batch_size
    dude_data_fpath = args.data_fpath
    tm_data_fpath=args.ddg_fpath
    wild_pdb_fpath=args.wild_pdb

    model = gnn(args)
    with open (args.train_keys, 'rb') as fp:
            train_keys = pickle.load(fp)

    with open (args.test_keys, 'rb') as fp:
            test_keys = pickle.load(fp)
    train_dataset = Dataset(train_keys, args.data_fpath, args.ddg_fpath,args.wild_pdb)
    test_dataset  = Dataset(test_keys, args.data_fpath, args.ddg_fpath,args.wild_pdb)                  
    train_dataloader = DataLoader(train_dataset, args.batch_size, \
        shuffle=True, num_workers = args.num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
        shuffle=True, num_workers = args.num_workers, collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.initialize_model(model, device)
    def heteroscedastic_loss(true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)

        
    loss_fn= heteroscedastic_loss
    # loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    for epochs in range(800):
        list1_test = []
        list2_test = []
        list1_train = []
        list2_train = []    
        train_losses = []
        test_losses = []
        list_log_var=[]
        st = time.time()
        model.train()    
        
        for i_batch, sample in enumerate(train_dataloader):
            model.zero_grad()
            H1,H2 , A1, A2,D1,D2, labels, key = sample
            labels = torch.Tensor(labels)
            H1,H2,A1,A2,D1,D2,labels=H1.to(device),H2.to(device),A1.to(device),A2.to(device),D1.to(device),D2.to(device),labels.to(device)
            pred, log_var, regulation  = model.train_model((H1,H2, A1,A2,D1,D2))
            # loss = loss_fn(pred, labels)
            loss = loss_fn(labels, pred, log_var) + regulation
            loss.backward()
            optimizer.step()
            train_losses.append(loss.data.cpu().numpy())
            pred=pred.data.cpu().numpy()
            labels=labels.data.cpu().numpy()
            list1_train=np.append(list1_train,labels)
            list2_train=np.append(list2_train,pred)
        model.eval()


        for i_batch, sample in enumerate(test_dataloader):
            model.zero_grad()
            H1,H2 , A1, A2,D1,D2, labels, key = sample
            labels = torch.Tensor(labels)
            H1,H2,A1,A2,D1,D2,labels=H1.to(device),H2.to(device),A1.to(device),A2.to(device),D1.to(device),D2.to(device),labels.to(device)
            pred, log_var, regulation = model.test_model((H1,H2, A1,A2,D1,D2))
            loss = loss_fn(labels, pred, log_var) + regulation
            # loss = loss_fn(pred, labels)
            test_losses.append(loss.data.cpu().numpy())
            labels=labels.data.cpu().numpy()
            pred=pred.data.cpu().numpy()
            list1_test=np.append(list1_test,labels)
            list2_test=np.append(list2_test,pred)
            acc=pred/labels

        
        et=time.time()
        
        rp_train = np.corrcoef(list2_train, list1_train)[0,1]
        rp_test = np.corrcoef(list2_test, list1_test)[0,1]
        test_losses = np.mean(np.array(test_losses))
        train_losses = np.mean(np.array(train_losses))
        x = np.array(list1_test).reshape(-1,1)
        y = np.array(list2_test).reshape(-1,1)
        end = time.time()
        rmse=np.sqrt(((y - x) ** 2).mean())

        print('epochs  train_losses   test_losses     pcc_train        pcc_test        rmse        time ')
        print ("%s  \t%.3f   \t%.3f   \t%.3f   \t%.3f   \t%.3f   \t%.3f" 
        %(epochs, train_losses,     test_losses,     rp_train,     rp_test,     rmse  ,  et-st ))
        if epochs==799:
            torch.save(model,"Ssym_model.pkl") 

            # dataframe_test = pd.DataFrame({'labels':list1_test,'predict':list2_test})
            # dataframe_test.to_csv(r"./S2648_test_plt_5e-5_1223_41120_41024_loss.csv",sep=',')




























