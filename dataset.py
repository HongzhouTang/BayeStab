from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle

def get_atom_feature(m):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(utils.atom_feature(m, i, None, None))
    H = np.array(H)
    return H+0        

class Dataset(Dataset):

    def __init__(self, keys, data_dir,ddg_dir,wild_dir):
        self.keys = keys
        self.data_dir = data_dir
        self.ddg_dir = ddg_dir
        self.wild_dir = wild_dir

        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        key = self.keys[index]

        mol_w=Chem.MolFromPDBFile('./'+self.wild_dir+'/'+key+'_wild.pdb')
        mol_m=Chem.MolFromPDBFile('./'+self.data_dir+'/'+key+'_mutation.pdb')
        with open('./'+ self.ddg_dir +'/'+key, 'rb') as f:
            labels = pickle.load(f)

        #mutation information
        n1 = mol_m.GetNumAtoms()
        c1 = mol_m.GetConformers()[0]
        d1 = np.array(c1.GetPositions())  
        adj1 = GetAdjacencyMatrix(mol_m)+np.eye(n1)
        H1 = get_atom_feature(mol_m)


        #wild information
        n2 = mol_w.GetNumAtoms()
        c2 = mol_w.GetConformers()[0]
        d2 = np.array(c2.GetPositions())  
        adj2 = GetAdjacencyMatrix(mol_w)+np.eye(n2)
        H2 = get_atom_feature(mol_w)
        labels=labels
        sample = {'H1': H1,\
                  'H2': H2,\
                  'A1': adj1, \
                  'A2': adj2, \
                  'D1': d1,\
                  'D2': d2,\
                  'labels': labels, \
                  'key': key, \
                  }
        return sample



def collate_fn(batch):
    max_natoms1 = max([len(item['H1']) for item in batch if item is not None])
    max_natoms2 = max([len(item['H2']) for item in batch if item is not None])
    H1 = np.zeros((len(batch), max_natoms1, 30))
    H2 = np.zeros((len(batch), max_natoms2, 30))
    A1 = np.zeros((len(batch), max_natoms1, max_natoms1))
    A2 = np.zeros((len(batch), max_natoms2, max_natoms2))
    D1 = np.zeros((len(batch), max_natoms1, 3))
    D2 = np.zeros((len(batch), max_natoms2, 3))

    keys = [] 
    labels=[]   
    for i in range(len(batch)):
        natom1 = len(batch[i]['H1'])
        natom2 = len(batch[i]['H2'])        
        H1[i,:natom1] = batch[i]['H1']
        H2[i,:natom2] = batch[i]['H2']
        A1[i,:natom1,:natom1] = batch[i]['A1']
        A2[i,:natom2,:natom2] = batch[i]['A2']
        D1[i,:natom1,:natom1] = batch[i]['D1']
        D2[i,:natom2,:natom2] = batch[i]['D2']
        keys.append(batch[i]['key'])
        labels.append(batch[i]['labels'])
    H1 = torch.from_numpy(H1).float()
    H2 = torch.from_numpy(H2).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

    return H1, H2, A1, A2,D1,D2, labels, keys