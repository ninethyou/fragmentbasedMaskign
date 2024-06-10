import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
# from torch_geometric.data import Data, Dataset, DataLoader
# from torch.utils.data import DataLoader

from torch_geometric.data import Data, DataLoader

from torch_geometric.data import InMemoryDataset

from itertools import compress


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger


RDLogger.DisableLog('rdApp.*')  


from rdkit.Chem.Scaffolds import MurckoScaffold

from tqdm import tqdm as core_tqdm
from typing import List, Set, Tuple, Union, Dict
from collections import defaultdict

from util import scaffold_split, balanced_scaffold_split
from util import read_smiles
from util import get_fragments

import networkx as nx

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def map_fragment_adjacency(mol, frag_indices):
    """
    Maps fragment adjacency based on molecular bonds.
    Returns a dict where keys are fragment indices and values are sets of adjacent fragment indices.
    """
    adjacency_map = {i: set() for i in range(len(frag_indices))}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for i, frag1 in enumerate(frag_indices):
            if a1 in frag1:
                for j, frag2 in enumerate(frag_indices):
                    if a2 in frag2 and i != j:
                        adjacency_map[i].add(j)
                        adjacency_map[j].add(i)
    return adjacency_map

def remove_adjacent_fragments(Graph, center_frag_idx, frag_indices, adjacency_map, percent=0.2):
    G = Graph.copy()
    num = int(np.floor(len(G.nodes) * percent))
    
    removed_nodes = []
    visited_frags = set()  # Set of visited fragments
    
    # First, unconditionally remove the fragment containing the center node
    initial_frag = frag_indices[center_frag_idx]
    for node in initial_frag:
        if node in G:
            G.remove_node(node)
            removed_nodes.append(node)
    visited_frags.add(center_frag_idx)

    # Initialize the queue with adjacent fragments of the initial fragment
    frags_to_remove = [adj_frag for adj_frag in adjacency_map[center_frag_idx] if adj_frag not in visited_frags]

    # Conditionally remove additional fragments, considering the 'num' limit
    while frags_to_remove and len(removed_nodes) < num:
        current_frag_idx = frags_to_remove.pop(0)
        if current_frag_idx in visited_frags:
            continue
        
        current_frag = frag_indices[current_frag_idx]
        # Check if adding this fragment exceeds the 'num' limit
        if len(removed_nodes) + len(current_frag) > num:
            continue  # Skip this fragment if it would exceed the limit
        
        for node in current_frag:
            if node in G:
                G.remove_node(node)
                removed_nodes.append(node)
        
        visited_frags.add(current_frag_idx)
        
        # Add unvisited adjacent fragments to the queue
        for adjacent_frag_idx in adjacency_map[current_frag_idx]:
            if adjacent_frag_idx not in visited_frags:
                frags_to_remove.append(adjacent_frag_idx)
    
    removed_edges = [(u, v) for u, v in Graph.edges() if u in removed_nodes or v in removed_nodes]

    return removed_nodes, removed_edges

class MolTestDatasetFrag(InMemoryDataset):
    def __init__(self, data_path, target, task, transform=None, pre_transform=None, pre_filter=None):
        super(InMemoryDataset, self).__init__(None, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)

        frag_indices, _  = get_fragments(mol)


        

        N = mol.GetNumAtoms()

        bonds = mol.GetBonds()

        # data_list.motif_batch = torch.full( (data_list.x.size(0), ), fill_value = -1, dtype=torch.long)  


        motif_batch = torch.zeros(x.size(0), dtype=torch.long)

        print(frag_indices)


    #   curr_indicator = 0
        curr_indicator = 1
        curr_num = 0

        for  indices in  frag_indices:
            for idx in indices:
                curr_idx = np.array(list(idx)) + curr_num
                if len(curr_idx) > 0 : # 추가한 부분 
                    motif_batch[curr_idx] = curr_indicator
                    curr_indicator += 1
            curr_num += N

        # Identify nodes with 0 fragment indicator and assign new indices
        zero_nodes = (motif_batch == 0).nonzero(as_tuple=True)[0]

        if len(zero_nodes) > 0:

            # Assign new indices to nodes with 0 fragment indicator
            new_indicator_start = motif_batch.max().item() + 1
            new_indicators = torch.arange(new_indicator_start, new_indicator_start + len(zero_nodes), dtype=torch.long)

            motif_batch[zero_nodes] = new_indicators

        motif_batch -= 1

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        center = random.sample(list(range(N)), 1)[0]

        center_frag_idx = next(i for i, frag in enumerate(frag_indices) if center in frag)  # Find center fragment index
        adjacency_map = map_fragment_adjacency(mol, frag_indices)  # Map fragment adjacency

        removed_node_index, removed_edge_index = remove_adjacent_fragments(molGraph, center_frag_idx, frag_indices, adjacency_map, percent=0.2)
        removed_node_index = torch.tensor(removed_node_index, dtype=torch.long)


        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        removed_edge_indices = []
        for edge in removed_edge_index:
            u, v = edge
            # 양방향 엣지 인덱스 찾기
            index_uv = ((edge_index[0] == u) & (edge_index[1] == v)).nonzero(as_tuple=True)[0]
            index_vu = ((edge_index[0] == v) & (edge_index[1] == u)).nonzero(as_tuple=True)[0]
            if index_uv.numel() > 0:  # (u, v) 방향의 인덱스가 존재하면 리스트에 추가
                removed_edge_indices.extend(index_uv.tolist())
            if index_vu.numel() > 0:  # (v, u) 방향의 인덱스가 존재하면 리스트에 추가
                removed_edge_indices.extend(index_vu.tolist())

        removed_edge_indices = torch.tensor(removed_edge_indices, dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, removed_node_index = removed_node_index, removed_edge_index = removed_edge_indices)

        # return data, N, frag_indices
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting,seed, random_masking = 0, mask_rate = 0.2, mask_edge = 0
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.seed = seed
        self.random_masking = random_masking
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        assert splitting in ['random', 'scaffold', 'balanced_scaffold']

    def get_data_loaders(self):

        train_dataset = MolTestDatasetFrag(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size, seed=self.seed)

            # define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_sampler = SubsetRandomSampler(test_idx)

        elif self.splitting == 'balanced_scaffold':
            train_idx, valid_idx, test_idx  = balanced_scaffold_split(
                train_dataset, train_dataset.smiles_data,
                frac_train=1 - self.valid_size - self.test_size, frac_valid=self.valid_size, frac_test=self.test_size,
                seed=self.seed
            )

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False ,)
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False,)       
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False, )
        
        #collate 없앰 
        return train_loader, valid_loader, test_loader


def collate_frag(batch):

    data_list,  atom_nums, frag_indices = zip(*batch)

    # frag_mols = [j for i in frag_mols for j in i]
    
    data_list = Batch.from_data_list(data_list)
    
    # data_list.motif_batch = torch.full( (data_list.x.size(0), ), fill_value = -1, dtype=torch.long)  


    data_list.motif_batch = torch.zeros(data_list.x.size(0), dtype=torch.long)
    # print(data_list.motif_batch.shape)

#   curr_indicator = 0
    curr_indicator = 1
    curr_num = 0
    for N, indices in zip(atom_nums, frag_indices):
        for idx in indices:
            curr_idx = np.array(list(idx)) + curr_num
            if len(curr_idx) > 0 : # 추가한 부분 
                data_list.motif_batch[curr_idx] = curr_indicator
                curr_indicator += 1
        curr_num += N

    # Identify nodes with 0 fragment indicator and assign new indices
    zero_nodes = (data_list.motif_batch == 0).nonzero(as_tuple=True)[0]

    if len(zero_nodes) > 0:

        # Assign new indices to nodes with 0 fragment indicator
        new_indicator_start = data_list.motif_batch.max().item() + 1
        new_indicators = torch.arange(new_indicator_start, new_indicator_start + len(zero_nodes), dtype=torch.long)

        data_list.motif_batch[zero_nodes] = new_indicators

    data_list.motif_batch -= 1

    return data_list


# 사용자 정의 collate 함수
def collate_fn(data_list):
    return Batch.from_data_list(data_list)