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
from torch_geometric.data import Data,  DataLoader

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

from util import scaffold_split, balanced_scaffold_split
from util import read_smiles
from util import MaskAtom

def removeSubgraph(Graph, percent=0.2, max_hop=0):
    assert percent <= 1
    G = Graph.copy()
    max_nodes = int(np.floor(len(G.nodes) * percent))

    removed_nodes = set()
    root_nodes = []  # List to store the starting points for each operation

    if max_hop == 0:
        # 0-hop: Randomly remove nodes up to max_nodes
        all_nodes = list(G.nodes())
        random.shuffle(all_nodes)
        for node in all_nodes:
            if len(removed_nodes) < max_nodes:
                removed_nodes.add(node)
                G.remove_node(node)
            else:
                break
    else:
        # >0-hop: Remove nodes in hops from random start nodes until max_nodes is reached
        while len(removed_nodes) < max_nodes:
            # Find a starting node that hasn't been removed
            remaining_nodes = list(set(G.nodes()) - removed_nodes)
            if not remaining_nodes:
                break
            start_node = random.choice(remaining_nodes)
            root_nodes.append(start_node)  # Keep track of this start node
            queue = [(start_node, 0)]
            visited = set([start_node])

            while queue and len(removed_nodes) < max_nodes:
                current_node, current_hop = queue.pop(0)
                if current_node in G.nodes():  # Ensure the node is still in the graph
                    if current_hop <= max_hop:
                        # First, list all neighbors before removing the node
                        neighbors = list(G.neighbors(current_node)) if current_hop < max_hop else []
                        # Remove the node
                        removed_nodes.add(current_node)
                        G.remove_node(current_node)
                        # Now, process neighbors
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, current_hop + 1))

    removed_edges = [(u, v) for u, v in Graph.edges() if u in removed_nodes and v in removed_nodes]

    return list(removed_nodes), removed_edges


class MolTestDataset(InMemoryDataset):
    def __init__(self, data_path, target, task, mask_rate, transform=None, pre_transform=None, pre_filter=None):
        super(InMemoryDataset, self).__init__(None, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        self.mask_rate = mask_rate
        

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

        bonds = mol.GetBonds()

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        removed_node_index, removed_edge_index = removeSubgraph(molGraph, percent = self.mask_rate,  max_hop = 1)
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

        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, removed_node_index = removed_node_index, removed_edge_index = removed_edge_indices)

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

        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task, mask_rate = self.mask_rate)
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
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )


        return train_loader, valid_loader, test_loader
