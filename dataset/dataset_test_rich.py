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

from util import MaskAtom

from util import scaffold_split, balanced_scaffold_split
from util import read_smiles
from util import get_fragments

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger

from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


RDLogger.DisableLog('rdApp.*')  


from rdkit.Chem.Scaffolds import MurckoScaffold

from tqdm import tqdm as core_tqdm
from typing import List, Set, Tuple, Union, Dict
from collections import defaultdict


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

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),

    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],


    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

# rich_feature로 사용할 feature
ATOM_FEATURES = {
    'atomic_num' : list(range(1, 119)),
    'degree' : [0,1,2,3,4,5],
    'formal_charge' : [-1, -2, 1, 2, 0],
    'chiral_tag' : [0,1,2,3],
    'num_Hs' : [0,1,2,3,4],
    'hybridization': [

        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def onek_encoding_unk(value : int, choices: List[int]) -> List[int]:
    """
        Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.

    """
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
hydrogen_acceptor = Chem.MolFromSmarts(
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
    "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
basic = Chem.MolFromSmarts(
    "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")
        

class MolTestDatasetRich(InMemoryDataset):
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

            
        ring_info = mol.GetRingInfo()
        hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(acidic), ())
        basic_match = sum(mol.GetSubstructMatches(basic), ())

        # 14 종류의  피처가 사용됨 
        # [118,6,5,4,5
        # ,5,1,1,7,1,
        #  1,1,1,6]
        
        atom_features_list = []


        for atom in mol.GetAtoms():
            features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
                        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
                        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
                        onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
                        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
                        onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
                        [1 if atom.GetIsAromatic() else 0] + \
                        [atom.GetMass() * 0.01]
            
            atom_idx = atom.GetIdx()
            features = features + \
                        onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                    [atom_idx in hydrogen_acceptor_match] + \
                    [atom_idx in hydrogen_donor_match] + \
                    [atom_idx in acidic_match] + \
                    [atom_idx in basic_match] + \
                    [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                        ring_info.IsAtomInRingOfSize(atom_idx, 4),
                        ring_info.IsAtomInRingOfSize(atom_idx, 5),
                        ring_info.IsAtomInRingOfSize(atom_idx, 6),
                        ring_info.IsAtomInRingOfSize(atom_idx, 7),
                        ring_info.IsAtomInRingOfSize(atom_idx, 8)]
            
        atom_features_list.append(features)


        normalized_2d_generator = rdNormalizedDescriptors.RDKit2DNormalized()
        x_add = normalized_2d_generator.process(self.smiles_data[index])

        # print('atomic_num', len(onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num'])))
        # print('degree', len(onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree'])))
        # print('charge',len(onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) ))
        # print('chiral_tag',len(onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])))
        # print('numHs',len(onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) ))
        # print('hydrization',len(onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) ))
        # print('aromatic', len([1 if atom.GetIsAromatic() else 0]  ))
        # print('getMass', len([atom.GetMass()]))
        # print('GetImplicitValence',len(onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) ))
        # print('hydrogen_accep',len([atom_idx in hydrogen_acceptor_match] ))
        # print('hydro_donor',len([atom_idx in hydrogen_donor_match] ))
        # print('acid_match',len([atom_idx in acidic_match] ))
        # print('basic_match',len([atom_idx in basic_match] ))
        # print('Isring' ,len([ring_info.IsAtomInRingOfSize(atom_idx, 3),
        #             ring_info.IsAtomInRingOfSize(atom_idx, 4),
        #             ring_info.IsAtomInRingOfSize(atom_idx, 5),
        #             ring_info.IsAtomInRingOfSize(atom_idx, 6),
        #             ring_info.IsAtomInRingOfSize(atom_idx, 7),
        #             ring_info.IsAtomInRingOfSize(atom_idx, 8)]))
        # print('\n')
        

        row, col, edge_feat = [], [], []

        x = torch.tensor(np.array(atom_features_list) ).type(torch.FloatTensor)

        
        num_bond_features = 2  # bond type, bond direction

        if len(mol.GetBonds()) > 0:
            edge_indices, edge_attrs = [], []

            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]

                bt = bond.GetBondType()
                edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
                edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

                # edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] \
                #     + [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())] \
                #     + [(bond.GetIsConjugated() if bt is not None else 0) ] \
                #     + [(bond.IsInRing() if bt is not None else 0)] 
                
                # edge_feature +=  onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

                # print( 'possible_bond',allowable_features['possible_bonds'].index(bond.GetBondType()))
                # print( 'possible_dir',allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))
                # print( 'is_conjugated',len([(bond.GetIsConjugated() if bt is not None else 0)]))
                # print( 'is_ring',len([(bond.IsInRing() if bt is not None else 0)] ))
                # print( 'get_streo',len( onek_encoding_unk(int(bond.GetStereo()), list(range(6)))))
                  
                # edge_attrs += [edge_feature, edge_feature]


            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        else: 
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
            
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)

        if x_add is None:
            # x_add가 None인 경우, 처리 방식 결정
            # 예: 빈 특성 리스트 또는 기본값 설정
            print(f"Warning: No features generated for SMILES: {self.smiles_data[index]}")
            x_add = [] # 예시 기본값
        else:
            # x_add를 텐서로 변환

            x_add = torch.tensor(np.array(x_add[1:]), dtype=torch.long).view(1, -1)
        

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, x_add = x_add)

        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapperRich(object):
    
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

        train_dataset = MolTestDatasetRich(data_path=self.data_path, target=self.target, task=self.task)
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
