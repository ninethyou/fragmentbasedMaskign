from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

import copy
import random
import networkx as nx
import numpy as np
from torch_geometric.utils import convert
from loader import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple
from rdkit import Chem
from rdkit.Chem import AllChem
from loader import mol_to_graph_data_obj_simple, \
    graph_data_obj_to_mol_simple

from loader import MoleculeDataset

import csv
import math

from typing import Callable, Optional, Union
import torch_scatter
from torch import Tensor
import torch.nn as nn
from itertools import compress


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
from tqdm import tqdm as core_tqdm

from typing import List, Set, Tuple, Union, Dict


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)
    mol2 = AllChem.MolFromSmiles(s2)
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        # 현재 엣지들을 문자열 집합으로 
        edge_set = set([str(data.edge_index[0, i].cpu().item()) + "," + str(
            data.edge_index[1, i].cpu().item()) for i in
                        range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data


class MaskEdge:
    def __init__(self, mask_rate):
        self.mask_rate = mask_rate

    def __call__(self, data, mask_rate=0.25):

        # data.masked_edge_attr = data.edge_attr.clone()

        num_nodes = data.num_nodes
        num_edges = data.num_edges

        num_mask_edges = max([0, data.edge_attr.size(0) // 2 - math.ceil((1.0-mask_rate) * num_edges)])
        mask_edges_i_single = random.sample(list(range(data.edge_attr.size(0)//2)), num_mask_edges)

        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]


        # data.mask_edge = mask_edges_i


        # 현재 엣지들을 문자열 집합으로 
        edge_set = set([str(data.edge_index[0, i].cpu().item()) + "," + str(
            data.edge_index[1, i].cpu().item()) for i in
                        range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data



class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


# TODO(Bowen): more unittests
class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = max(int(num_atoms * self.mask_rate),1)
            
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

# TODO(Bowen): more unittests
class MaskAtomTest:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            # sample_size = int(num_atoms * self.mask_rate + 1)

            sample_size = max(int(num_atoms * self.mask_rate),1)

            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)
        data.masked_x = copy.deepcopy(data.x)


        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0].long(), num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1].long(), num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            data.masked_edge_attr = copy.deepcopy(data.edge_attr)
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.masked_edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        # 추가된 부분 
        # create mask edge labels by copying edge features of edges that are bonded to
        # mask atoms
        # connected_edge_indices = []
        # for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
        #     for atom_idx in masked_atom_indices:
        #         if atom_idx in set((u, v)) and \
        #             bond_idx not in connected_edge_indices:
        #             connected_edge_indices.append(bond_idx)

        # if len(connected_edge_indices) > 0:
        #     # create mask edge labels by copying bond features of the bonds connected to
        #     # the mask atoms

        #     # data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
        #     # # modify the original bond features of the bonds connected to the mask atoms
        #     for bond_idx in connected_edge_indices:
        #         data.edge_attr[bond_idx] = torch.tensor(
        #             [self.num_edge_type, 0])
                    
        # data.connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


# TODO(Bowen): more unittests

class MaskSubGraph:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """


        # networkx 그래프 생성
        molGraph = nx.Graph()
        
        # 노드 개수는 data.x의 행 개수
        num_nodes = data.x.size(0)


        # edge_index를 올바른 형식으로 변환
        edges = data.edge_index.t().tolist()
        molGraph.add_edges_from(edges)
        # 노드 추가 (num_nodes 개수만큼 노드 추가)
        molGraph.add_nodes_from(range(num_nodes))

        N = len(molGraph.nodes)

        if N > 0:
            start_i = random.sample(list(range(N)), 1)[0]
        else:
            print(data)
            print(N, num_nodes)

            raise ValueError("The population size (N) must be greater than 0.")

        removed_node_index, removed_edge_index = removeSubgraph(molGraph, start_i, self.mask_rate)
        removed_node_index = torch.tensor(removed_node_index, dtype=torch.long)
        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in removed_node_index:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = removed_node_index
        data.masked_x = copy.deepcopy(data.x)


        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in data.masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            removed_edge_indices = []
            for edge in removed_edge_index:
                u, v = edge
                # 양방향 엣지 인덱스 찾기
                index_uv = ((data.edge_index[0] == u) & (data.edge_index[1] == v)).nonzero(as_tuple=True)[0]
                index_vu = ((data.edge_index[0] == v) & (data.edge_index[1] == u)).nonzero(as_tuple=True)[0]
                if index_uv.numel() > 0:  # (u, v) 방향의 인덱스가 존재하면 리스트에 추가
                    removed_edge_indices.extend(index_uv.tolist())
                if index_vu.numel() > 0:  # (v, u) 방향의 인덱스가 존재하면 리스트에 추가
                    removed_edge_indices.extend(index_vu.tolist())

            removed_edge_indices = torch.tensor(removed_edge_indices, dtype=torch.long)

            connected_edge_indices = removed_edge_index

                
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2], dtype=torch.long)

            else:
                connected_edge_indices = torch.tensor(
                    connected_edge_indices, dtype=torch.long)


            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)


            data.connected_edge_indices = connected_edge_indices


        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)
    



class MaskAdjacent:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """


        # networkx 그래프 생성
        molGraph = nx.Graph()
        
        # 노드 개수는 data.x의 행 개수
        num_nodes = data.x.size(0)


        # edge_index를 올바른 형식으로 변환
        edges = data.edge_index.t().tolist()
        molGraph.add_edges_from(edges)
        # 노드 추가 (num_nodes 개수만큼 노드 추가)
        molGraph.add_nodes_from(range(num_nodes))

        N = len(molGraph.nodes)

        if N > 0:
            start_i = random.sample(list(range(N)), 1)[0]
        else:
            print(data)
            print(N, num_nodes)

            raise ValueError("The population size (N) must be greater than 0.")

        removed_node_index, removed_edge_index = removeAdjacent(molGraph, percent=self.mask_rate, max_hop=1)
        removed_node_index = torch.tensor(removed_node_index, dtype=torch.long)
        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in removed_node_index:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = removed_node_index
        data.masked_x = copy.deepcopy(data.x)


        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in data.masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            removed_edge_indices = []
            for edge in removed_edge_index:
                u, v = edge
                # 양방향 엣지 인덱스 찾기
                index_uv = ((data.edge_index[0] == u) & (data.edge_index[1] == v)).nonzero(as_tuple=True)[0]
                index_vu = ((data.edge_index[0] == v) & (data.edge_index[1] == u)).nonzero(as_tuple=True)[0]
                if index_uv.numel() > 0:  # (u, v) 방향의 인덱스가 존재하면 리스트에 추가
                    removed_edge_indices.extend(index_uv.tolist())
                if index_vu.numel() > 0:  # (v, u) 방향의 인덱스가 존재하면 리스트에 추가
                    removed_edge_indices.extend(index_vu.tolist())

            removed_edge_indices = torch.tensor(removed_edge_indices, dtype=torch.long)

            connected_edge_indices = removed_edge_index

                
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2], dtype=torch.long)

            else:
                connected_edge_indices = torch.tensor(
                    connected_edge_indices, dtype=torch.long)


            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)


            data.connected_edge_indices = connected_edge_indices


        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class MaskFragment:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        # networkx 그래프 생성
        molGraph = nx.Graph()
        
        # 노드 개수는 data.x의 행 개수
        num_nodes = data.x.size(0)


        # edge_index를 올바른 형식으로 변환
        edges = data.edge_index.t().tolist()
        molGraph.add_edges_from(edges)
        # 노드 추가 (num_nodes 개수만큼 노드 추가)
        molGraph.add_nodes_from(range(num_nodes))

        N = len(molGraph.nodes)

        if N > 0:
            start_i = random.sample(list(range(N)), 1)[0]
        else:
            print(data)
            print(N, num_nodes)

            raise ValueError("The population size (N) must be greater than 0.")
        
        num = max(int(np.floor(len(molGraph.nodes) * self.mask_rate)), 1)

        removed_node_index, removed_edge_index = remove_by_motif_strict(molGraph, data.motif_batch, num)
        removed_node_index = torch.tensor(removed_node_index, dtype=torch.long).sort()[0]
        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in removed_node_index:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        


        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = removed_node_index
        data.masked_x = copy.deepcopy(data.x)


        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in data.masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            removed_edge_indices = []
            for edge in removed_edge_index:
                u, v = edge
                # 양방향 엣지 인덱스 찾기
                index_uv = ((data.edge_index[0] == u) & (data.edge_index[1] == v)).nonzero(as_tuple=True)[0]
                index_vu = ((data.edge_index[0] == v) & (data.edge_index[1] == u)).nonzero(as_tuple=True)[0]
                if index_uv.numel() > 0:  # (u, v) 방향의 인덱스가 존재하면 리스트에 추가
                    removed_edge_indices.extend(index_uv.tolist())
                if index_vu.numel() > 0:  # (v, u) 방향의 인덱스가 존재하면 리스트에 추가
                    removed_edge_indices.extend(index_vu.tolist())

            removed_edge_indices = torch.tensor(removed_edge_indices, dtype=torch.long)

            connected_edge_indices = removed_edge_index

                
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2], dtype=torch.long)

            else:
                connected_edge_indices = torch.tensor(
                    connected_edge_indices, dtype=torch.long)


            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)


            data.connected_edge_indices = connected_edge_indices


        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


def remove_by_motif_strict(Graph, motif_batch, num_to_delete):
    G = Graph.copy()
    total_nodes = len(G.nodes())
    motif_nodes = {i: np.where(motif_batch == i)[0].tolist() for i in np.unique(motif_batch)}
    removed_nodes = set()

    while len(removed_nodes) < num_to_delete:
        if len(G.nodes()) <= (total_nodes - num_to_delete):
            break  # Stop if the graph has fewer nodes left than needed to delete.

        available_motifs = [m for m in motif_nodes if len(set(motif_nodes[m]) - removed_nodes) > 0]
        if not available_motifs:
            break

        chosen_motif = random.choice(available_motifs)
        nodes_to_remove = set(motif_nodes[chosen_motif]) - removed_nodes

        # Check node existence in the graph before removal
        nodes_to_remove = {node for node in nodes_to_remove if node in G.nodes()}

        # Adjust the nodes to remove to not exceed num_to_delete
        if len(removed_nodes) + len(nodes_to_remove) > num_to_delete:
            excess = (len(removed_nodes) + len(nodes_to_remove)) - num_to_delete
            nodes_to_remove = set(random.sample(nodes_to_remove, len(nodes_to_remove) - excess))

        # Remove nodes from the graph and update the removed nodes set
        G.remove_nodes_from(nodes_to_remove)
        removed_nodes.update(nodes_to_remove)


    removed_edges = [(u, v) for u, v in Graph.edges() if u in removed_nodes or v in removed_nodes]

    return list(removed_nodes), removed_edges








def removeAdjacent(Graph, percent=0.25, max_hop=1):
    G = Graph.copy()
    max_nodes = max(int(np.floor(len(G.nodes) * percent)),1)

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


# TODO(Bowen): more unittests
class MaskAtomTestRich:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        self.num_chirality_tag = 3
        self.num_bond_direction = 3 


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = max(int(num_atoms * self.mask_rate),1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)
        data.masked_x = copy.deepcopy(data.x)


        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0].long(), num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1].long(), num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # # modify the original node feature of the masked node
        # for atom_idx in masked_atom_indices:
        #     data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        data.masked_x[masked_atom_indices] = torch.zeros((len(masked_atom_indices), 42))
        data.masked_x[masked_atom_indices, 0] = self.num_atom_type


        if self.mask_edge:
            data.masked_edge_attr = copy.deepcopy(data.edge_attr)
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms

                for bond_idx in connected_edge_indices:
                    data.masked_edge_attr[bond_idx] = torch.zeros(10)
                    data.masked_edge_attr[bond_idx, 0] = self.num_edge_type


                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)




def removeSubgraph(Graph, center, percent=0.25):
    assert percent <= 1
    G = Graph.copy()
    num = max(int(np.floor(len(G.nodes) * percent)), 1)


    removed_nodes = []
    temp = [center]

    while len(removed_nodes) < num:
        if not temp:  # temp가 비어있으면 더 이상 제거할 노드가 없음
            break
        neighbors = []
        for n in temp:
            if n in G:  # 노드가 G에 존재하는지 확인
                neighbors.extend([i for i in G.neighbors(n) if i not in temp and i not in removed_nodes])
                if len(removed_nodes) < num:
                    G.remove_node(n)
                    removed_nodes.append(n)
                else:
                    break
        temp = list(set(neighbors))

    # 제거된 노드들 사이에 있는 엣지만 찾기
    removed_edges = [(u, v) for u, v in Graph.edges() if u in removed_nodes and v in removed_nodes]
    
    return removed_nodes, removed_edges




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

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()

def read_smiles(data_path, target, task):
    smiles_data, labels = [], []

    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row.get('smiles', row.get('SMILES'))
                label = [row[t] for t in target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and all(l != '' for l in label):
                    smiles_data.append(smiles)
                    processed_labels = []
                    for l in label:
                        
                        if task == 'classification':
                            processed_labels.append(int(l))
                        elif task == 'regression':
                            processed_labels.append(float(l))
                        else:
                            ValueError('task must be either regression or classification')
                    processed_labels = [ -1 if x == 0 else x for x in processed_labels]
                    labels.append(processed_labels)

    
    return smiles_data, labels

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

class tqdm(core_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

def balanced_scaffold_split(dataset, smiles_list, 
                            task_idx=None, null_value=0,
                            frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                            seed = 0, return_smiles=False): 
    """
     스캐폴드를 큰것과 작은 것으로 분류

    """

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = [smiles for i, smiles in compress(enumerate(smiles_list), non_null)]

    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = [smiles for i, smiles in compress(enumerate(smiles_list), non_null)]

    # print(smiles_list)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = [x for x in index_sets if len(x) > len(dataset) * (frac_valid / 2) or len(x) > len(dataset) * (frac_test / 2)]
    small_index_sets = [x for x in index_sets if x not in big_index_sets]

    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)

    index_sets = big_index_sets + small_index_sets

    ## 인덱스를 이렇게 나누면 되겠다. 
    train, val, test = [], [], []
    for index_set in index_sets:
        if len(train) + len(index_set) <= len(dataset) * frac_train:
            train.extend(index_set)

        elif len(val) + len(index_set) <= len(dataset) * frac_valid:
            val.extend(index_set)

        else:
            test.extend(index_set)

    return train, val, test




if __name__ == "__main__":
    transform = NegativeEdge()
    dataset = MoleculeDataset("dataset/tox21", dataset="tox21")
    transform(dataset[0])

    """
    # TODO(Bowen): more unit tests
    # test ExtractSubstructureContextPair

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(m)
    root_idx = 13

    # 0 hops: no substructure or context. We just test the absence of x attr
    transform = ExtractSubstructureContextPair(0, 0, 0)
    transform(data, root_idx)
    assert not hasattr(data, 'x_substruct')
    assert not hasattr(data, 'x_context')

    # k > n_nodes, l1 = 0 and l2 > n_nodes: substructure and context same as
    # molecule
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(100000, 0, 100000)
    transform(data, root_idx)
    substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
                                                 data.edge_index_substruct,
                                                 data.edge_attr_substruct)
    context_mol = graph_data_obj_to_mol_simple(data.x_context,
                                               data.edge_index_context,
                                               data.edge_attr_context)
    assert check_same_molecules(AllChem.MolToSmiles(substruct_mol),
                                AllChem.MolToSmiles(context_mol))

    transform = ExtractSubstructureContextPair(1, 1, 10000)
    transform(data, root_idx)

    # increase k from 0, and increase l1 from 1 while keeping l2 > n_nodes: the
    # total number of atoms should be n_atoms
    for i in range(len(m.GetAtoms())):
        data = mol_to_graph_data_obj_simple(m)
        print('i: {}'.format(i))
        transform = ExtractSubstructureContextPair(i, i, 100000)
        transform(data, root_idx)
        if hasattr(data, 'x_substruct'):
            n_substruct_atoms = data.x_substruct.size()[0]
        else:
            n_substruct_atoms = 0
        print('n_substruct_atoms: {}'.format(n_substruct_atoms))
        if hasattr(data, 'x_context'):
            n_context_atoms = data.x_context.size()[0]
        else:
            n_context_atoms = 0
        print('n_context_atoms: {}'.format(n_context_atoms))
        assert n_substruct_atoms + n_context_atoms == len(m.GetAtoms())

    # l1 < k and l2 >= k, so an overlap exists between context and substruct
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(2, 1, 3)
    transform(data, root_idx)
    assert hasattr(data, 'center_substruct_idx')

    # check correct overlap atoms between context and substruct


    # m = AllChem.MolFromSmiles('COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(C)C(OC)=C2C)C=C1')
    # data = mol_to_graph_data_obj_simple(m)
    # root_idx = 9
    # k = 1
    # l1 = 1
    # l2 = 2
    # transform = ExtractSubstructureContextPaidata = mol_to_graph_data_obj_simple(m)r(k, l1, l2)
    # transform(data, root_idx)
    pass

    # TODO(Bowen): more unit tests
    # test MaskAtom
    from loader import mol_to_graph_data_obj_simple, \
        graph_data_obj_to_mol_simple

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    original_data = mol_to_graph_data_obj_simple(m)
    num_atom_type = 118
    num_edge_type = 5

    # manually specify masked atom indices, don't mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=False)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    assert not hasattr(data, 'mask_edge_label')
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()

    # manually specify masked atom indices, mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=True)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()
    # check that the correct rows in edge_attr have been modified to be mask edge
    # type, and the mask_edge_label are correct
    rdkit_bonds = []
    for atom_idx in masked_atom_indices:
        bond_indices = list(AllChem.FindAtomEnvironmentOfRadiusN(m, radius=1,
                                                                 rootedAtAtom=atom_idx))
        for bond_idx in bond_indices:
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetBeginAtomIdx(), m.GetBonds()[
                    bond_idx].GetEndAtomIdx()))
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetEndAtomIdx(), m.GetBonds()[
                    bond_idx].GetBeginAtomIdx()))
    rdkit_bonds = set(rdkit_bonds)
    connected_edge_indices = []
    for i in range(data.edge_index.size()[1]):
        if tuple(data.edge_index.numpy().T[i].tolist()) in rdkit_bonds:
            connected_edge_indices.append(i)
    assert (data.edge_attr[connected_edge_indices] ==
            torch.tensor(([num_edge_type, 0]))).all()
    assert (data.mask_edge_label == original_data.edge_attr[
        connected_edge_indices[::2]]).all() # data.mask_edge_label contains
    # the unique edges (ignoring direction). The data obj has edge ordering
    # such that two directions of a single edge occur in pairs, so to get the
    # unique undirected edge indices, we take every 2nd edge index from list
    """

