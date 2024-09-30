from functools import partial

import argparse
import copy
import wandb

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate

from dataloader import DataLoaderMasking, DataLoaderMaskingPred, DataLoaderMaskingPredTest#, DataListLoader
from batch import BatchMasking

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from tqdm import tqdm
from tqdm.auto import tqdm

import numpy as np
from datetime import datetime

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdchem
from IPython.display import SVG


from splitters import scaffold_split, random_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from dataset_pretrain import MoleculeDataset    

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import io
from PIL import Image

from util import MaskAtomTest

import networkx as nx


full_path = __file__

savefilename = os.path.basename(full_path)


def get_element_symbol(atomic_number):
    return rdchem.GetPeriodicTable().GetElementSymbol(atomic_number)

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def compute_accuracy(pred, target):
    # print(pred.shape, target.shape)
    # return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

        _, predicted_classes = torch.max(pred, dim=1) # dim=1은 노드별로 최댓값을 찾음

        correct_predictions = (predicted_classes == target)

        accuracy = correct_predictions.float().mean().item()

        return accuracy

def calc_acc(pred, label):
            # edge prediction   
        _, predicted_classes = torch.max(pred, dim=1) # dim=1은 노드별로 최댓값을 찾음

        correct_predictions = (predicted_classes == label)

        accuracy = correct_predictions.float().mean().item()

        return accuracy

# def plot_class_frequencies_filename(pred, label, fileName,args):

#     accuracy = calc_acc(pred, label)

#     _, predicted_classes = torch.max(pred, dim=1) # dim=1은 노드별로 최댓값을 찾음

#     # Convert tensors to numpy arrays for plotting
#     predicted_classes = predicted_classes.detach().cpu().numpy()
#     label = label.detach().cpu().numpy()
    

#     # Get unique classes and their counts
#     classes = np.unique(np.concatenate((predicted_classes, label)))
#     pred_counts = [np.sum(predicted_classes == cls) for cls in classes]
#     label_counts = [np.sum(label == cls) for cls in classes]
#         # 비율 계산 및 표시
#     total_pred = sum(pred_counts)
#     total_labels = sum(label_counts)


#     # Set up the figure and axis
#     fig, ax = plt.subplots(figsize=(8, 5))
    
#     # Set the bar width
#     bar_width = 0.35
#     # Set the index for groups
#     index = np.arange(len(classes))

#     bar1 = ax.bar(index, pred_counts, bar_width, label='Predicted')
#     bar2 = ax.bar(index + bar_width, label_counts, bar_width, label='Actual')
    
#     ax.set_xlabel('Classes')
#     ax.set_ylabel('Frequency')
#     ax.set_title('Predicted vs Actual Class Frequencies')
#     ax.set_xticks(index + bar_width / 2)
#     ax.set_xticklabels(classes)
#     ax.legend()
    
#     for bar, count in zip(bar1, pred_counts):
#         percent = 100 * count / total_pred
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percent:.1f}%', ha='center', va='bottom')
    
#     for bar, count in zip(bar2, label_counts):
#         percent = 100 * count / total_labels
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percent:.1f}%', ha='center', va='bottom')
    
#     accuracy_text = f'Accuracy: {accuracy:.2f}'
#     ax.text(0.15, 0.95, accuracy_text, transform=ax.transAxes, fontsize=12, 
#             verticalalignment='top', horizontalalignment='center')

#     # Save the plot to a file
#     plt.tight_layout()

#     plt.savefig(fileName)  # Saves the plot as a PNG file
#     plt.close()


# def plot_molecule(batch, pred_node, masked_node_indices, bn, save_dir):
        
#             drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
#             opts = drawer.drawOptions()
#             opts.padding = 0.2
#             opts.legendFontSize = 16
#             opts.atomLabelFontSize = 18
#             opts.useBWAtomPalette()
            
#             Ns = [mol.GetNumAtoms() for mol in batch.mols]
#             start_indices = np.cumsum([0] + Ns[:-1])
            
            
#             splitted_topNodes = []
#             masked_node_indices_set = set(masked_node_indices.cpu().numpy())
    
#             for start_idx, num_atoms in zip(start_indices, Ns):
#                 mol_indices = set(range(start_idx, start_idx + num_atoms))
#                 mol_topNodes = [int(idx - start_idx) for idx in masked_node_indices_set if idx in mol_indices]            
    
#                 splitted_topNodes.append(mol_topNodes)
    
            
#             _, predicted_classes = torch.max(pred_node[masked_node_indices], dim=1)
    
#             start_idx = 0
            
#             for i, (mol, nodes) in enumerate(zip(batch.mols, splitted_topNodes)):
#                 drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)  # Cairo 버전 사용
#                 opts = drawer.drawOptions()
#                 opts.padding = 0.2
#                 opts.legendFontSize = 16
#                 opts.atomLabelFontSize = 18
#                 opts.useBWAtomPalette()

#                 correct = 0
#                 for idx, node in enumerate(nodes):
#                         atomic_num = mol.GetAtomWithIdx(node).GetAtomicNum()
#                         element_symbol = get_element_symbol(atomic_num)\
                        
#                         pred_value = predicted_classes[start_idx].item()
#                         start_idx = start_idx + 1
#                         pred_symbol = get_element_symbol(pred_value) if pred_value <= 118 else "?"
#                         opts.atomLabels[node] = f"{element_symbol}\npred: {pred_symbol}"
#                         if element_symbol == pred_symbol: correct = 1

#                 drawer.DrawMolecule(mol, highlightAtoms=nodes)
#                 drawer.FinishDrawing()
#                 img_data = drawer.GetDrawingText()


#                 # Save SVG or convert to PNG
#                 file_name_base= f"molecule_{bn}_{i}"
#                 if correct:
#                     fn = '_correct.png'
#                 else:
#                     fn = '_false.png'
                    
#                 save_path = os.path.join(save_dir, file_name_base+fn)
            

#                 # Convert to image and save
#                 image = Image.open(io.BytesIO(img_data))
#                 image.save(save_path)

# def plot_total_elements_count(total_element_counts_mask, fileName):
    
#     elements = list(total_element_counts_mask.keys())
#     frequencies = list(total_element_counts_mask.values())
#     total_frequencies = sum(frequencies)

    
#     plt.figure(figsize=(12, 6))  # 그래프 크기 설정
#     bars = plt.bar(elements, frequencies, color='skyblue')  # 막대 그래프 생성
    
#     plt.xlabel('Element Symbols')  # x축 라벨
#     plt.ylabel('Frequency')  # y축 라벨
#     plt.title('Element Frequency Mask Across All Epochs')  # 그래프 제목
#     plt.xticks(rotation=45)  # x축 라벨 회전
#     plt.grid(axis='y', linestyle='--', alpha=0.7)  # y축 그리드 추가
    
#     # 각 막대 위에 빈도수 표시
#     for bar in bars:
#         yval = bar.get_height()
#         percentage = (yval / total_frequencies) * 100  # 비율 계산
#         plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)} ({percentage:.1f}%)', ha='center', va='bottom', fontsize=8)

#     plt.tight_layout()  # 그래프 레이아웃 조정
#     dir_name = f'./plot_frequency_mask/{args.dataset.lower()}/{savefilename}{args.seed}/{args.mask_rate}/'
#     try : 
#         if not os.path.exists(dir_name):
#             os.makedirs(dir_name)
#     except OSError:
#         print('Error: Creating directory. ' +  dir_name)


#     plt.savefig(fileName)  # Saves the plot as a PNG filep
#     plt.close()

# def plot_tsne(args, loader, model, device):
#         from sklearn.manifold import TSNE
    
#         from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
#         from torch_geometric.data import Batch
    
   
#         if args.graph_pooling == 'mean':
#             pool = global_mean_pool
#         elif args.graph_pooling == 'max':
#             pool = global_max_pool
#         elif args.graph_pooling == 'add':
#             pool = global_add_pool

#         full_path = __file__
#         \
#         savefilename = os.path.basename(full_path)


#         dir_tsne = os.path.join('plot_recon_embedding',f'{args.dataset}', savefilename)

#         if not os.path.exists(dir_tsne):
#             os.makedirs(dir_tsne)

#         current_time = datetime.now().strftime('%b%d_%H-%M-%S')


#         dir_name =  savefilename + args.dataset + '_' + str(args.num_layer) + '_' \
#         + str(args.emb_dim) +  '_' + str(args.dropout_ratio) + '_' \
#         + str(args.split) + '_' + str(args.deviceName) + '_' + str(args.seed) + '_' + str(current_time)


#         file_name_tsne = f'{dir_name}.png'
        
#         full_file_name_tsne  = os.path.join(dir_tsne, file_name_tsne)

    
        
#         all_batches = []
#         for batch_in_loader in loader:
#             x = batch_in_loader.x.to(device)
#             edge_index = batch_in_loader.edge_index.to(device)
#             edge_attr = batch_in_loader.edge_attr.to(device)
#             batch = batch_in_loader.batch.to(device)
#             y = batch_in_loader.y.to(device)

#             if args.task == 'classification':
#                 # y 값을 각 태스크로 확장하여 추가
#                 y = y.view(-1, args.num_tasks)

#             data = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)

#             all_batches.append(data)

#         medged_batch = Batch.from_data_list(all_batches)
#         del all_batches

#         merged_batch = medged_batch.to(device)
#         with torch.no_grad():
#             model.eval()
#             pred_node, _ = model(medged_batch)
#             graph_rep = pool(pred_node, medged_batch.batch)

#             pred_tsne_np = graph_rep.detach().cpu().numpy()
#             target_values = merged_batch.y.detach().cpu().numpy()

#             pred_tsne_np = TSNE(n_components = 2).fit_transform(pred_tsne_np)

#             if args.task == 'regression': 
#                 plt.figure(figsize=(10, 10))
#                 scatter = plt.scatter(pred_tsne_np[:, 0], pred_tsne_np[:, 1], c=target_values, cmap='viridis', alpha=0.7)
#                 plt.colorbar(scatter, label='Target Value')
#                 plt.title('Latent Space with Continuous Values', fontsize=15)
#                 plt.xlabel('Z1', fontsize=15)
#                 plt.ylabel('Z2', fontsize=15)
#                 plt.savefig(full_file_name_tsne)
#                 plt.close()
                

#             elif args.task == 'classification':


#                     pred_tsne_np = graph_rep.detach().cpu().numpy()

#                     pred_tsne_np = TSNE(n_components = 2).fit_transform(pred_tsne_np)

#                     # 각 태스크마다 개별 플롯 생성
#                     for task in range(args.num_tasks):

#                                             # 현재 태스크에 대한 타겟값 가져오기
#                         task_values = merged_batch.y[:, task].cpu().numpy()
#                         unique_classes = np.unique(task_values)  # 현재 태스크에 대한 유니크한 클래스 가져오기

#                         plt.figure(figsize=(8, 6))
#                         for class_label in unique_classes:
#                                 class_indices = task_values == class_label
#                                 plt.scatter(pred_tsne_np[class_indices, 0], pred_tsne_np[class_indices, 1], label=f'Class {class_label}', alpha=0.7)

#                         plt.title(f'Latent Space for Task {task + 1}', fontsize=15)
#                         plt.xlabel('Z1', fontsize=15)
#                         plt.ylabel('Z2', fontsize=15)
#                         plt.tight_layout()

#                         file_name_tsne = f'{dir_name}_{task + 1}.png'
            
#                         full_file_name_tsne  = os.path.join(dir_tsne, file_name_tsne)
#                         plt.savefig(full_file_name_tsne)
#                         plt.close()

    
    

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def plot_graph_with_masked_nodes(data, file_name):
    # data: BatchMasking 객체

    G = nx.Graph()

    # 노드 추가
    for i in range(data.x.size(0)):
        G.add_node(i)

    edge_index = data.edge_index.cpu()
    # 엣지 추가
    edge_index = edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # 마스킹된 노드 인덱스
    masked_nodes = data.masked_atom_indices.cpu().numpy()

    pos = nx.spring_layout(G)

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    
    # 마스킹된 노드 그리기
    nx.draw_networkx_nodes(G, pos, nodelist=masked_nodes, node_size=300, node_color='red')
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos)

    # 라벨 그리기
    nx.draw_networkx_labels(G, pos)

    plt.savefig(file_name)
    plt.close()


def train(args, model_list, device, loader, optimizer_list, epoch, alpha_l=1.0, loss_fn="sce" ):

    # element_counts_mol = defaultdict(int)
    # element_counts_mask = defaultdict(int)
    

    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    # model, dec_pred_atoms, dec_pred_bonds, alpha = model_list
    model, dec_pred_atoms, dec_pred_bonds, = model_list

    # optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_alpha = optimizer_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds  = optimizer_list


    model.train()
    dec_pred_atoms.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

        
    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    loss_edge_recon = None

    
    for step, batch in enumerate(tqdm(loader, desc="Iteration", )):  # tqdm 추가

        batch = batch.to(device)

        node_rep, pred = model(batch.masked_x, batch.edge_index, batch.edge_attr, batch.batch)

        ## loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        # pred_node = dec_pred_atoms(node_recon, batch.edge_index, batch.edge_attr, masked_node_indices)

        # 원-핫 인코딩된 원자 번호를 실제 번호로 변환
        atomic_numbers = np.argmax(node_attr_label.cpu().numpy(), axis = 1)


        pred_node = dec_pred_atoms(node_rep)

        if loss_fn == "sce":
            loss_recon = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss_recon = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:,0])
            
        # if args.epochs == epoch:
        #     dir_name = f'./plot_mask_pretrain_batch/{args.split}/{args.dataset.lower()}/{savefilename}{args.seed}/{args.mask_rate}/'
        #     try : 
        #         if not os.path.exists(dir_name):
        #             os.makedirs(dir_name)
        #     except OSError:
        #         print('Error: Creating directory. ' +  dir_name)
                
        #     fileName =f'batch_train{epoch}_{step}_{args.num_layer}_{args.emb_dim}_{args.dropout_ratio}_{args.decay}.jpg'
        #     full_name = os.path.join(dir_name, fileName)
            
        #     plot_class_frequencies_filename( pred_node[masked_node_indices], batch.mask_node_label[:,0], full_name,args)

        acc_node = compute_accuracy(pred_node[masked_node_indices], batch.mask_node_label[:,0])

        acc_node_accum += acc_node

        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep)

            num_classes = pred_edge.shape[1]
            one_hot_target = F.one_hot(batch.mask_edge_label[:,0], num_classes=num_classes).to(torch.float64)
            loss_recon_edge =  criterion(pred_edge.double(), one_hot_target)
            

            acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        # optimizer_alpha.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

    
        if loss_edge_recon is not None:
            loss =  (loss_recon + loss_edge_recon) 
            
        else:
            loss =  loss_recon 
            

        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()
        # optimizer_alpha.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())


    return loss_accum / (step +1), acc_node_accum / (step +1), 0 #acc_edge_accum / step



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument("--loss_fn", type=str, default="ce")
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='mask ratio (default: 0.25)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--alpha',type = float, default = 0.00001, help = "coefficient for reconstruction loss")
    parser.add_argument('--plot_train',type = int, default = 0, help = "coefficient for reconstruction loss")


    args = parser.parse_args()
    # wandb.config.update(args)

    args.dataset = args.dataset.lower() #Convert to lower case


    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    deviceName = 'cuda'+ str(device)[-1] if torch.cuda.is_available() else 'cpu'
    args.deviceName = deviceName

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset.lower() == 'lipo': args.dataset = 'lipophilicity'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        args.task = 'classification'
    elif args.dataset == "hiv":
        num_tasks = 1
        args.task = 'classification'

    elif args.dataset == "pcba":
        num_tasks = 128
        args.task = 'classification'

    elif args.dataset == "muv":
        num_tasks = 17
        args.task = 'classification'

    elif args.dataset == "bace":
        num_tasks = 1
        args.task = 'classification'

    elif args.dataset == "bbbp":
        num_tasks = 1
        args.task = 'classification'

    elif args.dataset == "toxcast":
        num_tasks = 617
        args.task = 'classification'

    elif args.dataset == "sider":
        num_tasks = 27
        args.task = 'classification'

    elif args.dataset == "clintox":
        num_tasks = 2
        args.task = 'classification'

    elif args.dataset == "lipophilicity":
        num_tasks = 1
        args.task = 'regression'
    elif args.dataset == "freesolv":
        num_tasks = 1
        args.task = 'regression'
    elif args.dataset == "esol":
        num_tasks = 1
        args.task = 'regression'
    elif args.dataset == "qm7":
        num_tasks = 1
        args.task = 'regression'
    elif args.dataset == "qm8":
        num_tasks = 16
        args.task = 'regression'

    elif args.dataset == "zinc_sample":
        num_tasks = 1
        args.task = 'regression'

    elif args.dataset == "zinc_standard_agent":
        num_tasks = 1
        args.task = 'regression'
    elif args.dataset == "pubchem-10m-clean":
        num_tasks = 1
        args.task = 'regression'
    elif args.dataset == "pretrain_example":
        num_tasks = 1
        args.task = 'regression'
        
    else:
        raise ValueError("Invalid dataset name.")
        
    args.num_tasks = num_tasks
    
    if args.task == 'classification':
        args.criterion = nn.BCEWithLogitsLoss(reduction = "none")

    elif args.task == 'regression':
        if args.dataset in ['qm7','qm8', 'qm9']:
            args.criterion = nn.L1Loss()
        else:
            args.criterion = nn.MSELoss()


    # 현재 파일의 전체 경로 및 파일명
    full_path = __file__

    savefilename = os.path.basename(full_path)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name =  savefilename + args.dataset + '_' + str(args.num_layer) + '_' \
    + str(args.emb_dim) + '_'  + str(args.dropout_ratio) + '_' \
    + str(args.split) + '_' + str(deviceName) + '_' + str(args.seed) + '_' + str(current_time)

    args.log_dir = os.path.join('./finetune', dir_name)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    layout = {
    "RECON": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
        "accuracy": ["Multiline", [ "accuracy/train", "accuracy/validation"]],
        },
    }
    writer.add_custom_scalars(layout)

    model_checkpoints_folder = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder,exist_ok=True)

    if args.dataset == 'zinc_standard_agent':
        from dataset_pretrain_zinc import MoleculeDataset
        
        dataset = MoleculeDataset(data_path = "dataset/" + args.dataset, )
    else: 
        from dataset_pretrain import MoleculeDataset
        
        dataset = MoleculeDataset(data_path = "dataset/" + args.dataset, )
    


    # if args.dataset ==  "pubchem-10m-clean":
    #     dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    # #set up dataset
    # else:
    #     dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset, args.dataset, args.seed)
    
    # if args.split == "scaffold":
    #     smiles_list = pd.read_csv('dataset - plot/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    #     print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    # #     print("random")
    # # elif args.split == "random_scaffold":
    # #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    # #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    # #     print("random scaffold")
    # else:
    #     raise ValueError("Invalid split option.")

    # train_loader = DataLoaderMaskingPredTest(dataset, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers = args.num_workers,  mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    transform = MaskAtomTest(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge) 

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda data_list: BatchMasking.from_data_list([transform(data) for data in data_list]))


    args.normalizer = None
    if args.dataset in ['qm7', 'qm9']:
        labels = []
        for d  in train_loader:
            labels.append(d.y)
        labels = torch.cat(labels)
        args.normalizer = Normalizer(labels)
        print(args.normalizer.mean, args.normalizer.std, labels.shape)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        print("load pretrained model from:", args.input_model_file)
        model.from_pretrained(args.input_model_file)


    NUM_NODE_ATTR = 119 # + 3 
    # atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    atom_pred_decoder = torch.nn.Linear(args.emb_dim, 119).to(device)
    bond_pred_decoder = torch.nn.Linear(args.emb_dim, 4).to(device)

    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        # bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None


    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})


    optimizer_model = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)


    # alpha = torch.tensor(args.alpha, dtype = torch.float)

    # alpha  = torch.nn.Parameter(torch.tensor(alpha, dtype = torch.float), requires_grad=True)

    # optimizer_alpha = torch.optim.Adam([alpha], lr=args.lr, weight_decay=args.decay)

    # optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_alpha]
    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds ]

    # model_list = [model, atom_pred_decoder, bond_pred_decoder, alpha] 
    model_list = [model, atom_pred_decoder, bond_pred_decoder] 
    model.to(device)
    # alpha.to(device)

    output_file_temp = "./pretrain2/" + "/node/"+ f"{args.dataset}_{args.gnn_type}_{args.mask_rate}_{args.seed}"        
    if not os.path.exists(output_file_temp):
        os.makedirs(output_file_temp)


    if args.scheduler:
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5

        scheduler_model = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=30, gamma=0.3)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        # scheduler_alpha = torch.optim.lr_scheduler.LambdaLR(optimizer_alpha, lr_lambda=scheduler)
        
        # scheduler_list = [scheduler_model, scheduler_dec, scheduler_alpha]
        scheduler_list = [scheduler_model, scheduler_dec]


    else:
        scheduler_model = None
        scheduler_dec = None
        # scheduler_alpha = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    best_val_acc = 0
    final_test_acc = 0

    best_val_loss = np.inf    
    final_test_loss = np.inf

    
    
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_loss, train_acc_atom, train_acc_bond,  = train(args, model_list, device, train_loader, optimizer_list, alpha_l=args.alpha_l, loss_fn=args.loss_fn, epoch= epoch)


        print("train_loss: %f, train_acc_atom: %f, train_acc_bond: %f" %(train_loss, train_acc_atom, train_acc_bond))
        

        if scheduler_model is not None:
                scheduler_model.step()
                scheduler_dec.step()
                # scheduler_alpha.step()


        if epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(output_file_temp, 'model.pth'))

        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc_atom, epoch)
        # wandb.log({'loss/train': train_loss, 'accuracy/train' : train_acc_atom})
        
    
    columns = ['Time', 'Seed', 'mask_rate','loss', 'node_acc', 'alpha']

    results_list = []
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') # 시간을 지정 

    results_list.append([current_time, args.seed, args.mask_rate, train_loss, train_acc_atom, args.alpha])

    
    directory_name = 'results_pretrain'
    os.makedirs(directory_name, exist_ok=True)
    df = pd.DataFrame(results_list, columns=columns)


    fn_base = f'{directory_name}/pretrain_node_{args.dataset}_{args.split}'

    if args.gnn_type != 'gin':
        fn_base += f'_{args.gnn_type}'
    
    if args.mask_edge: 
        fn_base += '_edge'
        
    fn = f'{fn_base}.csv'


    if os.path.exists(fn):
        # 기존 파일 읽기
        existing_df = pd.read_csv(fn)
        # 새로운 데이터 추가
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df


    combined_df.to_csv(
        fn, 
        index=False, 
    )

    # wandb.finish()

if __name__ == "__main__":
    main()
