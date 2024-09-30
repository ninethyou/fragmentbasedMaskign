from functools import partial

import argparse
import copy

from matplotlib import pyplot as plt

from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from dataloader import DataLoaderMasking, DataLoaderMaskingPred, DataLoaderMaskingPredTest#, DataListLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
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


from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import io
from PIL import Image

full_path = __file__

savefilename = os.path.basename(full_path)
# 현재 파일의 전체 경로 및 파일명

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

def plot_class_frequencies_filename(pred, label, fileName,args):

    accuracy = calc_acc(pred, label)

    _, predicted_classes = torch.max(pred, dim=1) # dim=1은 노드별로 최댓값을 찾음

    # Convert tensors to numpy arrays for plotting
    predicted_classes = predicted_classes.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    

    # Get unique classes and their counts
    classes = np.unique(np.concatenate((predicted_classes, label)))
    pred_counts = [np.sum(predicted_classes == cls) for cls in classes]
    label_counts = [np.sum(label == cls) for cls in classes]
        # 비율 계산 및 표시
    total_pred = sum(pred_counts)
    total_labels = sum(label_counts)


    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Set the bar width
    bar_width = 0.35
    # Set the index for groups
    index = np.arange(len(classes))

    bar1 = ax.bar(index, pred_counts, bar_width, label='Predicted')
    bar2 = ax.bar(index + bar_width, label_counts, bar_width, label='Actual')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')
    ax.set_title('Predicted vs Actual Class Frequencies')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(classes)
    ax.legend()
    
    for bar, count in zip(bar1, pred_counts):
        percent = 100 * count / total_pred
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percent:.1f}%', ha='center', va='bottom')
    
    for bar, count in zip(bar2, label_counts):
        percent = 100 * count / total_labels
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percent:.1f}%', ha='center', va='bottom')
    
    accuracy_text = f'Accuracy: {accuracy:.2f}'
    ax.text(0.15, 0.95, accuracy_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='center')

    # Save the plot to a file
    plt.tight_layout()

    plt.savefig(fileName)  # Saves the plot as a PNG file
    plt.close()




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




def train(args, model_list, device, loader, optimizer_list, epoch, alpha_l=1.0, loss_fn="sce" ):

    element_counts_mol = defaultdict(int)
    element_counts_mask = defaultdict(int)
    

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
    loss_accum_end = 0
    loss_accum_recon_node = 0
    loss_accum_recon_edge = 0
 
    
    acc_node_accum = 0
    acc_edge_accum = 0
    loss_recon_edge = None


    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        node_rep, pred = model(batch.masked_x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

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


        if args.task == 'classification': 
        #Whether y is non-null or not.
           
        
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = args.criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            loss_end = torch.sum(loss_mat)/torch.sum(is_valid)

        
        elif args.task == 'regression':

            if args.normalizer: 
                loss_end = args.criterion(pred, args.normalizer.norm(y.float()))
            else:
                loss_end = args.criterion(pred, y.float())
            

        

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        # optimizer_alpha.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

    
        if dec_pred_bonds is not None:
            loss =  (loss_recon + loss_recon_edge) * args.alpha + loss_end
            
        else:
            loss =  loss_recon * args.alpha  + loss_end
                      
        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()
        # optimizer_alpha.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        loss_accum_end += float(loss_end.cpu().item())
        loss_accum_recon_node += float(loss_recon.cpu().item())
        
  
        if dec_pred_bonds is not None:
        
            loss_accum_recon_edge += float(loss_recon_edge.cpu().item())
        else:
            loss_accum_recon_edge = 0

    # return loss_accum / (step+1), acc_node_accum / step+1, 0, element_counts_mol, element_counts_mask  #acc_edge_accum / step
    return acc_node_accum / (step+1), \
            loss_accum / (step+1), \
            loss_accum_end / (step+1), \
            ((loss_accum_recon_node) / (step+1), loss_accum_recon_edge / (step + 1))
            

def eval(args, model_list, device, loader, optimzier_list, alpha_l=1.0, loss_fn="sce", data_split = "val"):

    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    # model, dec_pred_atoms, dec_pred_bonds, alpha= model_list
    model, dec_pred_atoms, dec_pred_bonds, = model_list

    # optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_alpha = optimzier_list
    optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds = optimzier_list

    
    model.eval()
    dec_pred_atoms.eval()

    if dec_pred_bonds is not None:
        dec_pred_bonds.eval()
        acc_edge_accum = 0
        
    acc_node_accum = 0
    loss_accum = 0
    loss_accum_end = 0
    loss_accum_recon_node = 0
    loss_accum_recon_edge = 0

    y_true = []
    y_scores = []

    bn = 0 
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            _, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) 
            node_recon,_  = model(batch.masked_x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape).to(torch.float64)
            

        ## loss for nodes
        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        # pred_node = dec_pred_atoms(node_recon, batch.edge_index, batch.edge_attr, masked_node_indices)
        pred_node = dec_pred_atoms(node_recon)  
          

        if loss_fn == "sce":
            loss_recon = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            loss_recon = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:,0])

        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        
        acc_node = compute_accuracy(pred_node[masked_node_indices], batch.mask_node_label[:,0])
        acc_node_accum += acc_node


        if args.normalizer:
            pred = args.normalizer.denorm(pred)
        
        if args.task == 'classification':
            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)
        
        elif args.task == 'regression':
            if args.num_tasks > 1:
                y_scores.append(pred)
                y_true.append(batch.y.view(pred.shape))
                
            else:
                if device == 'cpu':
                    y_true.extend(batch.y.flatten().numpy())
                    y_scores.extend(pred.detach().numpy())
                else:
                    y_true.extend(batch.y.cpu().flatten().numpy())
                    y_scores.extend(pred.cpu().detach().numpy())


        if args.task == 'classification': 
        #Whether y is non-null or not.
           
        
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = args.criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            loss_end = torch.sum(loss_mat)/torch.sum(is_valid)

        
        elif args.task == 'regression':

            if args.normalizer: 
                loss_end = args.criterion(pred, args.normalizer.norm(y.float()))
            else:
                loss_end = args.criterion(pred, y.float())


        if data_split == 'test':

            dir_name = f'./plot_mask_reconOnly_batch2/test_{args.split}/{args.dataset.lower()}/{savefilename}{args.seed}/{args.mask_rate}/'
            try : 
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
            except OSError:
                print('Error: Creating directory. ' +  dir_name)

            fileName =f'batch_test_{step}_{args.num_layer}_{args.emb_dim}_{args.dropout_ratio}_{args.decay}.jpg'
            full_name = os.path.join(dir_name, fileName)

            plot_class_frequencies_filename( pred_node[masked_node_indices], batch.mask_node_label[:,0], full_name,args)

              
            # if args.split == 'random':
            #     save_dir = os.path.join('plot_mask_recon_molecule_random', args.dataset,  str(args.mask_rate), str(args.seed))          
            # elif args.split == 'scaffold':
            #     save_dir = os.path.join('plot_mask_recon_molecule_scaffold', args.dataset,  str(args.mask_rate), str(args.seed))          
                
            # if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)

            # plot_molecule(batch, pred_node, masked_node_indices, bn, save_dir)

        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_recon[masked_edge_index[0]] + node_recon[masked_edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep)

            num_classes = pred_edge.shape[1]
            one_hot_target = F.one_hot(batch.mask_edge_label[:,0], num_classes=num_classes).to(torch.float64)
            loss_recon_edge =  criterion(pred_edge.double(), one_hot_target)

            acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
            acc_edge_accum += acc_edge


        
        bn =  bn  +  1

        if dec_pred_bonds is not None:
            loss =  (loss_recon + loss_recon_edge) + loss_end
            
        else:
            loss =  loss_recon + loss_end
        
        loss_accum += float(loss.cpu().item())
        loss_accum_end += float(loss_end.cpu().item())
        loss_accum_recon_node += float(loss_recon.cpu().item())

        if dec_pred_bonds is not None:
        
            loss_accum_recon_edge += float(loss_recon_edge.cpu().item())
        else:
            loss_accum_recon_edge = 0

    if args.task == 'classification':
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()


        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

        
        score = sum(roc_list)/len(roc_list) 

    elif args.task == 'regression':

        
        if args.dataset in ['qm7','qm8', 'qm9']:

            if args.num_tasks > 1:
                y_true = torch.cat(y_true, dim = 0).cpu().numpy()
                y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            
            mae =  mean_absolute_error(y_true, y_scores)
            score = mae
    
        else:
            rmse = root_mean_squared_error(y_true, y_scores)
            score = rmse
            
    return score, acc_node_accum / (step + 1),  loss_accum / (step + 1), loss_accum_end / (step + 1),  \
            (loss_accum_recon_node / (step + 1), loss_accum_recon_edge / (step + 1))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
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
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument('--mask_rate', type=float, default=0.25,
                        help='mask ratio (default: 0.25)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--alpha',type = float, default = 0.001, help = "coefficient for reconstruction loss")
    parser.add_argument('--plot_train',type = int, default = 0, help = "coefficient for reconstruction loss")


    args = parser.parse_args()
    args.dataset = args.dataset.lower() #Convert to lower case


    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    deviceName = 'cuda'+ str(device)[-1] if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset.lower() == 'lipo': args.dataset = 'lipophilicity'

    if args.dataset == 'lipo': args.dataset = 'lipophilicity'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        args.num_tasks = 12
        
        args.task = 'classification'
    elif args.dataset == "hiv":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'classification'

    elif args.dataset == "pcba":
        num_tasks = 128
        args.num_tasks = 128
        
        args.task = 'classification'

    elif args.dataset == "muv":
        num_tasks = 17
        args.num_tasks = 17
        
        args.task = 'classification'

    elif args.dataset == "bace":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'classification'

    elif args.dataset == "bbbp":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'classification'

    elif args.dataset == "toxcast":
        num_tasks = 617
        args.num_tasks = 617
        
        args.task = 'classification'

    elif args.dataset == "sider":
        num_tasks = 27
        args.num_tasks = 27
        
        args.task = 'classification'

    elif args.dataset == "clintox":
        num_tasks = 2
        args.num_tasks = 2
        
        args.task = 'classification'

    elif args.dataset == "lipophilicity":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'regression'
    elif args.dataset == "freesolv":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'regression'
    elif args.dataset == "esol":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'regression'
    elif args.dataset == "qm7":
        num_tasks = 1
        args.num_tasks = 1
        
        args.task = 'regression'

    elif args.dataset == "qm8":
            num_tasks = 16
            args.num_tasks = 16
        
            args.task = 'regression'
        
    else:
        raise ValueError("Invalid dataset name.")


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
        os.makedirs(model_checkpoints_folder)


    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(args.dataset,dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random")
    # elif args.split == "random_scaffold":
    #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoaderMaskingPredTest(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers = args.num_workers,  mask_rate=args.mask_rate, mask_edge=args.mask_edge)
    val_loader = DataLoaderMaskingPredTest(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers,  mask_rate=args.mask_rate, mask_edge=args.mask_edge)
    test_loader = DataLoaderMaskingPredTest(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers,  mask_rate=args.mask_rate, mask_edge=args.mask_edge)

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


    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds ]

    # model_list = [model, atom_pred_decoder, bond_pred_decoder, alpha] 
    model_list = [model, atom_pred_decoder, bond_pred_decoder] 
    model.to(device)
    # alpha.to(device)


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

    if args.task == 'classification':
        score = 0
    elif args.task == 'regression':
        score = np.inf

    # result_filename = f"results/{args.dataset}_{args.input_model_file}.result"
    # fw = open(result_filename, "a")
    # fw.write(f"----- seed {args.seed} -------- \n")

    # if not args.filename == "":
    #     fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
    #     #delete the directory if there exists one
    #     if os.path.exists(fname):
    #         shutil.rmtree(fname)
    #         print("removed the existing file.")
    #     writer = SummaryWriter(fname)

    total_element_counts_mol = defaultdict(int)
    total_element_counts_mask = defaultdict(int)
    
    
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train_acc, train_loss,  train_loss_end, train_loss_recon \
        = train(args, model_list, device, train_loader, optimizer_list, alpha_l=args.alpha_l, loss_fn=args.loss_fn, epoch= epoch)


        print(f'train node acc: {train_acc:4f},  train loss:{train_loss:3f},train loss_end:{train_loss_end:3f}, train_loss_recon_node:{train_loss_recon[0]:3f}, train_loss_recon_node:{train_loss_recon[1]:3f}')

        # for element, count in epoch_element_counts_mol.items():
        #     total_element_counts_mol[element] += count

        # for element, count in epoch_element_counts_mask.items():
        #     total_element_counts_mask[element] += count


        if scheduler_model is not None:
                scheduler_model.step()
                scheduler_dec.step()
                # scheduler_alpha.step()

        print("====Evaluation")
        if args.eval_train:
            score_train, train_acc, train_loss, train_loss_end, train_loss_recon  \
            = eval(args, model_list, device, train_loader, optimizer_list,  alpha_l=args.alpha_l, loss_fn=args.loss_fn, data_split = "train")

        else:
            print("omit the training accuracy computation")
            train_acc = 0
            
        score_val, val_acc, val_loss,  val_loss_end, val_loss_recon \
        = eval(args, model_list, device, val_loader, optimizer_list,  alpha_l=args.alpha_l, loss_fn=args.loss_fn, data_split = "val")


        if args.task == 'classification':
            if score_val > score:
                score = score_val
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                
        elif args.task == 'regression':
            if score_val < score:
                score = score_val
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                
                          
            
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

        print("val_result: %f, best_val_result: %f  " %(score_val, score))
        
        print("train_loss: %f val_loss: %f" %(train_loss, val_loss))
        
        print("train_loss_end: %f val_loss_end: %f" %(train_loss_end, val_loss_end))
        print("train_loss_recon_node: %f val_loss_recon_node: %f" %(train_loss_recon[0], val_loss_recon[0]))
        print("train_loss_recon_edge: %f val_loss_recon_edge: %f" %(train_loss_recon[1], val_loss_recon[1]))
        
        
        print("train_acc: %f val_acc: %f" %(train_acc, val_acc,))


        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)



    model_path = os.path.join(args.log_dir, 'checkpoints', 'model.pth')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded trained model with success.")


    score_test, test_acc_node, test_loss, test_loss_end, test_loss_recon \
    = eval(args, model_list, device, test_loader, optimizer_list,  alpha_l=args.alpha_l, loss_fn=args.loss_fn,data_split = 'test')

    print(f'test node acc: {test_acc_node:4f},  test loss:{test_loss:3f}, \
    test loss_end:{test_loss_end:3f}, test_loss_recon_node:{test_loss_recon[0]:3f},  test_loss_recon_node:{test_loss_recon[1]:3f}')
    
    columns = ['Time', 'Seed', 'mask_rate','result','node_acc', 'alpha']

    results_list = []
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') # 시간을 지정 
    results_list.append([current_time, args.seed, args.mask_rate, score_test, test_acc_node, args.alpha])

    directory_name = 'results_pretrain+finetune'     
    os.makedirs(directory_name, exist_ok=True)
    df = pd.DataFrame(results_list, columns=columns)


    fn_base = f'{directory_name}/{args.dataset}_pretrain+finetune+recon'

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

if __name__ == "__main__":
    main()
