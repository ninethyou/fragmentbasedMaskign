from functools import partial

import argparse
import copy
# import wandb

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from dataloader import DataLoaderMaskingAdjacentPred#, DataListLoader

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


from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import io
from PIL import Image

full_path = __file__

savefilename = os.path.basename(full_path)

def compute_accuracy(pred, target):
    # print(pred.shape, target.shape)
    # return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

        _, predicted_classes = torch.max(pred, dim=1) # dim=1은 노드별로 최댓값을 찾음

        correct_predictions = (predicted_classes == target)

        accuracy = correct_predictions.float().mean().item()

        return accuracy
    
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
        # y = batch.y.view(pred.shape).to(torch.float64)

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
            # plot_class_frequencies_filename( pred_node[masked_node_indices], batch.mask_node_label[:,0], full_name,args)

        # plot_molecule(batch, pred_node[masked_node_indices], batch.mask_node_label[:,0], step, f'./example/')

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


    return loss_accum / (step +1), acc_node_accum / (step +1), 0,   #acc_edge_accum / step



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

    elif args.dataset == "zinc_standard_agent":
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
        os.makedirs(model_checkpoints_folder)

    if args.dataset == 'zinc_standard_agent':
        from dataset_pretrain_zinc import MoleculeDataset
        from util import MaskSubGraph
        from batch import BatchMasking        
        
        dataset = MoleculeDataset(data_path = "dataset/" + args.dataset, )
        
        transform = MaskSubGraph(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge) 

        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda data_list: BatchMasking.from_data_list([transform(data) for data in data_list]))

    
    else: 
        from loader import MoleculeDataset
        from util import MaskSubGraph
        from batch import BatchMasking    
        
        dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

        transform = MaskSubGraph(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge) 
    
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda data_list: BatchMasking.from_data_list([transform(data) for data in data_list]))

    print(dataset, args.dataset, args.seed)

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

    output_file_temp = "./pretrain2/" + "/subgraph/" + f"{args.dataset}_{args.gnn_type}_{args.mask_rate}_{args.seed}"        
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
        
        train_loss, train_acc_atom, train_acc_bond  = train(args, model_list, device, train_loader, optimizer_list, alpha_l=args.alpha_l, loss_fn=args.loss_fn, epoch= epoch)


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


    fn_base = f'{directory_name}/pretrain_subgraph_{args.dataset}_{args.split}'

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
