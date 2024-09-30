import argparse
import copy

from loader_rich import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
from datetime import datetime

from model_rich import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error


from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


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




def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        _,pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.x_add, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)


        if args.task == 'classification': 
        #Whether y is non-null or not.
           
        
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = args.criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            loss = torch.sum(loss_mat)/torch.sum(is_valid)

        
        elif args.task == 'regression':

            if args.normalizer: 
                loss = args.criterion(pred, args.normalizer.norm(y.float()))
            else:
                loss = args.criterion(pred, y.float())
            
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            _,pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.x_add, batch.batch)

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

        return sum(roc_list)/len(roc_list) #y_true.shape[1]

    elif args.task == 'regression':
        if args.num_tasks > 1:
            y_true = torch.cat(y_true, dim = 0).cpu().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()


        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        if args.dataset in ['qm7','qm8', 'qm9']:
            mae =  mean_absolute_error(y_true, y_scores)
            return mae
    
        else:
            rmse = root_mean_squared_error(y_true, y_scores)
            return rmse




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

    if args.dataset == 'lipo': args.dataset = 'lipophilicity'

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

    model_checkpoints_folder = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)


    #set up dataset
    dataset = MoleculeDataset("dataset - rich/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset - rich/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random")
    # elif args.split == "random_scaffold":
    #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random scaffold")
    else:
        raise ValueError("Invalid split option.")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

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
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if args.task == 'classification':
        best_val_acc = 0
        final_test_acc = 0

    elif args.task == 'regression':
        best_val_acc =  np.inf
        final_test_acc = np.inf



    # result_filename = f"results/{args.dataset}_{args.input_model_file}.result"
    # fw = open(result_filename, "a")
    # fw.write(f"----- seed {args.seed} -------- \n")

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        

        if args.task == 'classification':
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))


            else:
                final_val_acc = val_acc

        elif args.task == 'regression':
            if   val_acc < best_val_acc:
                best_val_acc = val_acc
                final_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))


            else:
                final_val_acc = val_acc


        print("train: %f val: %f  best_val_acc: %f" %(train_acc, val_acc, final_val_acc))

        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)


    model_path = os.path.join(args.log_dir, 'checkpoints', 'model.pth')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded trained model with success.")


    test_acc = eval(args, model, device, test_loader)
    print("test auc:", test_acc)
    
    columns = ['Time', 'Seed', 'result', ]

    results_list = []
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') # 시간을 지정 

    results_list.append([current_time, args.seed, test_acc])

    dir_name = 'results_pretrain+finetune'
    os.makedirs(dir_name, exist_ok=True)
    df = pd.DataFrame(results_list, columns=columns)


    if not args.input_model_file == "":
        fn_base = f'{dir_name}/{args.dataset}_pretrain+downstream_rich2'
    else:
        fn_base = f'{dir_name}/{args.dataset}_downstream_rich2'
        
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
