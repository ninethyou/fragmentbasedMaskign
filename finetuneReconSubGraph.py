import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import random
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from dataset.dataset_test_subgraph import MolTestDatasetWrapper
from dataset.get_config import get_config 
import argparse
from torch_geometric.utils import  scatter, softmax

from torch_geometric.data import Data

from util import random_node, get_neighbors

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))

def get_roc_auc_score(y_true, y_pred, is_valid):
    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_pred[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return  sum(roc_list)/len(roc_list)




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


           
layout = {
    "recon": {
        "loss_end": ["Multiline", ["loss_end/train", "loss_end/validation"]],
        "loss_recon_node" : ["Multiline", ["loss_recon_node/train"]],
        "loss_recon_edge" : ["Multiline", ["loss_recon_edge/train"]],
        "loss_total" : ["Multiline", ["loss_total/train", "loss_total/validation"]],
        "accuracy": ["Multiline", [ "accuracy/validation"]],
    },
}
layout = {
    "recon": {
        "loss_end": ["Multiline", ["loss_end/train", "loss_end/validation"]],
        "loss_recon_node" : ["Multiline", ["loss_recon_node/train"]],
        "loss_recon_edge" : ["Multiline", ["loss_recon_edge/train"]],
        "loss_total" : ["Multiline", ["loss_total/train", "loss_total/validation"]],
        "accuracy": ["Multiline", [ "accuracy/validation"]],
    },
}


# 현재 파일의 전체 경로 및 파일명
full_path = __file__

# 현재 파일명만 추출
savefilename = os.path.basename(full_path)

class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()


        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name =  savefilename + config['task_name'] + '_' + str(args.num_layer) + '_' \
        + str(args.emb_dim) + '_' + str(args.feat_dim)  + '_' + str(args.dropout) + '_' \
        + str(args.splitting) + '_' + str(args.deviceName) + '_' + str(args.seed) + '_' + str(current_time)

        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_custom_scalars(layout)

        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion =  nn.BCEWithLogitsLoss(reduction = "none")
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()
        self.criterion_recon = nn.CrossEntropyLoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            args.deviceName = "cuda" + str(device[-1])

        else:
            device = 'cpu'
            args.deviceName = 'cpu'

        print("Running on:", device)

        return device
    
    def _step_test(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':

            is_valid = data.y**2 > 0

            loss_mat = self.criterion(pred, (data.y+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))


            loss = torch.sum(loss_mat) / torch.sum(is_valid)

        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)
        return loss


    def _step(self, model_list, data, n_iter):
        model, linear_pred_atoms, linear_pred_bonds = model_list

        num_atom_type = 119
        num_edge_type = 4

        # get the prediction
        node_rep, pred = model(data)  # [N,C]

        masked_atom_indices = data.removed_node_index
        
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))

        mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        masked_atom_indices = torch.tensor(masked_atom_indices)
      

        masked_x = data.x.clone()

        for atom_idx in masked_atom_indices:
            masked_x[atom_idx] = torch.tensor([num_atom_type, 0]).to(self.device)


        if args.mask_edge:

            # edge_index 찾기
            connected_edge_indices = data.removed_edge_index

            masked_edge_attr = data.edge_attr.clone()

            
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                print(len(connected_edge_indices[::2]))
                
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))
                    

                mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms

                for bond_idx in connected_edge_indices:
                    masked_edge_attr[bond_idx] = torch.tensor(
                        [num_edge_type, 0])
                    
                connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2], dtype=torch.long).to(self.device)

            else:
                mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                connected_edge_indices = torch.tensor(
                    connected_edge_indices, dtype=torch.long).to(self.device)


            masked_edge_index = data.edge_index[:, connected_edge_indices]

            masked_data = Data(x = masked_x ,edge_index = data.edge_index, edge_attr = masked_edge_attr) 

            node_rep_masked, output2_masked = model(data)

            print(masked_edge_index[0].shape, masked_edge_index[1].shape)
            # edge_rep 
            edge_rep = node_rep_masked[masked_edge_index[0]] + node_rep_masked[masked_edge_index[1]]
            pred_edge = linear_pred_bonds(edge_rep)

            print(mask_edge_label.shape, pred_edge.shape)

            loss_recon_edge = self.criterion_recon(pred_edge.double(),  mask_edge_label[:,0])


            pred_node = linear_pred_atoms(node_rep_masked[masked_atom_indices])
            loss_recon_node = self.criterion_recon(pred_node.float(), mask_node_label[:,0])
        
        else: 
            masked_data = Data(x = data.x ,edge_index = data.edge_index, edge_attr = data.edge_attr) 

            node_repre2, output2_masked= model(masked_data, masked_atom_indices)
            
            pred_node = linear_pred_atoms(node_repre2[masked_atom_indices])
            loss_recon_node = self.criterion_recon(pred_node.float(), mask_node_label[:,0])


        if self.config['dataset']['task'] == 'classification':

            is_valid = data.y**2 > 0
            
            loss_mat = self.criterion(pred, (data.y+1)/2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))


            loss = torch.sum(loss_mat) / torch.sum(is_valid)



        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        
        if args.mask_edge:
            total_loss = loss + args.alpha * (loss_recon_node + loss_recon_edge)
            return pred, total_loss, loss, loss_recon_node, loss_recon_edge
        else:
            total_loss = loss + args.alpha * loss_recon_node
            return pred, total_loss, loss, loss_recon_node, 0


        

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d  in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)


        self.config['model']['num_task'] = len(self.config['dataset']['target'])
        if self.config['model_type'] == 'gin':
            from models.ginet_finetuneRecon import GINetReconEmbedding
            model = GINetReconEmbedding(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            # model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            # model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(self.device)
        linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(self.device)


        optimizer_model = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        optimizer_linear_pred_atoms = torch.optim.Adam(linear_pred_atoms.parameters(), lr=args.init_base_lr, weight_decay=args.weight_decay)
        optimizer_linear_pred_bonds = torch.optim.Adam(linear_pred_bonds.parameters(), lr=args.init_base_lr, weight_decay=args.weight_decay)

        model_list = [model, linear_pred_atoms, linear_pred_bonds]
        optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                predictions = []
                labels = []

                optimizer_model.zero_grad()
                optimizer_linear_pred_atoms.zero_grad()
                if args.mask_edge:
                    optimizer_linear_pred_bonds.zero_grad()

                data = data.to(self.device)
                pred, total_loss, loss_end, loss_recon, loss_recon_edge = self._step(model_list, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    # self.writer.add_scalar('loss_total/train', total_loss, global_step=n_iter)
                    self.writer.add_scalar('loss_total/train_loss_total', total_loss, epoch_counter)
                    self.writer.add_scalar('loss_end/train_loss_end', loss_end, epoch_counter)
                    self.writer.add_scalar('loss_recon_node/train_loss_recon_node', loss_recon, epoch_counter)
                    self.writer.add_scalar('loss_recon_edge/train_loss_recon_edge', loss_recon_edge, epoch_counter)


                    print(epoch_counter, bn, total_loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                optimizer_model.step()
                optimizer_linear_pred_atoms.step()
                if args.mask_edge:
                    optimizer_linear_pred_bonds.step()

                n_iter += 1


            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 


                    valid_loss, valid_cls = self._validate(model_list, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                    self.writer.add_scalar('accuracy/validation', valid_cls, epoch_counter)
                elif self.config['dataset']['task'] == 'regression':

             
                    valid_loss, valid_rgr = self._validate(model_list, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    self.writer.add_scalar('accuracy/validation', valid_rgr, epoch_counter)

                self.writer.add_scalar('loss_end/validation_loss_end', valid_loss, epoch_counter)
                self.writer.add_scalar('loss_total/validation_loss_total', valid_loss, epoch_counter)


            

                valid_n_iter += 1
        
        self._test(model_list, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model_list, valid_loader):
        model, linear_pred_atoms, linear_pred_bonds = model_list

        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()
            linear_pred_atoms.eval()
            linear_pred_bonds.eval()



            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step_test(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    
                    labels.append(data.y.view(pred.shape))
                    predictions.append(pred)

                else:
                    if self.device == 'cpu':
                        predictions.extend(pred.detach().numpy())
                        labels.extend(data.y.flatten().numpy())
                    else:
                        predictions.extend(pred.cpu().detach().numpy())
                        labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()
        linear_pred_atoms.train()
        linear_pred_bonds.train()


        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = root_mean_squared_error(labels, predictions, )
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            
            labels = torch.cat(labels, dim=0).cpu().numpy()
            predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()

            is_valid = labels**2 > 0
            roc_auc = get_roc_auc_score(labels, predictions, is_valid)
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model_list, test_loader):

        model, linear_pred_atoms, linear_pred_bonds = model_list


        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()
            linear_pred_atoms.eval()
            linear_pred_bonds.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step_test(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':

                    predictions.append(pred)
                    labels.append(data.y.view(pred.shape))

                else:
                    if self.device == 'cpu':
                        predictions.extend(pred.detach().numpy())
                        labels.extend(data.y.flatten().numpy())
                    else:
                        predictions.extend(pred.cpu().detach().numpy())
                        labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()
        linear_pred_atoms.train()
        linear_pred_bonds.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = root_mean_squared_error(labels, predictions, )
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 

            
            labels = torch.cat(labels, dim=0).cpu().numpy()
            predictions = torch.cat(predictions, dim=0).cpu().numpy()

            is_valid = labels**2 > 0
            self.roc_auc = get_roc_auc_score(labels, predictions, is_valid)

            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)


def main(config):

    dataset = MolTestDatasetWrapper(config['batch_size'],
                                    **config['dataset'],
                                    random_masking=args.random_masking,
                                    mask_rate=args.mask_rate,
                                    mask_edge=args.mask_edge)

    fine_tune = FineTune(dataset, config)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return fine_tune.mae
        else:
            return fine_tune.rmse


if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=32, help = "Batch size")
    parser.add_argument('--epochs', type=int, default=200, help = "Number of training epochs")
    parser.add_argument('--init_lr', type=float, default=0.0005, help = "Initial learning rate")
    parser.add_argument('--init_base_lr', type=float, default=0.0001, help = "Initial learning rate for base layers")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help = "Weight decay")

    parser.add_argument('--gpu', type=str, default='cuda:0', help = "GPU id. Set to 'cpu' if using CPU")
    parser.add_argument('--model_type', type=str, default='gin', help = "Type of GNN model")
    parser.add_argument('--num_layer', type=int, default=5, help = "Number of GNN layers")
    parser.add_argument('--emb_dim', type=int, default=300, help = "Dimension of node embedding")
    parser.add_argument('--feat_dim', type=int, default=300, help = "Dimension of input node features")
    parser.add_argument('--dropout', type=float, default=0.3, help = "Dropout ratio")
    parser.add_argument('--pool', type=str, default='mean', help = "Pooling method")


    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--task_name', type=str, default='BBBP'
                        , help = "Name of the downstream task.")
    parser.add_argument('--splitting', type=str, default='scaffold', help = "Type of splitting for the dataset.")
    parser.add_argument('--random_masking', type=int, default='0', help = "Whether to use random masking for the dataset.")
    parser.add_argument('--mask_rate', type=float, default='0.2', help = "Masking rate for the dataset.")
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges osr not together with atoms')
    parser.add_argument('--alpha', type=float, default=1, help = "Alpha for the loss function")


    args = parser.parse_args()
    seed = args.seed

    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['init_lr'] = args.init_lr
    config['init_base_lr'] = args.init_base_lr
    # config['weight_decay'] = args.weight_decay
    config['gpu'] = args.gpu   
    config['model']['num_layer'] = args.num_layer
    config['model']['emb_dim'] = args.emb_dim
    config['model']['feat_dim'] = args.feat_dim
    config['model']['drop_ratio'] = args.dropout
    config['model']['pool'] = args.pool

    config['task_name'] = args.task_name
    config['dataset']['seed'] = seed
    config['dataset']['splitting'] = args.splitting


    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    config['task_name'] = config['task_name'].lower()
    

    config = get_config(config)
    
    print(config)

    results_list = []
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') # 시간을 지정 

    # for target in target_list:
    #     config['dataset']['target'] = target
    #     result = main(config)

    result = main(config)

    columns = ['Time', 'Seed','result']
    

    results_list.append([current_time, seed,  result])


    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list, columns=columns)

    fn_base = 'experiments/{}_{}'.format(config['fine_tune_from'], config['task_name'])

    if args.splitting == 'random':
        fn_base += '_random'
    elif args.splitting == 'scaffold':
        fn_base += '_scaffold'
    elif args.splitting == 'balanced_scaffold':
        fn_base += '_balanced_scaffold'

    
    fn_base += '_random_masking'
    
    fn_base += savefilename

    fn = f"{fn_base}.csv"    

    file_path = fn
    
    if os.path.exists(file_path):
        # 기존 파일 읽기
        existing_df = pd.read_csv(file_path)
        # 새로운 데이터 추가
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df


    combined_df.to_csv(
        file_path, 
        index=False, 
    )