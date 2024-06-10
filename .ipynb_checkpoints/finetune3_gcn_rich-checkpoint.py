import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from dataset.dataset_test import MolTestDatasetWrapper
from dataset.get_config import get_config 
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 현재 파일의 전체 경로 및 파일명
full_path = __file__

# 현재 파일명만 추출
savefilename = os.path.basename(full_path)

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

from torch_geometric.utils import  scatter, softmax

def random_node(batch, ratio):
    # 각 배치의 노드 수 계산
    node_counts = scatter(torch.ones_like(batch), batch, reduce="sum")
    
    # 각 배치에서 선택할 노드의 수 계산
    select_counts = (ratio * node_counts).long()
    select_counts = torch.clamp(select_counts, min=1)
    
    # 각 노드에 대한 랜덤 점수 할당
    random_scores = torch.rand(batch.size(0), device=batch.device)

    
    # 각 노드에 대해 배치 ID와 함께 랜덤 점수를 결합
    combined_scores = batch.float() * 1e6 + random_scores
    
    # 결합된 점수를 기준으로 전체 노드를 정렬
    sorted_indices = torch.argsort(combined_scores)
    
    # 각 배치의 첫 번째 노드의 인덱스 찾기
    batch_first_indices = scatter(sorted_indices, batch[sorted_indices], reduce="min")
    
    # 각 노드의 배치 내 순위 계산
    batch_ranks = sorted_indices - batch_first_indices[batch]
    
    # 랜덤 점수가 높은 노드 선택
    selected_nodes = batch_ranks < select_counts[batch]

    true_indices = selected_nodes.nonzero().squeeze()


    return true_indices


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
        _, pred = model(data)  # [N,C]

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


    def _step(self, model, data, n_iter):
        # get the prediction
        _, pred = model(data)  # [N,C]


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
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
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

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

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
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss_total/train_loss_total', loss, epoch_counter)
                    self.writer.add_scalar('loss_end/train_loss_end', loss, epoch_counter)


                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    self.writer.add_scalar('accuracy/validation', valid_cls, epoch_counter)

                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    self.writer.add_scalar('accuracy/validation', valid_rgr, epoch_counter)
                    
                self.writer.add_scalar('loss_end/validation_loss_end', valid_loss, epoch_counter)
                self.writer.add_scalar('loss_total/validation_loss_total', valid_loss, epoch_counter)
                
                valid_n_iter += 1
        
        self._test(model, test_loader)

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

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

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
            predictions = torch.cat(predictions, dim=0).cpu().numpy()

            is_valid = labels**2 > 0
            roc_auc = get_roc_auc_score(labels, predictions, is_valid)
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

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
    parser.add_argument('--model_type', type=str, default='gin', help = "Type of GNN model gcn/gin")
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


    args = parser.parse_args()
    seed = args.seed

    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['init_lr'] = args.init_lr
    config['init_base_lr'] = args.init_base_lr
    # config['weight_decay'] = args.weight_decay

    config['model_type'] = args.model_type
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
    #     results_list.append([current_time, seed, target, result])

    result = main(config)
    results_list.append([current_time, seed,  result])

    columns = ['Time', 'Seed','result']

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list, columns=columns)

    fn_base = 'experiments/{}_{}'.format(config['fine_tune_from'], config['task_name'])

    if args.splitting == 'random':
        fn_base += '_random'
    elif args.splitting == 'scaffold':
        fn_base += '_scaffold'
    elif args.splitting == 'balanced_scaffold':
        fn_base += '_balanced_scaffold'
    
    
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