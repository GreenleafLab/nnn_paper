
"""
Define the training pipeline
"""
import os
import torch
os.environ['TORCH'] = torch.__version__
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, sys
import sklearn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import wandb

kB = 0.0019872 # Bolzman constant
C2T = 273.15 # conversion from celsius to kalvin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pandas.core.frame import properties
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, BatchNorm, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import Set2Set


def dotbracket2edgelist(dotbracket_str:str, 
                        edge_feature:bool=True):
    
    assert isinstance(dotbracket_str, str), f'{dotbracket_str} is not a string'
    assert dotbracket_str.count('(') == dotbracket_str.count(')'), \
        'Number of "(" and ")" should match in %s' % dotbracket_str

    # Backbone edges
    N = len(dotbracket_str)
    edge_5p_list = [[i, i+1] for i in range(N-1)]

    # Hydrogen bonds
    edge_hbond_list = []
    flag3p = N - 1
    for i,x in enumerate(dotbracket_str):
        if x == '(':
            for j in range(flag3p, i, -1):
                if dotbracket_str[j] == ')':
                    edge_hbond_list.append([i, j])
                    flag3p = j - 1
                    break

    # 5to3, 3to5, bidirectional hbond
    edge_list = edge_5p_list + [e[::-1] for e in edge_5p_list] + edge_hbond_list + [e[::-1] for e in edge_hbond_list]
    
    if edge_feature:
        n_backbone, n_hbond = len(edge_5p_list), len(edge_hbond_list)
        edge_attr = np.zeros((len(edge_list), 3), dtype=int)
        edge_attr[:n_backbone, 0] = 1
        edge_attr[n_backbone:n_backbone*2, 1] = 1
        edge_attr[-2 * n_hbond:, 2] = 1
        return edge_list, edge_attr
    else:
        return edge_list


def onehot_nucleotide(seq_str):
    map_dict = dict(A=0, T=1, C=2, G=3)
    N = len(seq_str)
    encode_arr = np.zeros((N, 4))
    for i,x in enumerate(seq_str.upper()):
        encode_arr[i, map_dict[x]] = 1
    return encode_arr

def norm_dH(dH):
    return (dH + 26) / 10

def unorm_dH(dH_norm):
    return dH_norm * 10 - 26

def norm_Tm(Tm):
    return (Tm - 40) / 7

def unorm_Tm(Tm_norm):
    return Tm_norm * 7 + 40

def row2graphdata(row):
    edge_list, edge_feat = dotbracket2edgelist(row['TargetStruct'])
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long) #int64
    edge_attr = torch.tensor(edge_feat, dtype=torch.float)
    x = torch.tensor(onehot_nucleotide(row['RefSeq']), dtype=torch.float)
    y = torch.tensor([norm_dH(row['dH']), norm_Tm(row['Tm'])], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

assert np.allclose(onehot_nucleotide('AT'), np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))

class NNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(os.path.join(self.raw_dir, self.raw_file_names[1]), 'r') as fh:
            self.data_split_dict = json.load(fh)

        self.arr = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), index_col=0)
        self.seqid = self.arr.index

    @property
    def raw_file_names(self):
        return ['arr.csv', 'train_val_test_split.json']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def train_set(self):
        ind = np.searchsorted(self.seqid, 
                                    self.data_split_dict['train_ind'])
        return self.index_select(ind)

    @property
    def val_set(self):
        ind = np.searchsorted(self.seqid, 
                              self.data_split_dict['val_ind'])
        return self.index_select(ind)

    @property
    def test_set(self):
        ind = np.searchsorted(self.seqid, 
                              self.data_split_dict['test_ind'])
        return self.index_select(ind)

    def process(self):
        print(self.raw_dir)
        self.arr = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), index_col=0)
        data_list = [row2graphdata(row) for _,row in self.arr.iterrows()]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class NNNDatasetCorrected(NNNDataset):
    @property
    def raw_file_names(self):
        return ['arr_corrected_inplace.csv', 'train_val_test_split.json']
    

def sweep_model():
    with wandb.init() as run:
        config = wandb.config
        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)

        # and use them to train the model
        train(model, train_loader, test_loader, criterion, optimizer, config)

        # and test its final performance
        test(model, train_loader, test_loader)

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="NNN_GNN", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, test_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, train_loader, test_loader)

    return model


def make(config):
    # Make the data
    torch.manual_seed(12345)
    if config['dataset'] == 'NNN_v0':
        dataset = NNNDataset(root='/mnt/d/data/nnn/')
    elif config['dataset'] == 'NNN_v0.1':
        dataset = NNNDatasetCorrected(root='/mnt/d/data/nnn/')
    has_gpu = torch.cuda.is_available()
    train_loader = DataLoader(dataset.train_set, batch_size=config['batch_size'],
                            shuffle=True, pin_memory=has_gpu)
    test_loader = DataLoader(getattr(dataset, config['mode']+'_set'), batch_size=config['batch_size'],
                            shuffle=False, pin_memory=has_gpu)

    # Make the model
    if config['architecture'] == 'GraphTransformer':
        model = GTransformer(config).to(device)
    # elif config['architecture'] == 'GAT':
    #     model = GAT(config['hidden_channels']).to(device)
    else:
        raise 'Check `architecture` in config dictionary!'

    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, test_loader, criterion, optimizer


class GTransformer(torch.nn.Module):
    def __init__(self, config):
        super(GTransformer, self).__init__()
        torch.manual_seed(12345)
        num_node_features = 4
        num_edge_features = 3
        num_params = 2
        num_heads = 1
        self.graphconv_dropout = config['graphconv_dropout']
        self.linear_dropout = config['linear_dropout']

        conv_list = ([
            TransformerConv(num_node_features, config['hidden_channels'],
                               heads=num_heads, edge_dim=num_edge_features, dropout=self.graphconv_dropout)] +
            [TransformerConv(config['hidden_channels'], config['hidden_channels'],
                               heads=num_heads, edge_dim=num_edge_features, dropout=self.graphconv_dropout)] *
            (config['n_graphconv_layer'] - 1))
        self.convs = ModuleList(conv_list)

        # self.norm = BatchNorm(in_channels=config['hidden_channels'])

        linear_list = []
        self.concat = config['concat']
        if config['concat']:
          n_pool = config['hidden_channels'] * config['n_graphconv_layer']
        else:
          n_pool = config['hidden_channels']
        if config['pooling'] == 'Set2Set':
            self.aggr = Set2Set(n_pool, 
                                processing_steps=config['processing_steps'])
            linear_list.append(Linear(2*n_pool, config['linear_hidden_channels'][0]))
        elif config['pooling'] == 'global_add_pool':
            self.aggr = global_add_pool
            linear_list.append(Linear(config['hidden_channels'], config['linear_hidden_channels'][0]))
        else:
            raise 'Invalid config.pooling %s' % config['pooling']
        
        if config['n_linear_layer'] >= 2:
            linear_list.extend([Linear(config['linear_hidden_channels'][i - 1], 
                                    config['linear_hidden_channels'][i])
                                for i in range(1, config['n_linear_layer'])])
        linear_list.append(Linear(config['linear_hidden_channels'][-1], 
                                  num_params))
        self.linears = ModuleList(linear_list)

        # self.trace = []

    def forward(self, x, edge_index, edge_attr, batch):
        self.trace = []
        # 1. Obtain node embeddings
        for i, l in enumerate(self.convs):
            x = l(x, edge_index, edge_attr)
            self.trace.append(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.graphconv_dropout, training=self.training)

        # 2. Pooling layer
        if self.concat:
          x = torch.cat(self.trace, dim=1)
        x = self.aggr(x, batch)

        # 3. Apply a final regressor
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.linear_dropout, training=self.training)

        x = torch.flatten(x)
        
        return x

    @property
    def n_parameters(self):
        n_param = 0
        for param in model.parameters():
            n_param += np.prod(np.array(param.shape))

        print(n_param)

def train_epoch(model, train_loader, criterion, optimizer, config):
    """
    Train one epoch, called by train()
    """

    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.

        out = model(data.x.to(device), data.edge_index.to(device), 
                    data.edge_attr.to(device), data.batch.to(device))  # Perform a single forward pass.

        loss = criterion(out, data.y.to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def get_loss(loader, model):
    """
    Gets the loss at log points during training
    """
    model.eval()

    rmse = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), 
                    data.edge_attr.to(device), data.batch.to(device)) 
        rmse += float(((out - data.y.to(device))**2).sum())  # Check against ground-truth labels.
    return np.sqrt(rmse / len(loader.dataset))  # Derive ratio of correct predictions.


def unorm(arr):
    arr[:,0] = unorm_dH(arr[:,0]) 
    arr[:,1] = unorm_Tm(arr[:,1])
    return arr


def model_pred(model, data, device='cuda:0'):
    """
    handy function to predict one sequence and convert the result to (1,2) np.array
    """
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device), 
                data.edge_attr.to(device), data.batch.to(device))
    return out.to('cpu').detach().numpy().reshape(-1, 2)


def get_truth_pred(loader, model):

    y = np.zeros((0,2))
    pred = np.zeros((0,2))
    model.eval()
    for i, data in enumerate(loader):
        y = np.concatenate((y, data.y.detach().numpy().reshape(-1,2)), axis=0)
        out = model_pred(model, data)
        pred = np.concatenate((pred, out), axis=0)

    y = unorm(y)
    pred = unorm(pred)

    return dict(y=y, pred=pred)

def plot_truth_pred(result, ax, param='dH', title='Train'):
    color_dict = dict(dH='c', Tm='cornflowerblue', dG_37='teal', dS='steelblue')
    if param == 'dH':
        col = 0
        lim = [-55, -5]
    elif param == 'Tm':
        col = 1
        lim = [20, 60]

    c = color_dict[param]

    y, pred = result['y'][:, col], result['pred'][:, col]
    ax.scatter(y, pred, c=c, marker='D', alpha=.05)
    rmse = np.sqrt(np.mean((y - pred)**2))
    mae = np.mean(np.abs(y - pred))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('measured ' + param)
    ax.set_ylabel('predicted ' + param)
    ax.set_title('%s: RMSE = %.3f, MAE = %.3f' % (title, rmse, mae))

    return rmse, mae


def train(model, train_loader, test_loader, criterion, optimizer, config):
    """
    Calls train_epoch() and logs
    """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    every_n_epoch = 10
    for epoch in range(config['n_epoch']):
        train_epoch(model, train_loader, criterion, optimizer, config)
        if epoch % every_n_epoch == 0:
            train_rmse = get_loss(train_loader, model)
            test_rmse = get_loss(test_loader, model)
            wandb.log({"train_rmse": train_rmse,
                    "test_rmse": test_rmse})
            print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')


def get_n_param(model):
    n_param = 0
    for param in model.parameters():
        n_param += np.prod(np.array(param.shape))
    return n_param

def test(model, train_loader, test_loader):

    train_result = get_truth_pred(train_loader, model)
    test_result = get_truth_pred(test_loader, model)

    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    _ = plot_truth_pred(train_result, ax[0,0], param='dH')
    _ = plot_truth_pred(train_result, ax[0,1], param='Tm')
    dH_rmse, dH_mae = plot_truth_pred(test_result, ax[1,0], param='dH', title='Validation')
    Tm_rmse, Tm_mae = plot_truth_pred(test_result, ax[1,1], param='Tm', title='Validation')
    wandb.log({'fig': fig})
    wandb.run.summary["dH_rmse"] = dH_rmse
    wandb.run.summary["dH_mae"] = dH_mae
    wandb.run.summary["Tm_rmse"] = Tm_rmse
    wandb.run.summary["Tm_mae"] = Tm_mae
    wandb.run.summary["n_parameters"] = get_n_param(model)

    plt.show()