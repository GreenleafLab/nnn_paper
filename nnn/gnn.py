
"""
Define the training pipeline
Do not import from `nnn` as those use a different environment
"""
import os
import torch
os.environ['TORCH'] = torch.__version__
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import json, os, sys
import sklearn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from pprint import pprint
from operator import itemgetter

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
    strand_break_ind = dotbracket_str.find('+')
        
    if strand_break_ind == -1:
        # hairpin
        edge_5p_list = [[i, i+1] for i in range(N-1)]
    else:
        # duplex
        dotbracket_str = dotbracket_str.replace('+', '')
        N -= 1
        edge_5p_list = [[i, i+1] for i in range(N-1) if \
                       (i != strand_break_ind - 1) and (i+1 != strand_break_ind)]
        
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
    """
    row['RefSeq'] is a list of 2 str for duplex but has to be joined into
    a single string as input to this function
    """
    map_dict = dict(A=0, T=1, C=2, G=3)
    N = len(seq_str)
    encode_arr = np.zeros((N, 4))
    for i,x in enumerate(seq_str.upper()):
        encode_arr[i, map_dict[x]] = 1
    return encode_arr

def norm_p(p, pname, sumstats_dict, method='normalize'):
    if sumstats_dict is None:
        return p
    
    if method == 'standardize':
        return (p - sumstats_dict[pname+'_mean']) / sumstats_dict[pname+'_std']
    elif method == 'normalize':
        return (p - sumstats_dict[pname+'_min']) / (sumstats_dict[pname+'_max'] - sumstats_dict[pname+'_min'])
    

def unorm_p(p, pname, sumstats_dict, method='normalize'):
    if sumstats_dict is None:
        return p
    
    if method == 'standardize':
        return p * sumstats_dict[pname+'_std'] + sumstats_dict[pname+'_mean']
    elif method == 'normalize':
        return p * (sumstats_dict[pname+'_max'] - sumstats_dict[pname+'_min']) + sumstats_dict[pname+'_min']

def calc_sumstats(df):
    sumstats_dict = dict(
        dH_mean = np.nanmean(df.dH),
        dH_std = np.nanstd(df.dH),
        dH_min = np.nanmin(df.dH),
        dH_max = np.nanmax(df.dH),
        Tm_mean = np.nanmean(df.Tm),
        Tm_std = np.nanstd(df.Tm),
        Tm_min = np.nanmin(df.Tm),
        Tm_max = np.nanmax(df.Tm)
    )
    return sumstats_dict    

def row2graphdata(row, sumstats_dict=None, method='normalize', y_type='dH_Tm'):
    edge_list, edge_feat = dotbracket2edgelist(row['TargetStruct'])
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long) #int64
    edge_attr = torch.tensor(edge_feat, dtype=torch.float)
    
    refseq = row['RefSeq']
    if isinstance(refseq, list):
        refseq = ''.join(refseq)
    elif '[' in refseq:
        refseq = ''.join(eval(refseq))
        
    x = torch.tensor(onehot_nucleotide(refseq), dtype=torch.float)
    
    if method is not None:
        # actually normalize y
        norm_fun = lambda p, pname: norm_p(p, pname, sumstats_dict, method=method)
    
    if y_type == 'dH_Tm':
        y = torch.tensor([norm_fun(row['dH'], 'dH'), 
                        norm_fun(row['Tm'], 'Tm')], dtype=torch.float)
    elif y_type == 'curve':
        # hard-coded, last 2 columns are refseq and targetstruct
        y = torch.tensor(row.values[:-2].astype(float), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

assert np.allclose(onehot_nucleotide('AT'), np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))

class NNNDataset(InMemoryDataset):
    """
    Abstract class
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(os.path.join(self.raw_dir, self.raw_file_names[1]), 'r') as fh:
            self.data_split_dict = json.load(fh)

        self.arr = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), index_col=0)
        self.seqid = self.arr.index
        self.sumstats_dict = calc_sumstats(self.arr.loc[self.data_split_dict['train_ind']])
        print('Initiating, summary statistis of the training set is:')
        pprint(self.sumstats_dict)

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

    def process_data_list(self, data_list):
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def process(self):
        print(self.raw_dir)
        self.arr = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), index_col=0)
        
        with open(os.path.join(self.raw_dir, self.raw_file_names[1]), 'r') as fh:
            self.data_split_dict = json.load(fh)
            
        self.sumstats_dict = calc_sumstats(self.arr.loc[self.data_split_dict['train_ind']])
        data_list = [row2graphdata(row, sumstats_dict=self.sumstats_dict, method='normalize') for _,row in self.arr.iterrows()]

        self.process_data_list(data_list)
        

class NNNDatasetdHTmV0(NNNDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.sumstats_dict = calc_sumstats(self.arr)

    @property
    def raw_file_names(self):
        return ['arr.csv', 'train_val_test_split.json']

    @property
    def processed_file_names(self):
        return ['data_v0.pt']

    def process(self):
        self.arr = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), index_col=0)
        self.sumstats_dict = calc_sumstats(self.arr.loc[self.data_split_dict['train_ind']])
        data_list = [row2graphdata(row, self.sumstats_dict, method='normalize') for _,row in self.arr.iterrows()]
        super().process_data_list(data_list)


class NNNDatasetdHTmCorrected(NNNDatasetdHTmV0):
    @property
    def raw_file_names(self):
        return ['arr_corrected_inplace.csv', 'train_val_test_split.json']

class NNNDatasetdHTmV1(NNNDatasetdHTmV0):
    """
    Predict fitted dH and Tm parameters
    Latest version
    """
        
    @property
    def raw_file_names(self):
        return ['arr_v1_n=27732.csv', 'data_split.json']
    
    @property
    def processed_file_names(self):
        return ['data_v1.pt']
    

""" p_unfold response variable"""    
class NNNCurveDataset(NNNDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return ['arr_p_unfold_n=30924.csv', 'data_split_p_unfold.json']
    
    @property
    def processed_file_names(self):
        return ['data_curve_v1.pt']
    
    def process(self):
        # No normalization as p_unfold is already normalized
        self.arr = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), index_col=0)
        data_list = [row2graphdata(row, method=None, y_type='curve') 
                        for _,row in self.arr.iterrows()]
        super().process_data_list(data_list)


class NNNDatasetWithDuplex(NNNDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset_name_list = ['arr', 'uv', 'lit_uv', 'ov']
        
    @property
    def raw_file_names(self):
        return ['combined_dataset.csv', 'combined_data_split.json']
    
    @property
    def processed_file_names(self):
        return ['combined_data_v0.pt']
    
    
        
    def get_data_split_subset(self, split='val', dataset_name='arr', sample=1):
        """
        Get a subset of the data by dataset name
        Args:
            dataset_name - str, {'arr', 'uv', 'lit_uv', 'ov'} as in `self.dataset_name_list`
            sample - float, 0~1 ratio to randomly sample the dataset
        """
        dataset_mask = self.arr.eval('dataset == "%s"' % dataset_name)
        split_ind = np.searchsorted(self.seqid, 
                self.data_split_dict[split+'_ind'])
        # indices for datapoints both in the dataset and the data split
        ind = list(set(split_ind) & set(np.where(dataset_mask)[0]))
        
        if sample < 1:
            np.random.seed(99)
            ind = np.random.choice(ind, size=int(sample*len(ind)), replace=False)
            
        return self.index_select(ind)
        
    @property
    def val_set(self):
        val_data = self.get_data_split_subset(split='val', dataset_name='arr')
        return val_data
        
    @property
    def test_set(self):
        return self.get_data_split_subset(split='test', dataset_name='arr')
        
    # def process(self):
    #     print('processing', self.raw_dir)
    #     self.arr = self.load_val_df(os.path.join(self.raw_dir, self.raw_file_names[0]))

    #     with open(os.path.join(self.raw_dir, self.raw_file_names[1]), 'r') as fh:
    #         self.data_split_dict = json.load(fh)
            
    #     self.sumstats_dict = calc_sumstats(self.arr.loc[self.data_split_dict['train_ind']])
    #     data_list = [row2graphdata(row, self.sumstats_dict, method='normalize') for _,row in self.arr.iterrows()]
    #     super().process_data_list(data_list)
        

def sweep_model():
    """
    Run a sweep
    """
    with wandb.init(project="NNN_GNN") as run:
        config = wandb.config
        # make the model, data, and optimization problem
        material_dict = make(config)
        model, train_loader, test_loader, criterion, optimizer = \
            itemgetter('model', 'train_loader', 'test_loader', 'criterion', 'optimizer')(material_dict)

        # and use them to train the model
        train(model, train_loader, test_loader, criterion, optimizer, config)

        # and test its final performance
        if 'test_loader_dict' in material_dict:
            test(model, train_loader, test_loader, material_dict['test_loader_dict'])
        else:
            test(model, train_loader, test_loader)

def run_saved_model(hyperparameters, 
                    test_result_fn=None,
                    log_wandb=True):
    """
    Run a single trained model from disk
    """
    if log_wandb:
        # tell wandb to get started
        with wandb.init(project="NNN_GNN", config=hyperparameters):
            # access all HPs through wandb.config, so logging matches execution!
            config = wandb.config
            saved_model_path = config['saved_model_path']

            # make the model, data, and optimization problem
            material_dict = make(config)
            model, train_loader, test_loader, criterion, optimizer = \
                itemgetter('model', 'train_loader', 'test_loader', 'criterion', 'optimizer')(material_dict)

            if 'test_loader_dict' in material_dict:
                extra_kwargs = dict(test_loader_dict=material_dict['test_loader_dict'])
            else:
                extra_kwargs = dict()
            
            if test_result_fn is not None:
                extra_kwargs['test_result_fn'] = test_result_fn
                
            # `model` is already "made" by `make()` but we want to load saved "state"
            print('Loading saved model from', saved_model_path)
            model.load_state_dict(torch.load(saved_model_path))
            model.eval()

            # and test its final performance
            test(model, train_loader, test_loader, **extra_kwargs)
    else:
        saved_model_path = config['saved_model_path']

        # make the model, data, and optimization problem
        material_dict = make(config)
        model, train_loader, test_loader = \
            itemgetter('model', 'train_loader', 'test_loader')(material_dict)

        if 'test_loader_dict' in material_dict:
            extra_kwargs = dict(test_loader_dict=material_dict['test_loader_dict'])
        else:
            extra_kwargs = dict()
        
        if test_result_fn is not None:
            extra_kwargs['test_result_fn'] = test_result_fn
            
        # `model` is already "made" by `make()` but we want to load saved "state"
        print('Loading saved model from', saved_model_path)
        model.load_state_dict(torch.load(saved_model_path))
        model.eval()

        # and test its final performance
        test(model, train_loader, test_loader, log_wandb=False, **extra_kwargs)
        
        

    

def model_pipeline(hyperparameters, save_model=False):
    """
    Run a single model
    """
    # tell wandb to get started
    with wandb.init(project="NNN_GNN", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        material_dict = make(config)
        model, train_loader, test_loader, criterion, optimizer = \
            itemgetter('model', 'train_loader', 'test_loader', 'criterion', 'optimizer')(material_dict)

        if 'test_loader_dict' in material_dict:
            extra_kwargs = dict(test_loader_dict=material_dict['test_loader_dict'])
        else:
            extra_kwargs = dict()
            
        # and use them to train the model
        train(model, train_loader, test_loader, criterion, optimizer, config, **extra_kwargs)

        # and test its final performance
        test(model, train_loader, test_loader, **extra_kwargs)
            
        if save_model:
            ## SAVING MODEL ##
            model_path = f'/mnt/d/data/nnn/models/gnn_state_dict_{wandb.run.name}.pt'
            torch.save(model.state_dict(), model_path)

    return model


def make(config):
    # Make the data
    torch.manual_seed(12345)
    root = '/mnt/d/data/nnn/'
    kwargs = dict(root=root)
    if config['dataset'] == 'NNN_v0':
        dataset = NNNDatasetdHTmV0(**kwargs)
    elif config['dataset'] == 'NNN_v0.1':
        dataset = NNNDatasetdHTmCorrected(**kwargs)
    elif config['dataset'] == 'NNN_v1':
        dataset = NNNDatasetdHTmV1(**kwargs)
    elif config['dataset'] == 'NNN_curve_v1':
        dataset = NNNCurveDataset(**kwargs)
    elif config['dataset'] == 'NNN_v2':
        dataset = NNNDatasetWithDuplex(root='/mnt/d/data/nnn/')
    else:
        raise ValueError(config['dataset'])
        
    ### Data Loaders ###
    has_gpu = torch.cuda.is_available()
    get_data_loader = lambda data_list, shuffle: DataLoader(data_list, batch_size=config['batch_size'], 
                                                            shuffle=shuffle, pin_memory=has_gpu)
    train_loader = get_data_loader(
        dataset.get_data_split_subset('train', 'arr', sample=config['use_train_set_ratio']),
        shuffle=True)
    test_loader = get_data_loader(getattr(dataset, config['mode']+'_set'), shuffle=False)
    train_loader.sumstats_dict = dataset.sumstats_dict
    test_loader.sumstats_dict = dataset.sumstats_dict
    
    
    # Make the model
    model = GNN(config).to(device)

    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'])
    
    # An EXTRA dict of validation datasets by dataset_name
    # only for NNN_v2 (with duplex) now
    if hasattr(dataset, 'dataset_name_list'):
        test_loader_dict = dict()
        for dataset_name in dataset.dataset_name_list:
            test_loader_dict[dataset_name] = get_data_loader(
                dataset.get_data_split_subset(split=config['mode'], dataset_name=dataset_name),
                shuffle=False
            )
            test_loader_dict[dataset_name].sumstats_dict = dataset.sumstats_dict
        
        return dict(model=model, train_loader=train_loader, test_loader=test_loader, 
                    test_loader_dict=test_loader_dict, criterion=criterion, optimizer=optimizer)
    
    return dict(model=model, train_loader=train_loader, test_loader=test_loader, 
                criterion=criterion, optimizer=optimizer)

class GNN(torch.nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        num_node_features = 4
        num_edge_features = 3
        num_params = 2
        num_heads = 1
        
        if config['dataset'] == 'NNN_curve_v1':
            dim_pred = 17
        else:
            dim_pred = 2
            
        self.graphconv_dropout = config['graphconv_dropout']
        self.linear_dropout = config['linear_dropout']

        if config['architecture'] == 'GCN':
            # GCN doesn't work yet
            GConv = GCNConv
            conv_list = ([
                GConv(num_node_features, config['hidden_channels'],
                        dropout=self.graphconv_dropout)] +
                [GConv(config['hidden_channels'], config['hidden_channels'],
                        dropout=self.graphconv_dropout)] *
                (config['n_graphconv_layer'] - 1))
        else:
            if config['architecture'] == 'GraphTransformer':
                GConv = TransformerConv
            elif config['architecture'] == 'GraphAttention':
                GConv = GATv2Conv
            else:
                raise 'Check `architecture` in config dictionary!'

            conv_list = ([
                GConv(num_node_features, config['hidden_channels'],
                                heads=num_heads, edge_dim=num_edge_features, dropout=self.graphconv_dropout)] +
                [GConv(config['hidden_channels'], config['hidden_channels'],
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
        
        if self.concat:
            self.trace = []
        # 1. Obtain node embeddings
        for i, l in enumerate(self.convs):
            x = l(x, edge_index, edge_attr)
            if self.concat:
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
    n = len(loader.dataset)
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), 
                    data.edge_attr.to(device), data.batch.to(device)) 
        error = float(((out - data.y.to(device))**2).sum())
        if np.isnan(error):
            print('edge_index',data.edge_index, 'batch', data.batch)
            n -= 1
        else:
            rmse += error  # Check against ground-truth labels.
            
    return np.sqrt(rmse / n)  # Derive ratio of correct predictions.


def unorm(arr, sumstats_dict):
    arr[:,0] = unorm_p(arr[:,0], 'dH', sumstats_dict) 
    arr[:,1] = unorm_p(arr[:,1], 'Tm', sumstats_dict)
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
    """
    Args:
        result - dict(y=y, pred=pred, aggr_list=aggr_list), dict[np.array (n,2)]
    """
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
            print(name, 'activation', activation[name].shape)
        return hook

    handle = model.aggr.register_forward_hook(get_activation('aggr'))
    aggr_list = []
    
    y = np.zeros((0,2))
    pred = np.zeros((0,2))
    model.eval()
    
    for i, data in enumerate(loader):
        y = np.concatenate((y, data.y.detach().numpy().reshape(-1,2)), axis=0)
        out = model_pred(model, data)
        pred = np.concatenate((pred, out), axis=0)
        
        aggr_list.append(activation['aggr'])

    # detach the hook
    handle.remove() 

    if hasattr(loader, 'sumstats_dict'):
        if len(loader.sumstats_dict) > 0:
            y = unorm(y, loader.sumstats_dict)
            pred = unorm(pred, loader.sumstats_dict)

    aggr_out = np.concatenate(aggr_list, axis=0)
    return dict(y=y, pred=pred, aggr_out=aggr_out)
    

def plot_truth_pred(result, ax, param='dH', title='Train'):
    """
    Args:
        result - dict(y=y, pred=pred), dict[np.array (n,2)]
    """
    get_dG_37 = lambda dH_Tm: dH_Tm[0] * (1 - (273.15 + 37) / (273.15 + dH_Tm[1]))
    color_dict = dict(dH='c', Tm='cornflowerblue', dG_37='teal', dS='steelblue')
    if param == 'dG_37':
        col = -1
        y = np.array([get_dG_37(dH_Tm) for dH_Tm in result['y']])
        pred = np.array([get_dG_37(dH_Tm) for dH_Tm in result['pred']])
        lim = [-7, 5]
    else:
        if param == 'dH':
            col = 0
            lim = [-55, -5]
        elif param == 'Tm':
            col = 1
            lim = [20, 60]

        y, pred = result['y'][:, col], result['pred'][:, col]
    
    c = color_dict[param]

    nan_mask = np.isnan(y)
    y, pred = y[~nan_mask], pred[~nan_mask]
    ax.scatter(y, pred, c=c, marker='D', alpha=.05)
    rmse = np.sqrt(np.nanmean((y - pred)**2))
    mae = np.nanmean(np.abs(y - pred))
    if np.isnan(rmse):
        print(y[:10], pred[:10])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('measured ' + param)
    ax.set_ylabel('predicted ' + param)
    ax.set_title('%s: RMSE = %.2f, MAE = %.2f' % (title, rmse, mae))
    sns.despine()

    return rmse, mae


def train(model, train_loader, test_loader, criterion, optimizer, config, test_loader_dict=None):
    """
    Calls train_epoch() and logs
    """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)   

    every_n_epoch = 10
    for epoch in range(config['n_epoch']):
        train_epoch(model, train_loader, criterion, optimizer, config)
        if epoch % every_n_epoch == 0:
            # Array train and test
            train_rmse = get_loss(train_loader, model)
            test_rmse = get_loss(test_loader, model)
            wandb.log(data={"train_rmse": train_rmse,
                            "test_rmse": test_rmse},
                      step=epoch)
            print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
            
            # External test, generalization ability
            if test_loader_dict is not None:
                # Temporarily not using `uv` as it's crap
                extra_test_result_dict = {dataset_name: get_truth_pred(loader, model) 
                            for dataset_name, loader in test_loader_dict.items() if dataset_name != 'uv'}
                
                for dataset_name, test_result in extra_test_result_dict.items():
                    log_extra_test_result(dataset_name, test_result, epoch)

def log_extra_test_result(dataset_name, test_result, epoch):
    """
    Log extra test results during training without plotting
    """
    Tm = test_result['y'][:,1]
    Tm_pred = test_result['pred'][:,1]
    assert len(Tm) == len(Tm_pred), 'len of Tm is %d, but len of Tm_pred is %d' % (len(Tm), len(Tm_pred))
    
    Tm_corr = pearsonr(Tm, Tm_pred)[0]
    Tm_bias = np.mean(Tm_pred) - np.mean(Tm)
    Tm_pred_adj = Tm_pred - Tm_bias
    Tm_mae = np.mean(np.abs(Tm - Tm_pred))
    Tm_mae_adj = np.mean(np.abs(Tm - Tm_pred_adj))
    
    wandb.log(data={
                    'Tm_corr_%s'%(dataset_name) : Tm_corr,
                    'Tm_bias_%s'%(dataset_name) : Tm_bias,
                    'Tm_mae_%s'%(dataset_name) : Tm_mae,
                    'Tm_mae_adj_%s'%(dataset_name) : Tm_mae_adj,},
              step=epoch,
              )



def get_n_param(model):
    n_param = 0
    for param in model.parameters():
        n_param += np.prod(np.array(param.shape))
    return n_param

def test(model, train_loader, test_loader, test_loader_dict=None, test_result_fn=None, log_wandb=True):

    train_result = get_truth_pred(train_loader, model)
    test_result = get_truth_pred(test_loader, model)

    if test_result_fn is not None:
        np.savez(test_result_fn, **test_result)

    fig, ax = plt.subplots(2, 3, figsize=(17,12))
    _ = plot_truth_pred(train_result, ax[0,0], param='dH')
    _ = plot_truth_pred(train_result, ax[0,1], param='Tm')
    _ = plot_truth_pred(train_result, ax[0,2], param='dG_37')
    dH_rmse, dH_mae = plot_truth_pred(test_result, ax[1,0], param='dH', title='Validation')
    Tm_rmse, Tm_mae = plot_truth_pred(test_result, ax[1,1], param='Tm', title='Validation')
    dG_37_rmse, dG_37_mae = plot_truth_pred(test_result, ax[1,2], param='dG_37', title='Validation')
    plt.tight_layout()
    if log_wandb:
        wandb.log({'fig': wandb.Image(fig)})
        wandb.run.summary["n_parameters"] = get_n_param(model)
    
        for m in ['dH_rmse', 'dH_mae', 'Tm_rmse', 'Tm_mae', 'dG_37_rmse', 'dG_37_mae']:
            wandb.run.summary[m] = eval(m)
    
    if test_loader_dict is not None:
        extra_test_result_dict = {dataset_name: get_truth_pred(loader, model) 
                                  for dataset_name, loader in test_loader_dict.items() if dataset_name != 'arr'}
        
        fig_extra, ax = plt.subplots(1, len(extra_test_result_dict), figsize=(9,3))
        i = 0
        for dataset_name, test_result in extra_test_result_dict.items():
            log_final_extra_test_results(dataset_name, test_result, Tm_only=True, ax=ax[i])
            i += 1
        sns.despine()
        plt.tight_layout() 
        if log_wandb:
            wandb.log({'extra test results': wandb.Image(fig_extra)})
        
        if test_result_fn is not None:
            # only saving aggr_out for lit_uv and ov here.
            # np.savez seems to support flat dictionaries
            extra_aggr_out = {k: v['aggr_out'] for 
                              k,v in extra_test_result_dict.items()}
            np.savez(test_result_fn.replace('.npz', '_extra.npz'), **extra_aggr_out)
        
    # plt.show()
    
    
def log_final_extra_test_results(dataset_name:str, test_result:dict, Tm_only:bool=True, ax=None):
    """
    Args:
        result - dict(y=y, pred=pred), dict[np.array (n,2)]
    """
    if Tm_only:
        result_df = pd.DataFrame(data=dict(
            Tm=test_result['pred'][:,1],
            Tm_pred=test_result['y'][:,1]))
                
        sns.scatterplot(data=result_df, x='Tm', y='Tm_pred', ax=ax,
                        marker='o', edgecolor='k', 
                        color=np.array([[252,223,120]])/256.)
        
        # Formatting the plot
        # Don't want dependency on util.py so copying the code here
        # Not the most elegant but who cares
        ax.set_title(dataset_name)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        lim_min = min(xlim[0], ylim[0])
        lim_max = max(xlim[1], ylim[1])
        new_lim = (lim_min, lim_max)
        ax.set_xlim(new_lim)
        ax.set_ylim(new_lim)
        Tm_locator = 10
        ax.xaxis.set_major_locator(MultipleLocator(Tm_locator))
        ax.yaxis.set_major_locator(MultipleLocator(Tm_locator))
        
        # Logging to wandb
        wandb.run.summary['Tm_mae_%s'%dataset_name] = np.nanmean(np.abs(result_df.Tm - result_df.Tm_pred))
        wandb.run.summary['Tm_corr_%s'%dataset_name] = pearsonr(result_df.Tm, result_df.Tm_pred)[0]
        fn = './out/%s_%s.csv' % (wandb.run.name, dataset_name)
        result_df.to_csv(fn)
        
        result_table = wandb.Table(dataframe=result_df)
        wandb.log({dataset_name+'_table': result_table})
        # result_table_artifact = wandb.Artifact("%s_artifact"%dataset_name, type="dataset")
        # result_table_artifact.add(result_table, dataset_name)
        # result_table_artifact.add_file(fn)
    else:
        #TODO log both dH and Tm, not really necessary right now
        pass