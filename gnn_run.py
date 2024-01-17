import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

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
import pprint

kB = 0.0019872 # Bolzman constant
C2T = 273.15 # conversion from celsius to kalvin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

sys.path.append('..')
from nnn.gnn import *

# When running saved model, only `saved_model_path` is actually used
# Everything else is just for logging purpose
linear_hidden_channels=[128]
config = dict(
    mode='test',
    n_epoch=200,
    params=['dH', 'Tm'], # not used by the program, for logging only
    norm_method='normalize', # not used by the program, for logging only
    hidden_channels=125,
    pooling='Set2Set',
    processing_steps=10,
    n_graphconv_layer=4,
    n_linear_layer=len(linear_hidden_channels),
    linear_hidden_channels=linear_hidden_channels,
    graphconv_dropout=0.012732466797412492,  
    linear_dropout=.25,#0.22559831635994448,
    batch_size=1842,
    learning_rate=0.0023788383566734047,
    dataset="NNN_v2", # NNN_v1 or NNN_v2 (+duplex) or NNN_curve_v1 (17 dim prediction)
    use_train_set_ratio=1,
    architecture="GraphTransformer",
    concat=False,
    saved_model_path='/path/to/saved/model/gnn_state_dict_ancient-sound-259.pt',
    )

# 3: Start the run

trained_model = run_saved_model(config, 
    test_result_fn='test_result_aggr_out.npz',
    log_wandb=False)

## SAVING MODEL ##
# model_path = f'/mnt/d/data/nnn/models/gnn_state_dict_{wandb.run.name}.pt'
# torch.save(trained_model.state_dict(), model_path)