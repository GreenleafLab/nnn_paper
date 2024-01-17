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
from pprint import pprint

kB = 0.0019872 # Bolzman constant
C2T = 273.15 # conversion from celsius to kalvin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

sys.path.append('..')
from nnn.gnn import *


linear_hidden_channels=[128]
fixed_config = dict(
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
    # use_train_set_ratio=1,
    mode='test',
    architecture="GraphTransformer",
    concat=False)
fixed_config = {k:dict(value=v) for k,v in fixed_config.items()}
# sweep_configuration = {
#     "name": "fine tune before test",
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "test_rmse"},
#     "parameters": {
#         "processing_steps": {"max":7, "min":5, "distribution":"int_uniform"},
#         "graphconv_dropout": {"max": .005, "min": 0.004, "distribution": "uniform"},
#         "linear_dropout": {"max": .3, "min": .2},
#         "hidden_channels": {"max": 128, "min":98, "distribution":"int_uniform"},
#         "learning_rate": {"max": .004, "min": .002},
#         "batch_size": {"max": 1200, "min":600, "distribution":"int_uniform"},
#     },
# }
myrange = np.logspace(-2, 0, 10).tolist()
# myrange = [0.021544346900318832,0.046415888336127774,]
sweep_configuration = {
    "name": "log scale use_train_set_ratio sweep on test",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "test_rmse"},
    "parameters": {
        "use_train_set_ratio": {"values": myrange},
    },
}
sweep_configuration['parameters'].update(fixed_config)
pprint(sweep_configuration)

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="NNN_GNN")

wandb.agent(sweep_id, function=sweep_model, count=10)