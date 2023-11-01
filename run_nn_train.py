import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json, os, pickle
# from collections import defaultdict
from joblib import Parallel, delayed

# from tqdm import tqdm
# import itertools

# import wandb
from nnn import fileio
from nnn import train_nn as tnn

### MODIFY HERE ###
fixed_pclass = ['hairpin_size', 'interior_size', 'bulge_size', 'hairpin_triloop', 'hairpin_tetraloop', 'terminal_mismatch', 'stack']
config = dict(
    use_train_set_ratio = 0.1,
    fit_method = 'svd',
    feature_method = 'get_feature_list',
    fit_intercept=False, 
    symmetry=False,
    sep_base_stack=True,
    fix_some_coef=True,
    fixed_pclass = fixed_pclass,
    test_mode = 'val', # {'val', 'test'}
    use_model_from = 'lr_dict', # {'lr_dict', 'json'}
    )

myrange = [.1, .2, .75, 1.0]

tags = ['nupack test']
### END MODIFY ###
def pipeline_fun(config, ratio):
    config.update(dict(use_train_set_ratio=ratio))
    tnn.model_pipeline(config, tags=tags)

n = len(myrange)
Parallel(n_jobs=n)(delayed(pipeline_fun)(config, ratio) for ratio in myrange)
