"""
Takes data from raw, process or read to processed, and output to knn_out.
'train' mode: 5-fold CV on train + val, including train on train + test on val
'test' mode: train on train + val, test on test
=======================================
root
    -- raw
        -- arr.csv
        -- train_val_test_split.json
    -- processed
        -- distance_mat_{train|val|test}-{train|val|test}_{seq|struct}.npy
        ...
    -- knn_out
=======================================
    
"""

from tqdm import tqdm
import sys, os, json
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from Levenshtein import distance
# from nnn import fileio, util


parser = argparse.ArgumentParser(description='Calculate distance matrix and train k-NN baseline')
parser.add_argument('-r', '--root', type=str,
                   help='Root dir with `raw` and `processed` subdir. `raw` has json file with \
                   data splitting indices, train_val_test_ind.json and arr.csv')
parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test', 'train-train'],
                   help='either "train" or "test"')
parser.add_argument('-k', type=int, default=8,
                    help='Number of neighbors k for k-NN')
parser.add_argument('--secondary_struct', default=False, action='store_true',
                    help='Whether to consider secondary structure for distance matrices or not')
parser.add_argument('--combine_metric', type=int, default=1, choices=[1, 2],
                    help='how to combine seq and struct distance, L1 or L2')
parser.add_argument('-n', '--num_core', default=1, type=int, metavar="N",
                   help='number of cores. default = 1 is without parallization')
        

def read_ml_data(datadir):
    """
    read arr df and train val test split for ML 
    """
    arr = pd.read_csv(os.path.join(datadir, 'combined_dataset.csv'), index_col=0)
    with open(os.path.join(datadir, 'combined_data_split.json'), 'r') as fh:
        data_split_dict = json.load(fh)
        
    return arr, data_split_dict
    
def save_fig(filename, fig=None):

    figdir, _ = os.path.split(filename)
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    if fig is None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        

def compute_edit_distance(str_arr1, str_arr2, num_core=1):
    """
    For dissimilar arrays of strings, compute pairwise edit distance.
    """
    def compute_one_row(str_i):
        distance_row = np.zeros(len(str_arr2), dtype=int)
        for j, str_j in enumerate(str_arr2):
            distance_row[j] = distance(str_i, str_j)
        return distance_row
        
    
    if num_core == 1:
        distance_mat = np.zeros((len(str_arr1), len(str_arr2)), dtype=int)
        for i,str_i in tqdm(enumerate(str_arr1)):
            distance_mat[i,:] = compute_one_row(str_i)
    elif num_core > 1:
        distance_rows = (Parallel(n_jobs=num_core, verbose=10)
                            (delayed(compute_one_row)(str_i)
                            for str_i in str_arr1))
        distance_mat = np.concatenate(distance_rows).reshape(len(str_arr1), len(str_arr2))

    return distance_mat
    

def get_distance_mat_y(root:str, mode:str, secondary_struct:bool, combine_metric:int,
                       num_core=1):
    """
    Load distance matrices, or make afresh if not made.
    test_distance - (n_test, n_train)
    """
    arr, data_split_dict = read_ml_data(os.path.join(root, 'raw'))
    
    train_ind = data_split_dict['train_ind']
    if mode == 'train':
        test_ind = data_split_dict['val_ind']
        test_lbl = 'val'
    elif mode == 'test':
        test_ind = data_split_dict['test_ind']
        test_lbl = 'test'
    elif mode == 'train-train':
        test_ind = data_split_dict['train_ind']
        test_lbl = 'train'
        
    # Find precomputed distance matrix from disk
    dist_fn = [os.path.join(root, 'processed', f'distance_mat_{test_lbl}_train_seq.npy')]
    if secondary_struct:
        dist_fn.append(os.path.join(root, 'processed', f'distance_mat_{test_lbl}_train_struct.npy'))
        n_mat = 2
    else:
        n_mat = 1
    
    dist_mat = [np.load(fn) for fn in dist_fn if os.path.exists(fn)]
    
    # Calculate distance mat if not on disk
    if len(dist_mat) < n_mat:
        dist_mat = [compute_edit_distance(arr.loc[test_ind, 'RefSeq'], arr.loc[train_ind, 'RefSeq'],
                                          num_core=num_core)]
        np.save(dist_fn[0], dist_mat[0])
        if secondary_struct:
            dist_mat.append(compute_edit_distance(arr.loc[test_ind, 'TargetStruct'], arr.loc[train_ind, 'TargetStruct'],
                                                  num_core=num_core))
            np.save(dist_fn[1], dist_mat[1])
            
    # Combine seq and struct distance if necessary
    if not secondary_struct:
        test_distance = dist_mat[0]
    else:
        if combine_metric == 1:
            test_distance = dist_mat[0] + dist_mat[1]
        elif combine_metric == 2:
            test_distance = np.sqrt(np.square(dist_mat[0]) + np.square(dist_mat[1]))
    
    # Keep train distance 0s for now
    n_train = len(train_ind)
    train_distance = np.zeros((n_train, n_train))

    # Targets
    y_train = arr.loc[train_ind, ['dH', 'Tm']]
    y_test = arr.loc[test_ind, ['dH', 'Tm']]
    
    return dict(train_distance=train_distance, test_distance=test_distance,
                y_train=y_train, y_test=y_test)


def get_knn_result(mydata, k:int, args=None):
    """
    Args:
        mydata - dict with keys 'train_distance', 'test_distance', 'y_train', 'y_test'
        k - k in k-NN
        args - dict, with keys 'root', 
            if not given, only return the rmse and y_pred_test
    """
    neigh = KNeighborsRegressor(n_neighbors=k, metric='precomputed', weights='distance')
    neigh.fit(mydata['train_distance'], mydata['y_train'])
    y_pred_test = neigh.predict(mydata['test_distance'])
    
    dH_rmse = np.sqrt(np.mean(np.square(mydata['y_test'].iloc[:,0] - y_pred_test[:,0])))
    dH_mae = np.mean(np.abs(mydata['y_test'].iloc[:,0] - y_pred_test[:,0]))
    Tm_rmse = np.sqrt(np.mean(np.square(mydata['y_test'].iloc[:,1] - y_pred_test[:,1])))
    Tm_mae = np.mean(np.abs(mydata['y_test'].iloc[:,1] - y_pred_test[:,1]))
    
    print('mode\tk\tsecondary_struct\tcombine_metric\tdH_rmse\tTm_rmse')
    print(f'{args.mode}\t{args.k}\t{args.secondary_struct}\t{args.combine_metric}\t{dH_rmse}\t{Tm_rmse}')
    if args is None:
        return dict(dH_rmse=dH_rmse, Tm_rmse=Tm_rmse, y_pred_test=y_pred_test)
    else:
        with open(os.path.join(args.root, 'knn_out', 'knn_rmse.csv'), 'a+') as fh:
            fh.write(f'{args.mode},{args.k},{args.secondary_struct},{args.combine_metric},{dH_rmse},{Tm_rmse}\n')

        test_result_df = pd.DataFrame(data=np.concatenate((mydata['y_test'].values, y_pred_test), axis=1), 
                               columns=['dH', 'Tm', 'dH_pred', 'Tm_pred'])
        test_result_df.to_csv(os.path.join(args.root, 'knn_out', 'knn_test_restul_df.csv'))
        
        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].scatter(mydata['y_test'].iloc[:,0], y_pred_test[:,0], 
                      s=70, c='c', alpha=0.05)
        ax[0].set_ylim([-60, 0])
        ax[0].set_xlim([-60, 0])
        ax[0].set_xlabel('measured dH')
        ax[0].set_ylabel('k-NN predicted dH')
        ax[0].set_title('RMSE = %.3f, MAE = %.3f' % (dH_rmse, dH_mae))
        sns.despine()

        ax[1].scatter(mydata['y_test'].iloc[:,1], y_pred_test[:,1], 
                      s=70, c='cornflowerblue', alpha=0.05)
        ax[1].set_ylim([20, 60])
        ax[1].set_xlim([20, 60])
        ax[1].set_xlabel('measured Tm')
        ax[1].set_ylabel('k-NN predicted Tm')
        ax[1].set_title('RMSE = %.3f, MAE = %.3f' % (Tm_rmse, Tm_mae))
        sns.despine()
        save_fig(os.path.join(args.root, 'knn_out', f'{args.mode}_{args.k}-NN_struct-{args.secondary_struct}_L{args.combine_metric}.pdf'))


if __name__ == "__main__":
    args = parser.parse_args()
        
    mydata = get_distance_mat_y(args.root, args.mode, 
                                args.secondary_struct, args.combine_metric,
                                num_core=args.num_core)
    result = get_knn_result(mydata, k=args.k, args=args)
        
    