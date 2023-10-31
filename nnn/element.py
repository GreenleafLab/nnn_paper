"""
To analyze 'elements' inserted into backbones in the style of sarah's paper
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from RiboGraphViz import LoopExtruder, StackExtruder
from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from scipy.spatial.distance import pdist

qual_palette = [
    (255,230,0),
    (214,0,0),
    (249,126,43),
    (25,137,30),
    (38,44,107),
    (142,4,150),
    (219,101,210),
    (131,164,255)]
qual_palette = np.array(qual_palette) / 256


def clean(x, seq_only=False):
    if seq_only:
        return x.replace(' ', '+').split(',')[0]
    else:
        return x.replace(' ','+').replace(',', '_')
        

    

def get_element(row, seq_only=False):
    element = None
    clean_fun = lambda row: clean(row, seq_only)
    
    if row['Series'] == 'WatsonCrick':
        stacks = StackExtruder(row.RefSeq, row.TargetStruct, stack_size=5)
        if isinstance(row.topScaffold, str):
            stack_ind = int(len(row.topScaffold) / 2)
        else:
            stack_ind = 0
        element = clean_fun(stacks[stack_ind])
        
    elif row['Series'] == 'TETRAloop':
        neighbor_bps = 1 #if row.ConstructType == 'NN' else 2
        element = clean_fun(LoopExtruder(row.RefSeq, row.TargetStruct, neighbor_bps=neighbor_bps)[0])
        
    elif row['Series'] in {'MisMatches', 'Bulges'}:
        element = None
        
    return element
    
    
def get_element_fingerprint(arr, scaffold_name='Scaffold', query=None, values=['dH', 'Tm'], 
                            flatten_col=True, seq_only=False, return_new_arr=False):
    """
    Returns the element by feature matrix and a mask for the nans
    """
    if query is not None:
        arr = arr.query(query)
        
    arr['element'] = arr.apply(lambda row: get_element(row, seq_only), axis=1)
    
    element_fingerprint = pd.pivot_table(arr, values=values, index=['element'], columns=[scaffold_name])
    
    if flatten_col:
        element_fingerprint.columns = ['_'.join(col) for col in element_fingerprint.columns.values]
    
    if return_new_arr:
        return element_fingerprint, arr
    else:    
        return element_fingerprint
    
    
def ward_cluster(mat_df, t, criterion='maxclust', split_Tm_dH=True, ax=None):
    mat_min, mat_max = np.nanmin(mat_df.values, axis=0).reshape(1,-1), np.nanmax(mat_df.values, axis=0).reshape(1,-1)
    mat_norm = (mat_df.values - mat_min) / (mat_max - mat_min)
    mat_norm = np.nan_to_num(mat_norm)
    
    mask = mat_df.isna()
    mat_df = mat_df.fillna(0)
    
    Z = ward(pdist(mat_norm))
    cluster_arr = fcluster(Z, t, criterion=criterion)
    cluster_color_list = [qual_palette[x] for x in cluster_arr]
    
    
    if split_Tm_dH:
        n = int(mat_df.shape[1] / 2)
        pltargs = dict(row_linkage=Z, row_colors=cluster_color_list, col_cluster=False)
        clustergrid = sns.clustermap(mat_df.iloc[:,:n], mask=mask.iloc[:,:n], 
                                    cmap='viridis', **pltargs)
        sns.clustermap(mat_df.iloc[:,n:2*n], mask=mask.iloc[:,n:2*n], cmap='cividis', **pltargs)
    else:
        pltargs = dict(row_linkage=Z, row_colors=cluster_color_list, col_cluster=True)
        clustergrid = sns.clustermap(mat_df, mask=mask, cmap='cividis', **pltargs)

    
    reordered_ind = clustergrid.dendrogram_row.reordered_ind
    
    return cluster_arr, reordered_ind