import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from . import util
import re

from ipynb.draw import draw_struct

def find_mm_parent(seq, struct, arr, which_side='both'):
    def get_parent_candidates(seq, struct):
        """
        parent1 is mutated at the 3' side, parent2 at the 5' side
        """
        assert which_side in {'both', '5p', '3p'}
        
        if '.(.' in struct:
            return None, None
        
        
        idxmm = (('('+struct).find('((.(') + 1, (struct+')').find(').))') + 1)
        if idxmm[0] == 0 or idxmm[1] == 1:
            return None, None
        else:
            parent1 = util.replace_at_index(seq, bp_dict[seq[idxmm[0]]], idxmm[1])
            mm1 = bp_dict[seq[idxmm[0]]] + '>' + seq[idxmm[1]]
            parent2 = util.replace_at_index(seq, bp_dict[seq[idxmm[1]]], idxmm[0])
            mm2 = bp_dict[seq[idxmm[1]]] + '>' + seq[idxmm[0]]
            context = seq[idxmm[0]-1:idxmm[0]+2] + seq[idxmm[1]-1:idxmm[1]+2]
            
            if which_side == 'both':
                return {parent1: mm1, parent2: mm2}, context
            elif which_side == '5p':
                return {parent2: mm2}, context
            elif which_side == '3p':
                return {parent1: mm1}, context
        
    bp_dict = dict(A='T', C='G', T='A', G='C')
    parent_candidates, context = get_parent_candidates(seq, struct)
    
    if parent_candidates is None:
        return np.nan
    
    idx_parent = []
    mm = []
    for candidate in parent_candidates:
        try:
            idx_parent.append(arr.RefSeq.tolist().index(candidate))
            mm.append(parent_candidates[candidate])
        except:
            continue
            
    if len(idx_parent) == 0:
        return np.nan
    else:
        return (arr.iloc[idx_parent].index.tolist(), mm, context)


def find_hp_parent(seq, struct, arr, ref_loop_seq='AAAG'):
    def get_parent_candidate(seq, struct):

        idxhp = (('('+struct).find('((..') + 1, (struct+')').find('..))') + 1)
        if idxhp[0] == 0 or idxhp[1] == 1:
            return None, None
        else:
            parent = seq[:idxhp[0]] + ref_loop_seq + seq[idxhp[1]+1:]
            loop = seq[idxhp[0]:idxhp[1]+1]

            return parent, loop
        
    parent, loop = get_parent_candidate(seq, struct)
    
    if parent is None:
        return np.nan
    
    try:
        idx_parent = arr.RefSeq.tolist().index(parent)
    except:
        return np.nan
            
    return (arr.iloc[idx_parent].name, loop)


def find_bg_parent(seq, struct, arr):
    def get_parent_candidate(seq, struct):

        pattern = '(\({1}[.]+\({1})|(\){1}[.]+\){1})'
        match = re.search(pattern, struct)
       
        if match is None:
            return None, None
        else:
            start = match.span()[0] + 1
            end = match.span()[1] - 1
            parent = util.replace_at_index(seq, '', start, end)
            loop = seq[start:end]

            return parent, loop
        
    parent, loop = get_parent_candidate(seq, struct)
    
    if parent is None:
        return np.nan
    
    try:
        idx_parent = arr.RefSeq.tolist().index(parent)
    except:
        return np.nan
            
    return (arr.iloc[idx_parent].name, loop)




def find_mm_parent_df(mm_df, arr, which_side):
    df = mm_df.copy()
    df['parent'] = df.apply(lambda row: find_mm_parent(row.RefSeq, row.TargetStruct, arr, which_side=which_side), axis=1)
    df.dropna(subset=['parent'], inplace=True)
    
    ## clean output col ##
    df['mismatch'] = df.parent.apply(lambda x: x[1][0])
    df['context'] = df.parent.apply(lambda x: x[2])
    df['parent'] = df.parent.apply(lambda x: x[0][0])

    ## get parent info ##
    df = df.set_index('parent').join(arr[['dG_37', 'dG_37_se']], rsuffix='_parent').reset_index()

    ## add info ##
    df['flank'] = df.context.apply(lambda x: x[0]+x[-1]+'x'+x[2:4])

    df['ddG_37'] = (df.dG_37 - df.dG_37_parent).astype(float)
    df['ddG_37_se'] = (df.dG_37_se + df.dG_37_se_parent).astype(float)
    
    return df
    
def find_hp_parent_df(hp_df, arr, ref_loop_seq):
    df = hp_df.copy()
    n = len(df.TargetStruct.values[0])
    df['parent'] = df.apply(lambda row: find_hp_parent(row.RefSeq, row.TargetStruct, arr, ref_loop_seq=ref_loop_seq), axis=1)

    df['loop'] = df.parent.apply(lambda x: np.nan if isinstance(x, float) else x[1])
    df['parent'] = df.parent.apply(lambda x: np.nan if isinstance(x, float) else x[0])

    hp_parents = np.unique([x for x in df.parent])
    for p in hp_parents:
        try:
            assert p in arr.index
        except:
            print(hp_parents) 

    df = df.set_index('parent').join(arr[['dG_37', 'dG_37_se']], rsuffix='_parent').reset_index()

    ## add info ##
    df['ddG_37'] = (df.dG_37 - df.dG_37_parent).astype(float)
    df['ddG_37_se'] = (df.dG_37_se + df.dG_37_se_parent).astype(float)
    
    return df


def find_bg_parent_df(hp_df, arr):
    df = hp_df.copy()
    # n = len(df.TargetStruct.values[0])
    df['parent'] = df.apply(lambda row: find_bg_parent(row.RefSeq, row.TargetStruct, arr), axis=1)

    df['loop'] = df.parent.apply(lambda x: np.nan if isinstance(x, float) else x[1])
    df['parent'] = df.parent.apply(lambda x: np.nan if isinstance(x, float) else x[0])

    hp_parents = np.unique([x for x in df.parent])
    for p in hp_parents:
        try:
            assert p in arr.index
        except:
            print(hp_parents) 

    df = df.reset_index().set_index('parent').join(arr[['dG_37', 'dG_37_se']], rsuffix='_parent').reset_index()

    ## add info ##
    df['ddG_37'] = (df.dG_37 - df.dG_37_parent).astype(float)
    df['ddG_37_se'] = (df.dG_37_se + df.dG_37_se_parent).astype(float)
    
    return df
 
 
def get_single_nt_contributions_df(df, y_col='ddG_37'):

    n = len(df.RefSeq.values[0])
    anova_df = get_pos_df(df, n=n, y_col=y_col)

    contributions = dict()
    for pos_col in anova_df.columns[:-1]:
        contributions[pos_col] = get_sum_sq_explained(anova_df, pos_col, y_col=y_col)
        
    return contributions
    
def plot_contributions_on_struct(contributions, target_struct,
                                 log_scale=True, clip_max=np.inf, label_num=True, ax=None):
    n = len(target_struct)
    c_log = np.zeros(n)
    c = np.zeros(n)
    for key,value in contributions.items():
        p = int(key.replace('p',''))
        c[p] = value
        
    if log_scale:
        c = np.log(c + 1)
        
    c_log = c.copy()
    c = np.clip(c, 0, clip_max)
    
    if ax is None:
        _, ax = plt.subplots()

    if label_num:
        text_label = ['%.1f'%x for x in c_log]
    else:
        text_label = ' ' * n
        
    draw_struct(text_label, target_struct, c=c, cmap='viridis', ax=ax)
    
    
def calc_and_plot_hp_contributions_on_struct(hp_df, arr, y_col, ref_loop_seq='', **kwargs):
    """
    Args:
        hp_df - should all have the same 2nd structure!!!
    """
    if y_col == 'ddG_37':
        assert len(ref_loop_seq) > 0
        df = find_hp_parent_df(hp_df, arr, ref_loop_seq=ref_loop_seq)
    else:
        df = hp_df.copy()
        
    target_struct = df.TargetStruct.values[0]
    contributions = get_single_nt_contributions_df(df, y_col=y_col)
    plot_contributions_on_struct(contributions, target_struct, **kwargs)
    
def get_santalucia_mm_ddG(row, sl_mm):
    def unmutate(nt):
        return nt.replace(mismatch[2], mismatch[0])
        
    context = row['context']
    mismatch = row['mismatch']
    
    nn1 = context[0] + context[1] + '_' + context[4] + context[5]
    nn2 = context[3] + context[4] + '_' + context[1] + context[2]
    dG_mm = sl_mm.loc[nn1, 'dG_37'] + sl_mm.loc[nn2, 'dG_37']
    
    wc_nn1 = nn1[0] + unmutate(nn1[1]) + '_' + unmutate(nn1[3]) + nn1[4]
    wc_nn2 = nn2[0] + unmutate(nn2[1]) + '_' + unmutate(nn2[3]) + nn2[4]
    dG_wc = sl_mm.loc[wc_nn1, 'dG_37'] + sl_mm.loc[wc_nn2, 'dG_37']
    
    return dG_mm - dG_wc


def get_explained_var(y, pred):
    var_dict = dict()

    var_dict['total'] = np.var(y)
    var_dict['unexplained'] = np.var((y - pred))
    var_dict['explained'] =  var_dict['total'] - var_dict['unexplained']
    var_dict['explained_ratio'] = var_dict['explained'] / var_dict['total']
    var_dict['unexplained_ratio'] = var_dict['unexplained'] / var_dict['total']
    return var_dict

def bootstrap_ddG_37_sl(tmp_df):
    n = 1 # num sample per datapoint
    bootstrap = np.zeros((len(tmp_df), n))

    for i,row in tmp_df.iterrows():
        mu = row.ddG_37_sl
        sigma = row.ddG_37_se
        bootstrap[i,:] = np.random.normal(mu, sigma, size=n)
        
    return bootstrap.flatten()


def ci_len(x):
    return np.percentile(x, 97.5) - np.percentile(x, 2.5)
    
def get_sum_sq_explained(df, group_col, y_col=None):
    if y_col is None:
        y_col = df.columns[-1]
        
    y = df[y_col]
    
    formula = f'{y_col} ~ C({group_col})'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    between_group_variance = anova_table['sum_sq'][0]
    
    return between_group_variance


def get_extra_explained_cp(anova_df, c1, c2, y_col=None):
    '''
    Difference in R2
    '''
    df = anova_df.copy()
    if y_col is None:
        y_col = df.columns[-1]
    
    y = df[y_col]
    cp = f'{c1}_{c2}'
    df[cp] = df.apply(lambda row: row[c1] + row[c2], axis=1)

    formula = f'{y_col} ~ C({c1}) + C({c2})'
    model_sep = ols(formula, data=df).fit()

    formula = f'{y_col} ~ C({cp})'
    model_joint = ols(formula, data=df).fit()

    return model_joint.rsquared - model_sep.rsquared
    
    
def get_pos_df(df, n, y_col='ddG_37'):
    pos_cols = ['p%d'%i for i in range(n)]

    for i in range(n):
        df['p%d'%i] = [x[i] for x in df.RefSeq]

    for p in pos_cols:
        if len(np.unique(df[p])) < 4:
            df = df.drop(columns=p)

    pos_cols = [x for x in df.columns if x.startswith("p")]
    anova_df = df[pos_cols+[y_col]]
    
    return anova_df
    
    
def get_variances(y, sigma, y_hat, 
                  regress_sigma=False, sigma_model=None, return_model=False,
                  verbose=True):
    """
    Args:
        sigma_model - LinearRegression, if None fit afresh
    """
    n = len(y)

    var = dict()
    var['tot'] = np.sum((y - np.mean(y))**2)/ n
    var['model'] = np.sum((y_hat - np.mean(y_hat))**2) / n
    
    if regress_sigma:
        if sigma_model is None:
            model = LinearRegression()
            model.fit(y.reshape(-1,1), sigma)
        else:
            model = sigma_model
        
        var['tech'] = np.sum((model.predict(y.reshape(-1,1)))**2) / n
        
        if verbose:
            a = model.coef_[0]
            b = model.intercept_
            print('sigma = %.2fy + %.2f\n' % (a,b))
            
    else:
        var['tech'] = np.sum(sigma**2) / n
        
    var['bio'] = var['tot'] - var['tech']
    var['res'] = np.sum((y - y_hat)**2) / n
    var['?'] = var['res'] - var['tech']
    
    if verbose:
        print('\t'.join(var.keys()))
        print('%.2f\t'*len(var) % tuple(var.values()))
    
    if return_model:
        return var, model
    else:
        return var
        
        
def plot_variances(var, x, ax, label_text=False, legend=False):
    offset = .18
    palette = np.array([
        [42,182,115],
        [246,182,102],
        [230,194,220],
        [214,0,0],
    ]) / 256.
    kwargs = dict(width=.36, edgecolor='k')
    ax.bar(x-offset, height=var['tech'], bottom=var['bio'], label='technical', color=palette[1], **kwargs)
    ax.bar(x-offset, var['bio'], bottom=0, label='biological', color=palette[2], **kwargs)
    
    explained_var = var['bio'] - var['?']
    if explained_var > 0:
        ax.bar(x+offset, explained_var, label='explained', color=palette[0], **kwargs)
    else:
        ax.bar(x+offset, -explained_var, label='added by model', color=palette[3], **kwargs)
        
    if label_text:
        ax.text(x, var['bio']+var['tech']/2, 'technical', ha='center')
        ax.text(x, var['bio']/2, 'unexplained', ha='center')
        ax.text(x, (var['bio'] - var['?'])/2, 'technical', ha='center')

    if legend:
        plt.legend()
        
    sns.despine()