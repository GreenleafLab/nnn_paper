from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import PassiveAggressiveRegressor

from .util import *
from . import plotting
from . import feature_list

R = 0.00198717 # gas constant in kcal/mol not J


def search_GU_pairs(seq):
    GU_counter=0
    for i, char in enumerate(seq[:7]):

        if char=='G':
            if seq[-1-i]=='T':
                GU_counter+=1
        elif char=='T':
            if seq[-1-i]=='G':
                GU_counter+=1
    return GU_counter


def get_model_metric(y, y_err, preds, n_feature):
    n = len(y)
    m = {}
    rss = np.sum(np.square(y - preds))
    m['bic'] = n*np.log(rss/n) + n_feature*np.log(n) 
    m['aic'] = n*np.log(rss/n) + 2*n_feature
    m['rmse'] = np.sqrt(np.mean(np.square(y - preds)))
    m['chi2'] = np.sqrt(np.mean(np.square(y - preds) / np.square(y_err)))
    m['dof'] = n - n_feature

    return m

def get_unknown_yerr_motif_se(X, res):
    """
    Calculate heteroskedasticity robust standard errors for fitted parameters
    """
    XT_X_1 = np.linalg.inv(X.T @ X)
    var_coef = XT_X_1 @ X.T @ np.diag(res**2) @ X @ XT_X_1
    return np.sqrt(np.diag(var_coef))


def solve_linear_regression(X, y, y_err, singular_value_rel_thresh=1e-15,
                        verbose=False):
    """
    Solves linear reggression with measurement error using SVD.
    With more plotting functions for sanity checks.
    Args:
        X is the feature matrix, may have a feature for intercept
        y - the measured parameters to be 
        y_err - a vector of the error of the measurements of each datapoint
    Return:

    """
    A = X / (y_err.reshape(-1,1))
    b = (y / y_err).reshape(-1,1)
    if verbose:
        print('Rank of the design matrix A is %d / %d'%(np.linalg.matrix_rank(A), A.shape[1]))
        print('Condition number of the design matrix A is %.2f'%np.linalg.cond(A))

    u,s,vh = np.linalg.svd(A, full_matrices=False)
    if verbose:
        # plot the singular values
        _, ax = plt.subplots()
        ax.plot(s / s[0], 'gx')
        ax.set_ylim(bottom=0)
        plt.show()
        
    s_inv = 1/s
    s_inv[s < s[0]*singular_value_rel_thresh] = 0
    coef_se = np.sqrt(np.sum((vh.T * s_inv.reshape(1,-1))**2, axis=1))
    # equivalent to np.linalg.pinv(A) @ b
    # doing this since we've already SVD'ed
    coef = (vh.T @ np.diag(s_inv) @ u.T @ b).flatten()
    
    if verbose:
        _, ax = plt.subplots(figsize=(10,6))
        ax.errorbar(np.arange(len(coef)), coef, coef_se, fmt='k.', capsize=3)
        if isinstance(X, pd.DataFrame):
            ax.set_xticks(np.arange(len(coef)))
            ax.set_xticklabels(X.columns, rotation=30)
        plt.show()
    
    return coef, coef_se
    
 


def get_feature_count_matrix(df, feature_method='get_stack_feature_list', feature_style='nnn', **kwargs):
        """
        Args:
            df - (n_variant, N) df
            feature_style - str, {'nnn', 'nupack'}, default to nnn.
                Determines token pattern for regex
            **kwargs - passed to `feature_method`
        Returns:
            feats - a (n_variant, n_feature) feature df
        """
        df['feature_list'] = df.apply((lambda row: getattr(feature_list, feature_method)(row, **kwargs)), axis=1)
        
        if feature_style == 'nnn':
            # token_pattern = r"\b[ATCGNxy().+_]+\s"
            token_pattern = r"\b[a-zA-Z0-9().+_-]+\s"
        elif feature_style == 'nupack':
            token_pattern = r"\b[a-z_]+\#[0-9ATCG]+\s"

        cv = CountVectorizer(token_pattern=token_pattern, lowercase=False)
        feats = pd.DataFrame.sparse.from_spmatrix(cv.fit_transform([' '.join(x)+' ' for x in df['feature_list']]),
                        index=df.index, columns=[x.strip() for x in cv.get_feature_names_out()])

        # Remove features that every construct contains and is not intercept
        if feature_style == 'nupack':
            intercept_symbol = 'intercept#0'
            if intercept_symbol in feats:
                intercept = feats.pop(intercept_symbol)
                feats['intercept#intercept'] = intercept
        else:
            intercept_symbol = 'intercept'
            
        for k in feats.keys():
            if len(feats[k].unique())==1 and k!=intercept_symbol:
                feats = feats.drop(columns=[k])
                
        # if intercept_symbol in feats.columns:
            

        return feats


def fit_linear_motifs(df, feature_method='get_stack_feature_list',
                      param='dG_37', err='_se_corrected', lim=None,
                      stack_sizes=[1,2,3], fit_intercept=False):

    fig, ax = plt.subplots(1, 3, figsize=(9,4), sharey=True)

    N_SPLITS = 5
    y = df[param]
    y_err = df[param + err]

    titles = {1:'N_N features (stacks only)', 2:'NN_NN (Nearest neighbor)', 3:'N(3)_N(3)', 4:'N(4)_N(4)', 5:'N(5)_N(5)'}

    coef_dfs=[]
    for i, stack_size in enumerate(stack_sizes):
        
        feats = get_feature_count_matrix(df, feature_method=feature_method, stack_size=stack_size)
        n_feature = feats.shape[1]
        #Perform linear regression fit
        mdl = Ridge(fit_intercept=fit_intercept)

        X = feats.values

        results = cross_validate(mdl, X, y, cv=N_SPLITS, return_estimator=True)
        coef_df = pd.DataFrame(columns=['motif', param])
        for x in results['estimator']:
            for j in range(len(feats.columns)):
                coef_df = coef_df.append({'motif': feats.columns[j], param: x.coef_[j]}, ignore_index=True)
        
        preds = cross_val_predict(mdl, X, y, cv=N_SPLITS)
        df['tmp_pred'] = preds
        residuals = y.values - preds
        motif_se = get_motif_se(X, y_err)
        # coef_df[param+'_se'] = motif_se
        # print(motif_se)

        coef_dfs.append(coef_df)
        
        m = get_model_metric(y, y_err, preds, n_feature)

        plt.subplot(1,3,stack_size)
        #errorbar(y, preds, xerr=y_err, fmt='.', alpha=0.1,zorder=0, color='k')
        hue_order = ['WC_5ntstem', 'WC_6ntstem', 'WC_7ntstem']
        sns.scatterplot(x=param, y='tmp_pred', data=df, hue='ConstructType', hue_order=hue_order, linewidth=0, s=10, alpha=0.6, palette='plasma')
        plt.xlabel('Fit '+param)
        plt.ylabel('CV-test-split predicted '+param)
        plt.title("%s, %d features\n RMSE: %.2f, $\chi^2$: %.2f \n BIC: %.2f" % (titles[stack_size], n_feature, m['rmse'], m['chi2'], m['bic']))
        
        if lim is not None:
            plt.plot(lim,lim,'--',color='grey',zorder=0)
            plt.xlim(lim)
            plt.ylim(lim)
        if i!=0: plt.legend([],frameon=False)

        plt.tight_layout()

    return coef_dfs, motif_se, feats, preds, results


def fit_NN_cv(df, feature_method='get_stack_feature_list_simple_loop', stack_size=2,
           param='dG_37', err='_se_corrected', lim=None,
           fit_intercept=False):


    N_SPLITS = 5
    y = df[param]
    y_err = df[param + err]
    y_weight = 1 / y_err**2

    feats = get_feature_count_matrix(df, feature_method=feature_method, stack_size=stack_size)
    n_feature = feats.shape[1]

    mdl = Ridge(fit_intercept=fit_intercept)
    X = feats.values
    results = cross_validate(mdl, X, y, cv=N_SPLITS, return_estimator=True, fit_params={'sample_weight': y_weight})
    preds = cross_val_predict(mdl, X, y, cv=N_SPLITS)
    m = get_model_metric(y, y_err, preds, n_feature)
    motif_se = get_motif_se(X, y_err)

    coef_df = pd.DataFrame(columns=['motif', param])
    for estimator in results['estimator']:
        for j in range(len(feats.columns)):
            coef_df = coef_df.append({'motif': feats.columns[j], param: estimator.coef_[j]}, ignore_index=True)

    motif_df = coef_df.groupby('motif').median()
    motif_df[param+'_cv_std'] = coef_df.groupby('motif').std()
    # assert motif_df.index.tolist() == feats.columns.tolist()
    motif_df = motif_df.join(pd.DataFrame(data=motif_se, index=feats.columns, columns=[param+'_se']))
    df['tmp_pred'] = preds

    hue_order = ['WC_5ntstem', 'WC_6ntstem', 'WC_7ntstem']
    fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(x=param, y='tmp_pred', data=df, hue='ConstructType', hue_order=hue_order,
                    linewidth=0, s=10, alpha=0.5, palette='plasma', ax=ax)
    plt.xlabel('Fit '+ param)
    plt.ylabel('CV-test-split predicted '+param)
    plt.title("%d features\n RMSE: %.2f, $\chi^2$: %.2f, \n BIC: %.2f" % (n_feature, m['rmse'], m['chi2'], m['bic']))

    if lim is not None:
        plt.plot(lim,lim,'--',color='grey',zorder=0)
        plt.xlim(lim)
        plt.ylim(lim)

    return motif_df, feats


def compare_fit_with_santalucia(df, santa_lucia, params=['dH', 'dS', 'dG_37']):

    for param in params:
        coef_dfs, _, _ = fit_NN_cv(df, param=param, err='_se')
        fit_param = pd.DataFrame(coef_dfs[1].groupby('motif').apply(np.nanmean), columns=[param]).join(
             pd.DataFrame(coef_dfs[1].groupby('motif').apply(lambda x: np.nanstd(x)/np.sqrt(5)), columns=[param+'_cv_se']))
        santa_lucia = santa_lucia.merge(fit_param, on='motif', how='inner', suffixes=('_SantaLucia', '_MANIfold'))

    return santa_lucia
    
def get_X_y(arr, split_dict, param, feats=None, split='train', sample_ratio=1):
    """
    Returns:
        feats - DataFrame, needs .values when fed into LinearRegressionSVD
    """
    assert split_dict is not None
    seqids = split_dict[split+'_ind']
    if feats is not None:
        feats_inlist_df = get_index_isinlist(feats, seqids)
        X = feats_inlist_df.values
    else:    
        feats = get_feature_count_matrix(arr, feature_method='get_feature_list', 
                                            fit_intercept=False, symmetry=False)
        X = feats.values
    df = arr.loc[feats_inlist_df.index,]
    y = df[param].values
    y_err = df[param+'_se'].values
    
    if sample_ratio < 1:
        sampled_inds = np.random.choice(len(y), int(len(y) * sample_ratio), replace=False)
        X, y, y_err = X[sampled_inds,:], y[sampled_inds], y_err[sampled_inds]
    
    return dict(X=X, y=y, y_err=y_err, feature_names=feats.columns.tolist(), param=param, split=split)
    
    
def fit_param(arr, data_split_dict, param, feats, ax=None, mode='val',
              fix_some_coef=False, method='svd', 
              train_only=False, use_train_set_ratio=1, 
              fix_coef_kwargs=dict(), regularization_kwargs=dict()):
    """
    Calls lr.fit() if not fix_some_coef, otherwise calls lr.fit_with_some_coef_fixed()
    Args:
        method - str, {'svd', 'regularized', 'passive_aggressive'}
        train_only - bool, skip validation or test
        fix_coef_kwargs - fixed_feature_names, coef_df are required
        regularization_kwargs - `coef_prior` or (`feature_names` & `coef_df`), `reg_lambda`.
    """
    color_dict = dict(dH='c', Tm='cornflowerblue', dG_37='teal', dS='steelblue')
    
    train_data = get_X_y(arr, data_split_dict, param=param, feats=feats, split='train', sample_ratio=use_train_set_ratio)
    val_data = get_X_y(arr, data_split_dict, param=param, feats=feats, split=mode)
    
    
    if method == 'svd':
        lr = LinearRegressionSVD(param=param)
        if not fix_some_coef:
            lr.fit(train_data['X'], train_data['y'], train_data['y_err'], feature_names=train_data['feature_names'],
                skip_rank=False)
        else:
            lr.fit_with_some_coef_fixed(train_data['X'], train_data['y'], train_data['y_err'], feature_names=train_data['feature_names'],
                **fix_coef_kwargs)
            
        if not train_only:
            plotting.plot_truth_predict(lr, val_data, ax=ax, title='NNN OLS model',
                                    color=color_dict[param], alpha=.05)
        
    elif method == 'regularized':
        lr = LinearRegressionRegularized()
        
        if fix_some_coef:
            fixed_feature_names = fix_coef_kwargs['fixed_feature_names']
        
        # Prepare `coef_df`
        try:
            coef_df = fix_coef_kwargs['coef_df']
        except:
            coef_df = regularization_kwargs['coef_df']
        # Add intermediate parameters not in original file and set to 0    
        extra_coef_list = [x for x in feats.columns if not x in coef_df.index]
        extra_coef_df = pd.DataFrame(index=extra_coef_list, data=0.0, columns=coef_df.columns)
        coef_df = pd.concat((coef_df, extra_coef_df), axis=0)
            
        if 'reg_lambda' in regularization_kwargs:
            lr.reg_lambda = regularization_kwargs['reg_lambda']
        
        print('Regularization strength is %f' % lr.reg_lambda)
        
        if not fix_some_coef:
            lr.fit(train_data['X'], train_data['y'], feature_names=train_data['feature_names'],
                   coef_df=coef_df)
        else:
            lr.fit_with_some_coef_fixed(train_data['X'], train_data['y'], 
                                        feature_names=train_data['feature_names'], fixed_feature_names=fixed_feature_names,
                                        coef_df=coef_df
            )
    
    elif method == 'passive_aggressive':
        lr = PassiveAggressiveRegressor()
        lr.fit(train_data['X'], train_data['y'])
         
    
    return lr
    
    
def pred_variant_lr(q_row, lr, feature_list_kwargs_dict, 
                 verbose=False, only_return_pred=False):
    """
    Predict a single variant given a linear regression model
    """
    feature_list_dict = feature_list_kwargs_dict.copy()
    feature_method = feature_list_dict.pop('feature_method')
    for key in ['lr_dict', 'DNA_conc']:
        if key in feature_list_dict:
            feature_list_dict.pop(key)
    
    q_feat_list = getattr(feature_list, feature_method)(q_row, **feature_list_dict)

    try:
        pred = lr.coef_df.loc[q_feat_list].sum().values[0]
    except:
        # if some feature is not in the parameter set in lr_dict
        pred = np.nan
    
    if only_return_pred:
        return pred
    else:
        gt = q_row[lr.param]
    
        if verbose:
            print("%.4f\t%.4f" % (pred, gt))
    
        return dict(pred=pred, gt=gt)
        
        
def pred_variant_Tm(q_row, lr_dict, feature_list_kwargs_dict, 
                    salt_adjust=False, sodium=1.0):
    """
    Predict (salt adjusted) Tm for hairpins
    """
    pred_dict = {p : pred_variant_lr(q_row, lr_dict[p], feature_list_kwargs_dict, only_return_pred=True)
                 for p in ['dH', 'dG']}
    Tm = get_Tm(pred_dict['dH'], pred_dict['dG'], celsius=37)
    
    if salt_adjust:
        if isinstance(q_row.RefSeq, list):
            seq = q_row.RefSeq[0]
        else:
            seq = q_row.RefSeq
        GC_content = get_GC_content(seq)
        return get_Na_adjusted_Tm(Tm, np.nan, GC_content, Na=sodium)
    else:
        return Tm 
        
def pred_variant(seq, struct, sodium=1.0, DNA_conc=None,
                 lr_dict=None, feature_list_kwargs_dict=None,
                 ):
        """
        Returns:
            result_dict
        """
        q_row = dict(RefSeq=seq, TargetStruct=struct)
        pred_dict = {p : pred_variant_lr(q_row, lr_dict[p], feature_list_kwargs_dict, only_return_pred=True)
                    for p in ['dH', 'dG']}
        if '+' in struct:
            if seq[0] == seq[1]:
                lnK = np.log(1/DNA_conc)
            else:
                lnK = np.log(2/DNA_conc)
                
            pred_dict['Tm'] = pred_dict['dH'] / ((pred_dict['dH'] - pred_dict['dG']) / (273.15 + 37) - R * lnK) - 273.15
            return get_Na_adjusted_param(Na=sodium, from_Na=1.0, seq=seq[0], **pred_dict)
        else:
            pred_dict['Tm'] = get_Tm(pred_dict['dH'], pred_dict['dG'], celsius=37)
            return get_Na_adjusted_param(Na=sodium, from_Na=1.0, seq=seq, **pred_dict)
            
            
"""
def pred(val_df, lr_dict):
    result = []
    for i,row in val_df.iterrows():
        result.append(pred_variant_Tm(row, lr_dict, feature_list_kwargs, salt_adjust=True, sodium=0.065))
"""    