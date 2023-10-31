from typing import Tuple
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from lmfit.models import PowerLawModel
from sklearn import linear_model
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
import lmfit
from tqdm import tqdm
tqdm.pandas()

from .util import *
from . import motif_fit as mf

sns.set_style('ticks')
sns.set_context('paper')

# make sure the text is editable in illustrator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# set font to arial
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

# some constants
kB = 0.0019872 # Bolzman constant
C2T = 273.15 # conversion from celsius to kalvin

###############################
##### Combine Experiments #####
###############################

def get_combined_ddX(p1, p2, e1, e2):
    e = np.sqrt(e1**2 + e2**2)
    ddx = p1 - p2
    z =( ddx - np.nanmean(ddx)) / e

    return z, e, np.nanmean(ddx)


def get_combined_param(params, errors):
    """
    Simply combine, weighted by 1/var
    Args:
        params, errors - (n_variant, n_rep) array like
    Returns:
        parameter and error - (n_variant, 2) array, of the combined dataset
    """
    params, errors = np.atleast_2d(np.array(params)), np.atleast_2d(np.array(errors))

    assert params.shape == errors.shape, f"Shapes don't match between params {params.shape} and errors {errors.shape}"

    e = (np.nansum(1 / errors**2, axis=1))**(-1)
    p = np.nansum(params / errors**2, axis=1) * e
    return p, np.sqrt(e)


def get_combined_param_bt(params, errors, n_sampling = 100):
    """
    Bootstrapped combined error
    Args:
        params, errors - (n_variant, n_rep) array like
    Returns:
        parameter and error - (n_variant, 2) array, of the combined dataset
    """
    p, e = None, None
    n_variant, n_rep = params.shape
    rnd_sample = np.array([np.random.normal(params, errors) for _ in range(n_sampling)])
    rnd_sample = np.moveaxis(rnd_sample, 0, 2) # (n_variant, n_rep, n_sampling)
    rnd_sample = np.reshape(rnd_sample, (n_variant, n_rep * n_sampling), order='C') # (n_variant, n_rep * n_sampling)
    p = np.median(rnd_sample, axis=1)
    ub = np.percentile(rnd_sample, 95, axis=1)
    lb = np.percentile(rnd_sample, 5, axis=1)
    e = (ub - lb) / 2
    return dict(p=p, e=e, ub=ub, lb=lb)


def combine_replicates(reps, rep_names, 
                       error_type='lb_ub', param=None, verbose=True) -> pd.DataFrame:
    """
    Args:
        reps - iterable, each is a df from one replicate
        rep_names - iterable of str
        error_type - str, {'lb_ub', 'se'}, indicates how error is represented in `reps`
    """
    def get_cols_2_join(rep, params):
        cols = [c for c in rep.columns if any(c.startswith(p) for p in params)]
        if 'n_clusters' in rep.columns:
            cols += ['n_clusters']
        return cols

    def get_rep_param_from_df(df, param):
        """
        Returns:
            (n_variant, n_rep) df
        """
        return df[[c for c in df.columns if (('-' in c) and c.split('-')[0] == param)]]


    if param is None:
        params = ['dH', 'Tm', 'dG_37', 'dS', 'fmax', 'fmin']
    cols = [get_cols_2_join(rep, params) for rep in reps]
    
    for i, (rep, col, rep_name) in enumerate(zip(reps, cols, rep_names)):
        if i == 0:
            df = rep[col].rename(columns={c: c+'-'+rep_name for c in col})
        else:
            df = df.join(rep[col].rename(columns={c: c+'-'+rep_name for c in col}), how='outer')

    for param in params:
        if verbose:
            print(f'\nCombining {param}')
        df[param], df[param+'_se'] = get_combined_param(get_rep_param_from_df(df, param), get_rep_param_from_df(df, param+'_se'))
        df[param+'_lb'], _ = get_combined_param(get_rep_param_from_df(df, param+'_lb'), get_rep_param_from_df(df, param+'_se'))
        df[param+'_ub'], _ = get_combined_param(get_rep_param_from_df(df, param+'_ub'), get_rep_param_from_df(df, param+'_se'))

    return df

def combine_replicate_p_unfold(p_unfold_dict, celsius_dict):
    """
    !!! Only works when only one rep has missing temperature condition
    !!! Not weighted by error
    """
    # find the rep with missing celsius data
    n_celsius = [len(value) for value in celsius_dict.values()]
    idmin, idmax = np.argmin(n_celsius), np.argmax(n_celsius)
    repmin, repmax = list(celsius_dict.keys())[idmin], list(celsius_dict.keys())[idmax]
    celsius_list = celsius_dict[repmax]
    missing_celsius = set(celsius_dict[repmax]) - set(celsius_dict[repmin])
    p_unfold_dict[repmin][['%s_%.1f'%(repmin, x) for x in missing_celsius]] = np.nan

    for i, rep_name in enumerate(p_unfold_dict):
        if i == 0:
            df = p_unfold_dict[rep_name][['%s_%.1f'%(rep_name, x) for x in celsius_list]]
        else:
            df = df.join(p_unfold_dict[rep_name][['%s_%.1f'%(rep_name, x) for x in celsius_list]], how='outer')
    
    p_unfold = np.nanmean(
        df.values.reshape(-1, len(celsius_dict), len(celsius_dict[repmax])),
        axis=1)
    return pd.DataFrame(data=p_unfold, columns=celsius_list, index=df.index)
    

def correct_interexperiment_error(r1, r2, plot=True, figdir=None, return_debug=False):
    """
    Returns:
        A, k - correction parameters
    """
    def plot_zscores(df, figdir):
        l = 7.5
        cm = 1/2.54
        bins = np.arange(-l, l, 0.05)
        fig, ax = plt.subplots(1,3,figsize=(3*4.25*cm,3.5*cm))
        sns.histplot(df['ddH_zscore'], bins=50, stat='density', color=palette[0], ax=ax[0])
        sns.histplot(df['dTm_zscore'], bins=50, stat='density', color=palette[1], ax=ax[1])
        sns.histplot(df['ddG_37_zscore'], bins=50, stat='density', color=palette[2], ax=ax[2])
        
        for i in range(3):
            ax[i].plot(bins, norm.pdf(bins, 0, 1), 'k--')
        
        ax[0].set_xlim([-l, l])
        ax[0].set_title('dH offset: %.2f kcal/mol\nzscore std: %.2f'% (offset['dH'], np.std(df['ddH_zscore'])),
                        size=6)
        ax[1].set_xlim([-l, l])
        ax[1].set_title('Tm offset: %.2f K\nzscore std: %.2f' % (offset['Tm'], np.std(df['dTm_zscore'])),
                        size=6)
        ax[2].set_xlim([-l, l])
        ax[2].set_title('dG 37°C offset: %.2f kcal/mol\nzscore std: %.2f'%(offset['dG_37'], np.std(df['ddG_37_zscore'])),
                        size=6)
        sns.despine()
        if figdir is not None:
            save_fig(os.path.join(figdir, 'zscores.pdf'), fig=fig)
        else:
            plt.show()

    def plot_powerlaw(powerlaw_result):
        powerlaw_result.plot(xlabel='intra-experimental error',
            ylabel='std of ddG z-score')
        if figdir is not None:
            save_fig(os.path.join(figdir, 'fit_powerlaw.pdf'),)
        else:
            plt.show()

    def plot_corrected_dG_se(df):
        fig, ax = plt.subplots()
        sns.histplot(df.dG_37_se, color='gray', stat='density')
        sns.histplot(df.dG_37_se_corrected, color='brown', stat='density')
        plt.legend(['before correction', 'after correction'])
        plt.xlim([0,0.5])
        if figdir is not None:
            save_fig(os.path.join(figdir, 'corrected_dG_se.pdf'),)
        else:
            plt.show()

    def plot_corrected_zscore(df):
        fig, ax = plt.subplots()

        l = 20
        offset = df['dG_37-x'] - df['dG_37-y']
        bins = np.arange(-l, l, 0.5)
        plt.plot(bins, norm.pdf(bins, 0, 1), 'k--')
        zscore = df['ddG_37_zscore']
        sns.histplot(zscore[np.abs(zscore)<l], bins=bins, stat='density', color='gray')
        zscore = df['ddG_37_zscore_corrected']
        sns.histplot(zscore[np.abs(zscore)<l], bins=bins, stat='density', color='brown')
        plt.xlim([-l, l])
        plt.legend(['expected', 'before correction', 'after correction'])
        plt.xlabel('ddG z-score')
        if figdir is not None:
            save_fig(os.path.join(figdir, 'corrected_ddG_zscore.pdf'),)
        else:
            plt.show()

    if not figdir is None:
        if not os.path.isdir(figdir):
            os.makedirs(figdir)

    df = combine_replicates((r1, r2), ('x', 'y'))

    params = ['dG_37', 'Tm', 'dH']
    offset = {p:0.0 for p in params}
    for param in params:
        df[f'd{param}_zscore'], df[f'd{param}_se'], offset[f'{param}'] = get_combined_ddX(df[f'{param}-x'], df[f'{param}-y'], df[f'{param}_se-x'], df[f'{param}_se-y'])
    plot_zscores(df, figdir)

    df['dG_bin'] = pd.qcut(df.ddG_37_se, 100)
    sigma_df = df[['ddG_37_zscore', 'dG_bin']].groupby('dG_bin').apply(np.std).rename(columns={'ddG_37_zscore':'ddG_37_zscore_std'})
    sigma_df['intra_err'] = [x.mid for x in sigma_df.index.values]

    model = PowerLawModel()
    powerlaw_result = model.fit(sigma_df.ddG_37_zscore_std[1:-1], x=sigma_df.intra_err[1:-1])
    plot_powerlaw(powerlaw_result)

    df['dG_37_se_corrected'] = df.dG_37_se * powerlaw_result.eval(x=df.ddG_37_se)
    df['ddG_37_se_corrected'] = df.ddG_37_se * powerlaw_result.eval(x=df.ddG_37_se)
    offset = df['dG_37-x'] - df['dG_37-y']
    df['ddG_37_zscore_corrected'] = (offset - np.mean(offset)) / df.ddG_37_se_corrected
    plot_corrected_dG_se(df)
    plot_corrected_zscore(df)

    if return_debug:
        return powerlaw_result, df
    else:
        return powerlaw_result.best_values['amplitude'], powerlaw_result.best_values['exponent']


def get_variances(y, sigma, y_hat, 
                  regress_sigma=False, sigma_model=None, return_model=False,
                  verbose=True):
    """
    Args:
        sigma_model - LinearRegression, if None fit afresh
    """
    n = len(y)

    var = defaultdict()
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