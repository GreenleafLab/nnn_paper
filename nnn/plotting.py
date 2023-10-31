import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
from scipy.stats import chi2, pearsonr, norm
# from sklearn.metrics import r2_score
from ipynb.draw import draw_struct
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.cm as cm

from .util import *
from . import util

sns.set_style('ticks')
sns.set_context('paper')

palette = ['#2f4f4f','#228b22','#00ff00','#000080','#1e90ff','#00ffff','#ff8c00','#deb887','#8b4513','#ff0000','#ff69b4','#800080',]
# palette = cc.glasbey_dark
# palette=[
#     '#201615',
#     '#4e4c4f',
#     '#4c4e41',
#     '#936d60',
#     '#5f5556',
#     '#537692',
#     '#a3acb1']#cc.glasbey_dark
from scipy.stats import gaussian_kde

def calc_kde_pdf(data):
    """
    Args:
        data - (n_points, n_dim)
    Returns:
        (n_points)
    """
    kde = gaussian_kde(data.dropna().T)
    return kde.evaluate(data.T)


def plot_colored_scatter_comparison(data, x, y, 
                                    palette='plasma', alpha=1, ax=None, lim=None, 
                                    rasterized=True, show_cbar=True, color_by_density=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))

    if lim is None:
        # auto lim calculation
        margin = .5
        ll = min(np.percentile(data[x].dropna(), 1), np.percentile(data[y].dropna(), 1))
        ul = max(np.percentile(data[x].dropna(), 99), np.percentile(data[y].dropna(), 99))
        r = ul - ll
        lim = [ll - r * margin, ul + r * margin]

    ax.plot(lim, lim, '--', zorder=0, color=[.9,.9,.9])
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    df = data.copy()
    
    if color_by_density:
        df['density'] = calc_kde_pdf(data[[x,y]])
        # df['size'] = 200#100 / data[x]**2
        hue_norm = (0 * np.max(df.density), 1 * np.max(df.density))
        
        norm = plt.Normalize(hue_norm[0], hue_norm[1])
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        sns.scatterplot(data=df, x=x, y=y, hue='density', size=.1, hue_norm=hue_norm,
                        palette=palette, alpha=alpha, legend=False, ax=ax, rasterized=rasterized, **kwargs)
    else:
        show_cbar = False
        sns.scatterplot(data=df, x=x, y=y, size=.1,
                        palette=palette, alpha=alpha, ax=ax, legend=False, rasterized=rasterized, **kwargs)
        
        
    sns.despine()
    if show_cbar:
        cbar = plt.colorbar(sm)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label('density', size=5)


def generate_color_palette(index):
    # return sns.dark_palette(sns.color_palette('Set2', 4)[index], reverse=False, as_cmap=True)
    return sns.dark_palette(cc.glasbey_light[index], reverse=False, as_cmap=True)
    
def plot_kde_comparison(df, col, lim, color='#8b4513'):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.kdeplot(data=df, x=col, y=col+'_final', color=color)
    plt.plot(lim, lim, 'k--')
    plt.xlim(lim)
    plt.ylim(lim)
    
def plot_dis_comparison(df, col, lim, color='#deb887'):
    # fig, ax = plt.subplots(figsize=(6,6))
    sns.displot(data=df, x=col, y=col+'_final', color=color)
    plt.plot(lim, lim, 'k--')
    plt.xlim(lim)
    plt.ylim(lim)

def plot_se_dist(df):
    fig, ax = plt.subplots(1,3,figsize=(12,3), sharey=True)
    if 'dG_37_se' in df.columns:
        sns.histplot(df.dG_37_se, kde=False, bins=30, color=palette[0], ax=ax[0])
    sns.histplot(df.Tm_se, kde=False, bins=30, color=palette[1], ax=ax[1])
    sns.histplot(df.dH_se, kde=False, bins=30, color=palette[2], ax=ax[2])


def plot_rep_comparison(r1, r2, param, lim, kind='kde', add_final=False, color='#deb887'):
    """
    Args:
        kind - {'kde', 'scatter', 'kde_scatter'}
    """
    if add_final:
        col = param + '_final'
    else:
        col = param
        
    df = r1[[col]].merge(r2[[col]], left_index=True, right_index=True)
    rsqr = r2_score(df[col+'_x'], df[col+'_y'])
    pearson, _ = pearsonr(df[param+'_x'], df[param+'_y'])

    fig, ax = plt.subplots(figsize=(6,6))
    l = np.abs(lim[1] - lim[0])
    plt.plot(lim, lim, '--', c='gray')
    if kind == 'kde':
        sns.kdeplot(data=df, x=col+'_x', y=col+'_y', color=color)
    elif kind == 'scatter':
        sns.scatterplot(data=df, x=col+'_x', y=col+'_y', color=color)
    elif kind == 'kde_scatter':
        sns.scatterplot(data=df, x=col+'_x', y=col+'_y', color=color)
        sns.kdeplot(data=df, x=col+'_x', y=col+'_y', color=color, fill=True)


    plt.xlim(lim)
    plt.ylim(lim)    
    plt.xlabel('r1')
    plt.ylabel('r2')
    plt.text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
    plt.text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
    plt.title(param)
    
def plot_rep_comparison_by_series(r1, r2, annotation, param, lim,
    suffixes=('_x', '_y'), xlabel='r1', ylabel='r2'):
    df = r1[[param]].merge(r2[[param]], left_index=True, right_index=True)
    df = df.merge(annotation, left_index=True, right_index=True)

    series = df.groupby('Series').apply(len).sort_values(ascending=False)
    l = np.abs(lim[1] - lim[0])

    fig, ax = plt.subplots(3,4,figsize=(20,15), sharex=True, sharey=True)
    ax = ax.flatten()

    for i, s in enumerate(series.index[:12]):
        series_df = df.query('Series == "%s"'%s)
        print('Series %s,  %d variants' % (s, len(series_df)))
        ax[i].plot(lim, lim, '--', c='gray')
        if len(series_df) > 100:
            sns.kdeplot(data=series_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])
            # rsqr = r2_score(series_df[param+suffixes[0]], series_df[param+suffixes[1]])
            pearson, _ = pearsonr(series_df[param+suffixes[0]], series_df[param+suffixes[1]])
            # ax[i].text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            sns.scatterplot(data=series_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)    
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title('%s, N=%d'%(s, series[s]))

    plt.suptitle(param)

    return fig, ax


def plot_rep_comparison_by_ConstructType(r1, r2, annotation, param, series, lim,
    suffixes=('_x', '_y'), xlabel='r1', ylabel='r2'):

    df = r1[[param]].merge(r2[[param]], left_index=True, right_index=True)
    df = df.merge(annotation, left_index=True, right_index=True).query('Series == "%s"'%series)

    types = df.groupby('ConstructType').apply(len).sort_values(ascending=False)
    l = np.abs(lim[1] - lim[0])

    fig, ax = plt.subplots(1, 3, figsize=(18,6), sharex=True, sharey=True)
    ax = ax.flatten()

    for i, s in enumerate(types.index):
        types_df = df.query('ConstructType == "%s"'%s)
        print('ConstructType %s,  %d variants' % (s, len(types_df)))
        ax[i].plot(lim, lim, '--', c='gray')
        if len(types_df) > 100:
            sns.kdeplot(data=types_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])
            rsqr = r2_score(types_df[param+suffixes[0]], types_df[param+suffixes[1]])
            pearson, _ = pearsonr(types_df[param+suffixes[0]], types_df[param+suffixes[1]])
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            sns.scatterplot(data=types_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)    
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title('%s, N=%d'%(s, types[s]))

    plt.suptitle(param)

    return fig, ax


def plot_comparison_by_series(vf, param, suffix = '_NUPACK_salt_corrected',
    annotation=None, lim=None, xlabel=None, ylabel=None):

    df = vf.copy()
    if annotation is not None:
        df = df.join(annotation)
        
    df.loc[df.Series == 'External', 'Series'] = 'Control'
    df.loc[df.Series == 'TRIloop', 'Series'] = 'Hairpin Loops'
    df.loc[df.Series == 'TETRAloop', 'Series'] = 'Hairpin Loops'

    series = df.query('Series != "Control"').groupby('Series').apply(len).sort_values(ascending=False)
    print(series)
    l = np.abs(lim[1] - lim[0])

    fig, ax = plt.subplots(2,2,figsize=(10,10), sharex=False, sharey=False)
    ax = ax.flatten()

    for i, s in enumerate(series.index[:4]):
        series_df = df.query('Series == "%s"'%s)[[param+suffix, param]].dropna(subset=[param, param+suffix])
        print('Series %s,  %d variants' % (s, len(series_df)))
        ax[i].plot(lim, lim, '--', c='gray', zorder=0)
        if len(series_df) > 100:
            # plot_colored_scatter_comparison(data=series_df, x=param+suffix, y=param,
            #     lim=lim, palette=generate_color_palette(i), ax=ax[i])
            sns.scatterplot(data=series_df, x=param+suffix, y=param,
                color=palette[i % len(palette)], alpha=.1, ax=ax[i])
            # sns.kdeplot(data=series_df, x=param+suffix, y=param,
            #     color=palette[i % len(palette)], fill=False, ax=ax[i])
            # rsqr = r2_score(series_df[param+suffix], series_df[param])
            pearson, _ = pearsonr(series_df[param+suffix], series_df[param])
            # ax[i].text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            # plot_colored_scatter_comparison(data=series_df, x=param+suffix, y=param,
            #     lim=lim, palette=generate_color_palette(i))
            sns.scatterplot(data=series_df, x=param+suffix, y=param,
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)
        if xlabel is None:
            xlabel = 'NUPACK $dG_{37}$ (kcal/mol)' 
        if ylabel is None:
            ylabel = 'MANifold $dG_{37}$ (kcal/mol)'
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title('%s, N=%d'%(s, series[s]))

    plt.suptitle(param)

    return fig, ax


def plot_comparison_by_type(vf, param, suffix = '_NUPACK_salt_corrected',
    series='WatsonCrick', annotation=None, lim=None):
    df = vf.query(f'Series == "{series}"').copy()
    if annotation is not None:
        df = df.join(annotation)
        
    construct_types = df.groupby('ConstructType').apply(len).sort_values(ascending=False)
    fig, ax = plt.subplots(2,3,figsize=(15,10), sharex=False, sharey=False)
    ax = ax.flatten()
    l = np.abs(lim[1] - lim[0])

    for i, s in enumerate(construct_types.index[:12]):
        type_df = df.query('ConstructType == "%s"'%s)
        print('Construct Type %s,  %d variants' % (s, len(type_df)))
        ax[i].plot(lim, lim, '--', c='gray')
        if len(type_df) > 40:
            sns.scatterplot(data=type_df, x=param+suffix, y=param,
                color=palette[i % len(palette)], alpha=.1, ax=ax[i])
            sns.kdeplot(data=type_df, x=param+suffix, y=param,
                color=palette[i % len(palette)], fill=False, ax=ax[i])
            # rsqr = r2_score(type_df[param+suffix], type_df[param])
            pearson, _ = pearsonr(type_df[param+suffix], type_df[param])
            # ax[i].text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            sns.scatterplot(data=type_df, x=param+suffix, y=param,
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)    
        ax[i].set_xlabel('NUPACK')
        ax[i].set_ylabel('MANifold')
        ax[i].set_title('%s, N=%d'%(s, construct_types[s]))

    plt.suptitle(param)

    return fig, ax


def plot_actual_and_expected_fit(row, ax, c='k', conds=None):
    """
    Takes replicate data
    """
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*((Tm+273.15)**-1 - x)))
    if conds is None:
        conds = [x for x in row.keys() if x.endswith('_norm')]
        
        errs = [x for x in row.keys() if x.endswith('_norm_std')]
    else:
        errs = [x+'_std' for x in conds]

    vals = np.array(row[conds].values,dtype=float) 
    errors = np.array(row[errs].values / np.sqrt(row['n_clusters']),dtype=float)

    T_celsius=[float(x.split('_')[1]) for x in conds]
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH'],row['Tm'], row['fmax'], row['fmin'], T_inv)
    pred_ub = function(row['dH_lb'], row['Tm_lb'], row['fmax_ub'], row['fmin_ub'], T_inv)
    pred_lb = function(row['dH_ub'], row['Tm_ub'], row['fmax_lb'], row['fmin_lb'], T_inv)
    
    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.axhline(row['fmax'], c=c, linestyle='--')
    ax.axhline(row['fmin'], c=c, linestyle='--')
    ax.errorbar(T_celsius, vals, yerr=errors,fmt='.',c=c)
    ax.plot(T_celsius, pred_fit, c=c, lw=0.5)
    ax.fill_between(T_celsius, pred_ub, pred_lb, color=c, alpha=.3)
    ax.set_title('%s, RMSE: %.3f  [%d%d]'% (row.name, row['RMSE'], row['enforce_fmax'], row['enforce_fmin']))


def plot_renorm_actual_and_expected_fit(row, ax, c='k', conds=None):
    """
    Re-normalized to between 0 and 1
    NOT TESTED
    """
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*((Tm+273.15)**-1 - x)))
    renorm = lambda x, fmax, fmin: x / (fmax - fmin)  - fmin
    if conds is None:
        conds = [x for x in row.keys() if x.endswith('_norm')]
        
        errs = [x for x in row.keys() if x.endswith('_norm_std')]
    else:
        errs = [x+'_std' for x in conds]

    fmax, fmin = row.fmax, row.fmin
    vals = renorm( np.array(row[conds].values,dtype=float), fmax, fmin )
    errors = np.array(row[errs].values / np.sqrt(row['n_clusters']),dtype=float) / (fmax - fmin)

    T_celsius=[float(x.split('_')[1]) for x in conds]
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH'],row['Tm'], 1, 0, T_inv)
    
    ax.set_xlim([18,62])
    ax.set_ylim([-0.1,1.1])

    ax.errorbar(T_celsius, vals, yerr=errors,fmt='.',c=c)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)
    ax.set_title('%s, RMSE: %.3f  [%d%d]'% (row.name, row['RMSE'], row['enforce_fmax'], row['enforce_fmin']))


def plot_NUPACK_curve(row, ax, T_celsius=np.arange(20,62.5,2.5), c='k'):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*((Tm+273.15)**-1 - x)))

    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH_NUPACK'],row['Tm_NUPACK'], 1, 0, T_inv)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)

    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.set_title('%s, NUPACK dH = %.2f, Tm = %.2f'% (row.name, row['dH_NUPACK'],row['Tm_NUPACK']))


def plot_corrected_NUPACK_curve(row, ax, T_celsius=None, c='k', conds=None, sodium=1.0):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*((Tm+273.15)**-1 - x)))

    if T_celsius is None:
        T_celsius = np.arange(20,62.5,2.5)
        ax.set_xlim([13,62])
    else:
        ax.set_xlim([np.min(T_celsius)-5, np.max(T_celsius)+5])
        
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    GC_content = get_GC_content(row.RefSeq)
    Tm = get_Na_adjusted_Tm(row['Tm_NUPACK'], row['dH_NUPACK'], GC_content, Na=sodium)
    pred_fit = function(row['dH_NUPACK'],Tm, 1, 0, T_inv)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)

    
    ax.set_ylim([-0.1,1.4])

    ax.set_title('%s, NUPACK dH = %.2f, Tm = %.2f'% (row.name, row['dH_NUPACK'],Tm))


def plot_candidate_variant_summary(candidate, df_with_targetstruct, df_with_curve, df_with_nupack):
    fig,ax = plt.subplots(1,3,figsize=(12,4))
    draw_struct(df_with_targetstruct.loc[candidate, 'RefSeq'], df_with_targetstruct.loc[candidate, 'TargetStruct'],ax=ax[0])
    plot_actual_and_expected_fit(df_with_curve.loc[candidate,:], ax=ax[1])
    plot_corrected_NUPACK_curve(df_with_nupack.loc[candidate,:], ax=ax[2])

    print('====Library Info===\n', df_with_nupack.loc[candidate,:])
    cols = ['dH', 'Tm', 'dS', 'dG_37', 'dG_37_se_corrected', 'RMSE']
    print('\n====Fit Info===\n', df_with_targetstruct.loc[candidate,cols])
    print('\n%d clusters'%df_with_curve.loc[candidate,'n_clusters'])


def draw_target_struct(seqid, arr, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    draw_struct(arr.loc[seqid, 'RefSeq'], arr.loc[seqid, 'TargetStruct'], ax=ax)
    ax.set_title(seqid)
    

def draw_target_mfe_struct(row=None, seq=None, target_struct=None, celsius=0.0):
    """
    Either supply `row` with `RefSeq` and `TargetStruct` fileds or `seq` and `target_struct` directly
    """
    if row is not None:
        seq = row.RefSeq,
        target_struct = row.TargetStruct
        
    _, ax = plt.subplots(1,3,figsize=(9,3))
    draw_struct(seq, target_struct, ax=ax[0])
    ax[0].set_title('Target')
    draw_struct(seq, util.get_mfe_struct(seq, sodium=0.081, celsius=celsius), ax=ax[1])
    ax[1].set_title('MFE, 81mM $Na^+$')
    draw_struct(seq, util.get_mfe_struct(seq, sodium=0.081,celsius=celsius), ax=ax[2])
    ax[2].set_title('MFE, 1M $Na^+$')


def plot_motif_param_errorbar(motif_df, param):
    fig, ax = plt.subplots(figsize=(10,3))
    plt.errorbar(range(len(motif_df)), motif_df[param], motif_df[param+'_se'], fmt='.')
    plt.xticks(range(len(motif_df)), motif_df.index, rotation=20)
    plt.title('WC $%s$ NN parameters' % param)


def plot_fitting_evaluation(fitted_variant_df_list, legend, save_pdf_file=None):
    for col in ['RMSE', 'rsqr', 'chisq', 'red_chisq', 'dH_se', 'Tm_se', 'dG_37_se', 'dS_se']:
        plt.figure()
        for i,df in enumerate(fitted_variant_df_list):
            sns.kdeplot(df[col], color=palette[i])

        plt.legend(legend)
    
    if save_pdf_file is not None:
        save_multi_image(save_pdf_file)
        
def get_pairwise_pearsonr_matrix(df_list, param='dG_37'):
    """
    df_list - List[DataFrame] with only the desired col
    """
    n = len(df_list)
    corr_mat = np.eye(n, dtype=float)
    for i in range(n):
        df_list[i].columns = [param]
    
    for i,df1 in enumerate(df_list):
        for j in range(i+1, n):
            df2 = df_list[j]
            df = df1.join(df2, lsuffix='_x', rsuffix='_y')[[param+'_x', param+'_y']].dropna()
            pearson, _ = pearsonr(df[param+'_x'], df[param+'_y'])
            corr_mat[i, j] = pearson
            corr_mat[j, i] = pearson
    
    return corr_mat
    
    
##### Plotting functions for LinearRegressionSVD #####
    
def plot_fitted_coef(linear_regression:LinearRegressionSVD, ax=None, **kwargs):

    coef, coef_se = linear_regression.coef_, linear_regression.coef_se_
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
        
    ax.errorbar(np.arange(len(coef)), coef, coef_se, fmt='k.', capsize=3, **kwargs)
    ax.set_xticks(np.arange(len(linear_regression.feature_names_in_)))
    ax.set_xticklabels(linear_regression.feature_names_in_, rotation=30)
    sns.despine()
    # plt.show()

    return ax
    
def plot_truth_predict(lr:LinearRegressionSVD, 
                          data_dict,
                          lim=None, ax=None, title=None,
                          nupack_prediction=None, **kwargs):
    """
    Args:
        data_dict - has keys X, y, y_err, param, feature_names
        nupack_prediction - array-like. If set, just give `lr`=None,
            only containing the col to be plotted e.g. dG_37_NUPACK_salt_corrected
            and the rows in test set
        title - str, add to the first line
        kwargs - passed to ax.errorbar
    """
    X_test, y_test, yerr_test, param = data_dict['X'], data_dict['y'], data_dict['y_err'], data_dict['param']
    if nupack_prediction is None:
        pred = lr.predict(X_test)
        pred_err = lr.predict_err(X_test)
    else:
        pred = nupack_prediction
        pred_err = np.zeros_like(pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
        
    if len(kwargs) == 0:
        kwargs = dict(color='k', alpha=.3)
        
        
    ax.errorbar(y_test, pred, 
                xerr=yerr_test, yerr=pred_err, fmt='.', **kwargs)

    if lim is None:
        if param == 'dG_37':
            #lim = [-3, 1.5]
            lim = [-4.5, 0]
        elif param == 'dH':
            lim = [-60, 0]
            # lim = [-60, -10]
        elif param == 'dS':
            lim = [-.2,0]
        elif param == 'Tm':
            lim = [19, 61]
        # plt.axis('equal')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim, lim, '--', c='gray', zorder=0)
    
    if param == 'Tm':    
        ax.set_xlabel('measurement (°C)')
        ax.set_ylabel('prediction (°C)')
    else:
        ax.set_xlabel('measurement (kcal/mol)')
        ax.set_ylabel('prediction (kcal/mol)')
    corr, _ = pearsonr(y_test, pred)
    
    
    if nupack_prediction is None:
        title_full = ('%d features\n$R^2$ = %.3f, corr = %.3f\nRMSE = %.3f, MAE = %.3f' % 
            (len(lr.coef_), r2_score(y_test, pred), corr, util.rmse(y_test, pred), util.mae(y_test, pred)))
    else:
        title_full = ('NUPACK\n$R^2$ = %.3f, corr = %.3f\nRMSE = %.3f, MAE = %.3f' % 
                      (r2_score(y_test, pred), corr, util.rmse(y_test, pred), util.mae(y_test, pred)))
    
    if title is not None:
        if param == 'dG_37':
            param_name = 'dG°_{37}'
        else:
            param_name = param
        title += ': %s %s' % (param_name, data_dict['split'])
        title_full = r"$\bf{" + title.replace(' ', '\ ') + '}$\n' + title_full
    
    ax.set_title(title_full)
    
    sns.despine()
    return ax


def plot_connected_dot(subset_data, width:float=0.2, ax=None):
    """
    Connected dot plot to show subsets
    Args:
        subset_data - List[List[List[int]]]
        width - how far those in the same group are dispersed
    """
    plt.rcParams["axes.prop_cycle"] = util.get_cycle("Dark2")
    
    if ax is None:
        _,ax = plt.subplots(figsize=(5,3))
        
    # Add the connected dot plot
    for idx, subsets in enumerate(subset_data):
        offset = np.linspace(-width,width,len(subsets))
        for i, subset_points in enumerate(subsets):
            y_positions = np.array([idx + offset[i]] * len(subset_points))
            ax.plot(subset_points, y_positions, 
                    marker='o', linestyle='-', linewidth=8, markersize=8, alpha=0.3
                    )
            
            
def plot_arc_diagram(nodes, edges, edge_weights=None, edge_width_factor=1, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Position the nodes along the x axis
    node_positions = {node: i for i, node in enumerate(nodes)}
    
    # Plot the nodes
    nodes_y = .5
    for node in nodes:
        ax.plot(node_positions[node], nodes_y, 'o', color='blue', markersize=10)
        ax.text(node_positions[node], nodes_y - 0.2, node, ha='center', va='top')
    
    if edge_weights is None:
        edge_weights = np.ones(len(edges))
    
    edge_weights = edge_width_factor * np.array(edge_weights)
    
    # Draw arcs for each edge
    for i,edge in enumerate(edges):
        start, end = edge
        start_pos, end_pos = node_positions[start], node_positions[end]
        
        # Use a Bezier curve for the arc for more control
        verts = [
            (start_pos, nodes_y),  # P0
            ((start_pos + end_pos) / 2, abs(start_pos - end_pos) / 3),  # P1
            (end_pos, nodes_y),  # P2
        ]

        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', 
                                  lw=edge_weights[i], edgecolor='k', alpha=.1)
        ax.add_patch(patch)
    
    # Set axis properties
    ax.set_ylim(bottom=-.5) # avoid cropping out the node circles at the edge
    ax.set_aspect('equal')
    ax.axis('off')

