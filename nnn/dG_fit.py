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
from . import plotting, arraydata

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

def dG_T(xdata, f_norm):
    return -kB * xdata.reshape(1,-1) * np.log(1/f_norm - 1)

def transform_2_dG(xdata, rep, curve, curve_se, epsilon):
    """
    Transforms fluorescence signal into dG space and prepagate uncertainties
    Args:
        xdata - in Kalvins not Celsius
    """
    fmax = rep.fmax.values.reshape(-1, 1)
    fmin = rep.fmin.values.reshape(-1, 1)
    curve_norm = np.clip((curve - fmin) / (fmax - fmin),
                        a_min=epsilon, a_max=1-epsilon)
    dG = dG_T(xdata, f_norm=curve_norm)
    curve_norm_se = curve_se.values / (rep.fmax - rep.fmin).values.reshape(-1, 1)
    curve_norm_uarray = unumpy.uarray(curve_norm, curve_norm_se)
    dG_uarray = -kB * xdata.reshape(1,-1) * unumpy.log(1/curve_norm_uarray - 1)
    dG_se_arr = unumpy.std_devs(dG_uarray)
    dG_se = pd.DataFrame(data=dG_se_arr, index=dG.index, columns=dG.columns)
    
    return dG, dG_se
    
def get_rep_dG(arraydata, rep_name, epsilon=1e-2, 
               plot_dG_se=False, dG_se_perc_thresh=90):
    """
    Loads data and calls transform_2_dG
    Optionally plot ECDF of dG error to set threshold for outlier data points
    in the next step.
    """
    rep = arraydata.get_replicate_data(rep_name)
    xdata, curve, curve_se = arraydata.get_replicate_curves(rep_name)
    
    dG, dG_se = transform_2_dG(xdata, rep, curve, curve_se,
                               epsilon=epsilon)
    
    if plot_dG_se:
        perc = dG_se_perc_thresh
        fig, ax = plt.subplots()
        ax.axvline(np.percentile(dG_se.values.flatten(), perc), color='gray', linestyle=':')
        ax.axhline(perc/100, color='gray', linestyle=':')
        ax.text(np.percentile(dG_se.values.flatten(), perc), .5, '%d percentile threshold\nfor distinguishing outlier datapoints'%perc, ha='center')
        sns.ecdfplot(dG_se.values.flatten(), color='g')
        ax.set_xlim([0,2])

        ax.set_title('ECDF of dG s.e.')
        save_fig('./fig/alternative_fitting/z_ECDF_dG_se.pdf')
    
    return xdata, dG, dG_se, rep
    
    
def add_intercept(arr):
    """
    Helper function for fitting with LinearRegressionSVD
    """
    arr_colvec = arr.reshape(-1, 1)
    return np.concatenate((arr_colvec, np.ones_like(arr_colvec)), axis=1)
    
    
def get_inlier_mask_RANSAC(x, y, yerr, yerr_thresh,
                           dG_thresh=None):
    """
    Args:
        np.array col vectors
        dG_thresh - Tuple of len 2, values < f_thresh[0] or > f_thresh[1] will be cut as outliers
    Returns:
        flat arrays
    """
    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)
    inlier_mask = np.logical_and(ransac.inlier_mask_, yerr.flatten() < yerr_thresh)
    
    if dG_thresh is not None:
        inlier_mask[np.logical_or( y.flatten() < dG_thresh[0], y.flatten() > dG_thresh[1]).flatten()] = False
        
    # inlier_mask = yerr < yerr_thresh
    outlier_mask = np.logical_not(inlier_mask)
    
    return inlier_mask, outlier_mask
    
    
def fit_dG_line(x, y, yerr, inlier_mask, return_ols =True):
    """
    Fit inlier data points with OLS
    Args:
        x - col vector
    Returns:
        p - dict with fitted parameters and uncertainties
        line_X, line_y_ols - for plotting
    """
    # Fit the inlier data points with OLS
    ols = LinearRegressionSVD()
    ols.fit(add_intercept(x[inlier_mask]), y[inlier_mask], yerr[inlier_mask])

    # Predict data of estimated models
    line_X = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y_ols = ols.predict(add_intercept(line_X))

    dH = ols.coef_[1]
    dH_se = ols.coef_se_[1]
    dS = -ols.coef_[0]
    dS_se = ols.coef_se_[0]
    Tm = float(dH / dS - C2T)
    dH_ufloat = ufloat(dH, dH_se)
    Tm_ufloat = -dH_ufloat / ufloat(ols.coef_[0], ols.coef_se_[0])
    Tm_se = Tm_ufloat.std_dev()
    dG_37_ufloat = dH_ufloat * (1 - (37 + C2T) / Tm_ufloat)
    dG_37, dG_37_se = dG_37_ufloat.nominal_value, dG_37_ufloat.std_dev()

    p = dict(dH=dH, dH_se=dH_se, dS=dS, dS_se=dS_se,
             Tm=Tm, Tm_se=Tm_se, dG_37=dG_37, dG_37_se=dG_37_se)
    
    if return_ols:
        return p, ols
    else:
        return p, line_X, line_y_ols
 
    
def evaluate_linearity(x, y, yerr, inlier_mask, 
                       fit_figname=None, metric_figname=None,
                       drop_first_last=4, yerr_thresh=.5):

    mask = inlier_mask.copy().flatten()
    mask[:drop_first_last], mask[-drop_first_last:] = False, False
    mask[yerr.flatten() > yerr_thresh] = False

    x_values = x.flatten()[mask] - C2T
    y_values = y.flatten()[mask]
    res = [None for i in range(3)]
    
    lmod = lmfit.models.LinearModel(nanpolicy='omit')
    res[0] = lmod.fit(y_values, x=x_values)

    qmod = lmfit.models.QuadraticModel(nanpolicy='omit')
    res[1] = qmod.fit(y_values, x=x_values)

    cmod = lmfit.models.PolynomialModel(degree=3, nanpolicy='omit')
    params = cmod.make_params()
    params['c0'].set(value=-30)
    params['c1'].set(value=.1)
    params['c2'].set(value=0, min=-5, max=5)
    params['c3'].set(value=0, min=-5, max=5)
    res[2] = cmod.fit(y_values, params, x=x_values)
    
    if fit_figname is not None:
        fig, ax = plt.subplots(1,3,figsize=(9,3), sharey=True)
        for i in range(3):
            res[i].plot_fit(ax=ax[i])
        save_fig(fit_figname, fig)
        
    if metric_figname is not None:
        fig, ax = plt.subplots(2, 1, figsize=(4,5), sharex=True)
        bar_args = dict(width=.3, fill=True, facecolor='cadetblue')
        ax[0].bar(np.arange(3), [r.bic for r in res], **bar_args)
        ax[0].set_title('BIC')
        ax[1].bar(np.arange(3), [r.redchi for r in res], **bar_args)
        ax[1].margins(.2)
        ax[1].set_xticks(np.arange(3), ['linear', 'quadratic', 'cubic'])
        ax[1].set_title('reduced $\chi^2$')
        save_fig(metric_figname, fig)
        
    df = pd.DataFrame(index=['bic', 'redchi'], columns=['linear', 'quadratic', 'cubic'])
    df.loc['bic'] = [r.bic for r in res]
    df.loc['redchi'] = [r.redchi for r in res]
    
    return df
        
      
def fit_dG_lines(xdata, dG, dG_se, dG_se_perc_thresh=90, 
                 f_thresh=0.025, verbose=True):
    """
    Fit each variant in the dG dataframe.
    Args:
        dG_se and rep must contain all the SEQIDs in dG
        xdata in Kalvin
        f_thresh - float, margin to 0 and 1 in f_norm space for outliers to be determined
    """
    yerr_thresh = np.percentile(dG_se.values.flatten(), dG_se_perc_thresh)
    
    # prepare result_df
    param_col = ['dH', 'dS', 'dG_37', 'Tm']
    param_col += [s+'_se' for s in param_col]
    fit_col = ['rmse', 'rsqr', 'chisq', 'dof', 'redchi']
    bic_col = ['BIC%d' % i for i in [1,2,3]]
    columns = param_col + fit_col + bic_col
    result_df = pd.DataFrame(index=dG.index, columns=columns)
    
    x = xdata.reshape(-1,1)
    for seqid in tqdm(dG.index):
        y = dG.loc[seqid, :].values.reshape(-1,1)
        yerr = dG_se.loc[seqid, :].values.reshape(-1,1)
        # row = rep.loc[seqid, :]
        
        # get inlier mask with RANSAC, may swap out to other methods in the future
        if f_thresh is not None:
            dG_min, dG_max = dG_T(xdata, f_thresh).flatten(), dG_T(xdata, 1 - f_thresh).flatten()
            
        inlier_mask, _ = get_inlier_mask_RANSAC(x, y, yerr, yerr_thresh)
        
        # only fit if there are enough inlier datapoints
        if np.sum(inlier_mask) > 2:
            # fit the inlier data points
            p, ols = fit_dG_line(x, y, yerr, inlier_mask=inlier_mask, return_ols=True)
            # print(ols.metrics['rmse'])
            try:
                p, ols = fit_dG_line(x, y, yerr, inlier_mask=inlier_mask, return_ols=True)
        
                # pack into result dataframe
                result_df.loc[seqid, param_col] = p
                result_df.loc[seqid, fit_col] = {s: ols.metrics[s] for s in fit_col}
            except:
                print(seqid, 'has problem with fitting dG line')
                
            # evaluate linearity with model selection
            # inlier_mask is modified for higher stringency
            try:
                linearity_metric_df = evaluate_linearity(x, y, yerr, inlier_mask,
                                                         drop_first_last=3)
                result_df.loc[seqid, bic_col] = linearity_metric_df.loc['bic', :].values
            except:
                if verbose:
                    print(seqid + ': Problem with fitting linear, quadratic and cubic models')
                
    
    # print some sanity checks
    result_nan = np.isnan(result_df.values.astype(float))
    # print('%.3f%% variants have nan(s)' % (np.sum(np.sum(result_nan, axis=1).astype(bool)) / result_nan.shape[0]))
    print('%.3f%% variants have nan(s) in fitted parameters' % (np.sum(np.sum(result_nan[:, :8], axis=1).astype(bool)) / result_nan.shape[0]))
        
    return result_df
    
    
def plot_dG_fit(seqid:str, xdata:np.array, dG:pd.DataFrame, dG_se:pd.DataFrame, 
                rep:pd.DataFrame, curve:pd.DataFrame, #curve_norm:pd.DataFrame,
                dG_se_perc_thresh:float=90, verbose:bool=True):
    """
    Fit dG to one variant and plot
    """
    curve_function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/kB*((Tm+273.15)**-1 - x**-1)))
    yerr_thresh = np.percentile(dG_se.values.flatten(), dG_se_perc_thresh)
        
    x = xdata.reshape(-1,1)
    y = dG.loc[seqid, :].values.reshape(-1,1)
    yerr = dG_se.loc[seqid, :].values.reshape(-1,1)
    row = rep.loc[seqid, :]
    ycurve = curve.loc[seqid, :].values.flatten()
    # ycurve_norm = curve_norm.loc[seqid, :].values.flatten()
    
    # get inlier mask with RANSAC, may swap out to other methods in the future
    inlier_mask, outlier_mask = get_inlier_mask_RANSAC(x, y, yerr, yerr_thresh)
    
    # only fit if there are enough inlier datapoints
    if np.sum(inlier_mask) <= 2:
        print('Only %d inlier datapoints. Not fitted.' % np.sum(inlier_mask))
    else:
        # fit the inlier data points
        p, ols = fit_dG_line(x, y, yerr, inlier_mask=inlier_mask, return_ols=True)
    
        # evaluate linearity with model selection
        linearity_metric_df = None
        try:
            linearity_metric_df = evaluate_linearity(x, y, yerr, inlier_mask)
        except:
            print(seqid + ': Problem with fitting linear, quadratic and cubic models')
            
        # get the parameters and lines to plot
        dH_curve = row['dH']
        dS_curve = row['dS']
        Tm_curve = row['Tm']
        dG_37_curve = row['dG_37']
        
        line_X = np.arange(xdata.min(), xdata.max())[:, np.newaxis]
        line_y_ols = ols.predict(add_intercept(line_X))
        line_y_curve = -(dH_curve / (Tm_curve + C2T)) * line_X + dH_curve

        pred_curve = curve_function(p['dH'], p['Tm'], row['fmax'], row['fmin'], xdata)
        
        # plot the fit
        fig, ax = plt.subplots(2, 2, figsize=(10,7.5))
        ax = ax.flatten()
        # panel 1: dG vs T
        ax[0].axhline(y=0, c='gray', linestyle=':')
        ax[0].errorbar(xdata[inlier_mask] - C2T, y.flatten()[inlier_mask], yerr=yerr.flatten()[inlier_mask],
                    color='cornflowerblue', fmt='.', label='inlier')
        ax[0].scatter(xdata[outlier_mask] - C2T, y.flatten()[outlier_mask], 
                    marker='x', color='r', label='outlier')
        ax[0].plot(
            line_X - C2T,
            line_y_ols,
            color="cornflowerblue",
            linewidth=2,
            label="from dG fitting",
        )
        ax[0].plot(
            line_X - C2T,
            line_y_curve,
            color="c",
            linewidth=2,
            label="from curve fitting",
        )
        ax[0].legend(loc='best')
        ax[0].set_title('$\Delta G_T$')
        ax[0].set_ylabel('kcal/mol')
        ax[0].set_xlabel('temperature (°C)')
        print('dG fitting $\chi^2$ = %.2f, reduced = %.2f, curve fitting $\chi^2$ = %.2f$' % (ols.chisq, ols.redchi, row.chisq / row.dof))
        # panel 2: original signal space
        plotting.plot_actual_and_expected_fit(rep.loc[seqid,:].squeeze(),
                                            ax=ax[1], c='c')
        ax[1].scatter(xdata[outlier_mask] - C2T, ycurve[outlier_mask], 
                    marker='x', color='r', zorder=10, label='outlier')
        ax[1].plot(xdata - C2T, pred_curve,
                'cornflowerblue', label='from dG fitting')
        ax[1].legend(loc='best')
        ax[1].set_ylabel('a.u.')
        ax[1].set_xlabel('temperature (°C)')
        ax[1].set_title('normalized fluorescence')

        # panel 3: fitted parameters
        n_p = 3
        param_err = np.array([[row['dH_lb'], row['dH_ub']],
                              [row['dS_lb'], row['dS_ub']],
                              [row['dG_37_lb'], row['dG_37_ub']]]).squeeze().T
        param_err_dG = [p['dH_se'], p['dS_se'], p['dG_37_se']]
        errorbar_args = dict(fmt='_', markersize=10, capsize=3)
        ax[2].errorbar(np.arange(n_p) - .1, np.array((dH_curve, dS_curve, dG_37_curve)).squeeze(), yerr=param_err, 
                    c='c', label='curve fitting', **errorbar_args)
        ax[2].errorbar(np.arange(n_p) + .1, np.array((p['dH'], p['dS'], p['dG_37'])).squeeze(), yerr=param_err_dG,
                    c='cornflowerblue', label='dG fitting', **errorbar_args)
        values = np.array([dH_curve, dS_curve, dG_37_curve, p['dH'], p['dS'], p['dG_37']])
        for xx,yy,value,c in zip(np.concatenate((np.arange(n_p) - .35, np.arange(n_p) + .35)),
                            values,
                            ['%.2f'%n for n in values],
                            ['c']*n_p + ['cornflowerblue']*n_p):
            ax[2].text(xx, yy, value, c=c, ha='center', va='center')
            
        ax[2].set_xticks(np.arange(n_p), ['dH', 'dS', '$dG_{37}$'])
        ax[2].set_ylabel('kcal/mol')
        ax[2].margins(x=.4)

        ax[3].errorbar(np.arange(1) - .08, np.array((Tm_curve)).squeeze(), yerr=np.array([row['Tm_lb'], row['Tm_ub']]).reshape(2,1), 
                    c='c', label='curve fitting', **errorbar_args)
        ax[3].errorbar(np.arange(1) + .08, np.array((p['Tm'])).squeeze(), yerr=p['Tm_se'],
                    c='cornflowerblue', label='dG fitting', **errorbar_args)

        values = np.array([Tm_curve, p['Tm']])
        for xx,yy,value,c in zip(np.concatenate((np.arange(1) - .35, np.arange(1) + .35)),
                            values,
                            ['%.2f'%n for n in values],
                            ['c'] + ['cornflowerblue']):
            ax[3].text(xx, yy, value, c=c, ha='center', va='center')
        
        ax[3].set_xticks(np.arange(1), ['Tm'])
        ax[3].set_ylabel('°C')
        ax[3].margins(x=8)
        ax[3].legend()
        
        plt.suptitle(seqid)
        sns.despine()

        # util.save_fig('./fig/alternative_fitting/%s.pdf'%y.index[0])
        # Evaluate linearity
        mask = inlier_mask.copy()
        # mask[:4], mask[-4:] = False, False
        mask[yerr.flatten() > .5] = False

        x_values = xdata[mask] - C2T
        y_values = y.flatten()[mask]
        res = [None for i in range(3)]
        lmod = lmfit.models.LinearModel(nanpolicy='omit')
        res[0] = lmod.fit(y_values, x=x_values)

        qmod = lmfit.models.QuadraticModel(nanpolicy='omit')
        res[1] = qmod.fit(y_values, x=x_values)

        cmod = lmfit.models.PolynomialModel(degree=3, nanpolicy='omit')
        params = cmod.make_params()
        params['c0'].set(value=-30)
        params['c1'].set(value=.1)
        params['c2'].set(value=0, min=-5, max=5)
        params['c3'].set(value=0, min=-5, max=5)
        res[2] = cmod.fit(y_values, params, x=x_values)

        fig, ax = plt.subplots(1,3,figsize=(9,3), sharey=True)
        for i in range(3):
            res[i].plot_fit(ax=ax[i])
            
        fig, ax = plt.subplots(2, 1, figsize=(4,5), sharex=True)
        bar_args = dict(width=.3, fill=True, facecolor='cadetblue')
        ax[0].bar(np.arange(3), [r.bic for r in res], **bar_args)
        ax[0].set_title('BIC')
        ax[1].bar(np.arange(3), [r.redchi for r in res], **bar_args)
        ax[1].margins(.2)
        ax[1].set_xticks(np.arange(3), ['linear', 'quadratic', 'cubic'])
        ax[1].set_title('reduced $\chi^2$')


def line_fit_all_replicates(version='v0.0.2'):
    """
    A little script to do dG fit on all of the replicates in arraydata
    Save as tsv files to the fitted_variant folder.
    """
    annotation_file = './data/annotation/NNNlib2b_annotation_20220519.tsv'
    replicate_df = pd.read_table('./data/nnnlib2b_replicates.tsv')
    arraydata = arraydata.ArrayData(replicate_df=replicate_df.iloc[:4,:],
                        annotation_file=annotation_file)
    
    rep_names = replicate_df['name'].tolist()
    rep_names.remove('salt')
    for rep_name in rep_names:
        print(rep_name)
        xdata, dG, dG_se, rep = get_rep_dG(arraydata, rep_name)
        result_df = fit_dG_lines(xdata, dG.sample(), dG_se, dG_se_perc_thresh=90, verbose=False)
        result_df.to_csv('./data/fitted_variant/dG_fit_%s_%s.tsv' % (replicate_df.set_index('name').loc[rep_name, 'replicate'], version), sep='\t')