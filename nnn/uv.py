"""
Functions for UV melting analysis
"""
import numpy as np
import pandas as pd
from scipy import stats
import os, json
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from tqdm import tqdm
from lmfit import minimize, Minimizer, Parameters, report_fit
from scipy.interpolate import interp1d
from scipy import signal
from hampel import hampel
from .util import *
from . import processing

import warnings
warnings.filterwarnings("ignore")

def get_blanked_fn(fn):
    split_fn = os.path.splitext(fn)
    blanked_fn = split_fn[0] + '_blanked' + split_fn[1]
    return blanked_fn

def query_curve_in_df(df, curve_name):
    """
    Args:
        curve_name - Dict
    """
    row = df.query("curve_date == '%s' & curve_num == '%s'" % (curve_name['curve_date'], curve_name['curve_num']))
    return row

def lookup_sample_df(df, df_ref, key):
    # looks up `key` in `df_ref`
    return df.apply(lambda row: df_ref.query("curve_date == '%s' & curve_num == '%s'" % (row['curve_date'], row['curve_num']))[key].values[0], axis=1)

def find_blank_reference_curve_str(curve_str: str, blank_to):
    if not np.isnan(blank_to):
        curve_str_list = curve_str.split('_')
        blank_curve_str = '%s_%d_%s' % (curve_str_list[0], blank_to, curve_str_list[2])
    else:
        blank_curve_str = curve_str
    return blank_curve_str

def format_cd_data(fn):
    """
    Format CD txt data to the same as from ECL
    """
    curve = pd.read_table(fn, header=None)

    curve.columns = ['celsius', 'cd', 'v', 'absorbance']
    curve.drop(columns = ['cd', 'v'], inplace=True)

    curve.to_csv(fn.replace('.txt', '.csv'), index=False, header=None)
    

def read_curve(fn):
    curve = pd.read_csv(fn, header=None)
    curve.columns = ['celsius', 'absorbance']
    curve.sort_values(by='celsius', inplace=True)
    return curve
    
def read_sample_sheet(fn):
    sample_df = pd.read_csv(fn, index_col=0)
    sample_df['curve_date'] = sample_df['curve_date'].astype(str)
    sample_df['curve_num'] = sample_df['curve_num'].astype(str)
    
    #----- fill out default settings -----
    sample_df.celsius_min.fillna(sample_df.MinTemperature, inplace=True)
    sample_df.celsius_max.fillna(sample_df.MaxTemperature, inplace=True)
    # sample_df.loc[np.logical_and(sample_df.Blank == "manual", sample_df.SEQID != 'blank'), "BlankTo"].fillna(7)
    sample_df.loc[:,"BlankTo"].fillna(int(7), inplace=True)
    
    return sample_df

def parse_curve_name(fn):
    curve_date = fn.split('/')[-2].split('_')[0]
    curve_num = fn.split('/')[-1].split('_')[0]
    curve_name = fn.split('/')[-1].split('_')[1].split('.csv')[0]
    curve_str = f'{curve_date}_{curve_num}_{curve_name}'
    return dict(curve_date=curve_date, curve_num=curve_num, 
                curve_name=curve_name, curve_str=curve_str)

def format_fit_result(out):
    result_dict = {}
    for p in ('dH', 'Tm', 'fmax', 'fmin', 's1', 's2'):
        result_dict[p] = out.params[p].value
        result_dict[p+'_std'] = out.params[p].stderr
    result_dict['rmse'] = np.sqrt(np.mean(np.square(out.residual)))
    return result_dict

def plot_curve_basic(curve):
    plt.plot(curve.celsius, curve.absorbance, '.')

def plot_curve_fit_result(row, return_curve=False):
    fn = row['data_file']
    blanked_fn = get_blanked_fn(fn)
    if os.path.isfile(blanked_fn):
        fn = blanked_fn
    curve = read_curve(fn)
    
    curve_predict = curve_model(curve.celsius, **{x:row[x] for x in ['dH','Tm','fmax','fmin','s1', 's2']})
    plt.plot(curve.celsius, curve.absorbance, '.')
    plt.plot(curve.celsius, 
             curve_predict - curve_predict[0] + curve.absorbance[0])
    plt.axvline(row['Tm'], linestyle='--', c='gray')
    
    if 'SEQID' in row.index:
        plt.title('%s %s' % (row['SEQID'], row['curve_name']))
    else:
        plt.title('%s' % (row['data_file']))
    sns.despine()
    plt.show()
    
    if return_curve:
        return curve
    
def plot_curve_preview_of_datadir(datadir:str, sample_sheet_file:str, plot_fn:str=None):
    """
    Plot raw absorbance of all curves in a single pdf file for preview and troubleshotting.
    Upper row: raw
    Lower row: if maunal blank, the blanked curves
    """
    color_dict = dict(
        MeltingCurve='#a6cee3',
        CoolingCurve='#1f78b4',
        SecondaryMeltingCurve='#b2df8a',
        SecondaryCoolingCurve='#33a02c',
        TertiaryMeltingCurve='#fb9a99',
        TertiaryCoolingCurve='#e31a1c'
    )
    
    if plot_fn is None:
        plot_fn = os.path.join(datadir, "preview_raw_absorbance_all_curves.pdf")
        
    data_list = [fn for fn in absolute_file_paths(datadir) if fn.endswith('.csv')]
    sample_sheet = read_sample_sheet(sample_sheet_file)#.iloc[-21:,:]
    
    result_df = make_empty_result_df(data_list, sample_sheet, blank=True)
    result_df.sort_values(by=['curve_date', 'curve_num'], inplace=True)
    result_df['SEQID'] = lookup_sample_df(result_df, sample_sheet, 'SEQID')
    result_df['Blank'] = lookup_sample_df(result_df, sample_sheet, 'Blank')
    curve_dates = np.unique(result_df.curve_date)
    figs = [None for _ in range(len(curve_dates))]
    
    for i,curve_date in enumerate(curve_dates):
        formatted_curve_date = '%s-%s-%s %s:00' % (curve_date[:2], curve_date[2:4], curve_date[4:6], curve_date[6:])
        figs[i], ax = plt.subplots(2,7,figsize=(12,4), sharex=False, sharey=False)
        # ax=ax.flatten()
        plt.suptitle(formatted_curve_date)
        exp_df = result_df.query('curve_date == "%s"'%curve_date)
        for j,row in exp_df.iterrows():
            # raw curves
            curve = read_curve(row.data_file)
            curve_num = int(row.curve_num)
            curve_color = color_dict[row.curve_name]
            ax[0,curve_num-1].plot(curve.celsius, curve.absorbance, 
                                 c=curve_color, linewidth=2)
            try:
                seqid = row['SEQID']
            except:
                seqid = ""
            ax[0,curve_num-1].set_title(seqid)
            
            # blanked curves
            if row.Blank == 'manual':
                try:
                    blank_curve = read_curve(exp_df.query('curve_num == "7" & curve_name == "%s"'%row.curve_name).data_file.values[0])
                    blanked_curve = curve.absorbance - blank_curve.absorbance
                    ax[1,curve_num-1].plot(curve.celsius, blanked_curve,
                                           c=curve_color, linewidth=2, linestyle=':')
                except:
                    pass
                
            sns.despine()
            
        _ = [ax[1,i].set_xlabel('Celsius (°C)') for i in range(7)]
        _ = [ax[i,0].set_ylabel('Absorbance') for i in range(2)]
                     
    save_multi_image(plot_fn, figs)
    
    return result_df

def get_2nd_diff_autocorrelation(y, plot=False):
    y = y - np.mean(y)
    yacorr = np.correlate(y, y, 'full')[:len(y)]
    yacorr = yacorr / np.var(y) / len(y)
    yacorr_2diff = np.diff(yacorr, n=2)
    if plot:
        _,ax = plt.subplots(1,3, figsize=(12,2))
        ax[0].plot(y)
        ax[1].plot(yacorr)
        ax[2].plot(yacorr_2diff)
        
    return yacorr_2diff

def qc_blank_curve(curve):
    isgood = True
    zscore = np.nanstd(curve.absorbance) / np.nanmedian(curve.absorbance)
    if zscore > 0.2:
        isgood = False
    acorr_2diff = get_2nd_diff_autocorrelation(curve.absorbance)
    if np.max(np.abs(acorr_2diff)) > 0.005:
        isgood = False
    return isgood

### Directly fit the curves ###
def curve_model(x, dH, Tm, fmin, fmax, s1, s2):
    # define the function
    return fmin + s1 * x + (((s2 - s1) * x + fmax - fmin)/(1 + np.exp(dH /0.0019872 * ((Tm + 273.15)**(-1) - (x + 273.15)**(-1)))))

def residual(pars, x, data):
    dH, Tm, fmax, fmin, s1, s2 = pars['dH'], pars['Tm'], pars['fmax'], pars['fmin'], pars['s1'], pars['s2']
    model = curve_model(x, dH, Tm, fmin, fmax, s1, s2)
    return model - data

def fit_param_direct(curve, Tm=None, celsius_min=5, celsius_max=95, smooth=True, plot_title=''):
    pfit = Parameters()
    data_max = np.max(curve.absorbance)
    data_min = np.min(curve.absorbance)
    pfit.add(name='dH', value=-20)
    if Tm is None:
        pfit.add(name='Tm', value=(celsius_max + celsius_min) * 0.5)
    else:
        pfit.add(name='Tm', value=Tm, vary=False)
        
    pfit.add(name='fmax', value=2*data_max, min=data_min, max=20*data_max)
    pfit.add(name='fmin', value=max(0.2*data_min, 0.01), min=-0.1, max=data_max)
    pfit.add(name='s1', value = 1e-5, max=5.0, min=-1.0)
    pfit.add(name='s2', value = 1e-5, max=2.0, min=-2.0)
    # pfit.add(name='delta_f', )

    curve_used = curve.query(f'celsius >= {celsius_min} & celsius <= {celsius_max}')
    if smooth:
        # Anomaly detection with hampel
        outlier_idx = hampel(curve_used.loc[:, 'absorbance'])
        curve_used.drop(index = curve_used.index[outlier_idx], inplace=True)

    out_tmp = minimize(residual, pfit, args=(curve_used.celsius,), 
                   kws={'data': curve_used.absorbance})
    
    pfit['Tm'].set(vary=True)
    for p in ['dH', 'fmin', 'fmax', 's1', 's2']:
        pfit[p].set(value=out_tmp.params[p].value)
        
    out = minimize(residual, pfit, args=(curve_used.celsius,), 
                   kws={'data': curve_used.absorbance})
    
    best_fit = curve_used.absorbance + out.residual
    
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(curve.celsius, curve.absorbance, '+', c='purple')
    ax.plot(curve_used.celsius, curve_used.absorbance, 'x', c='g')
    ax.plot(curve_used.celsius, best_fit, 'orange', linewidth=2.5)
    ax.set_xlabel('temperature (°C)')
    ax.set_ylabel(r'absorbance')
    ax.set_title(plot_title)
    sns.despine()
    
    return out


### Use the d_absorbance method ###
def fit_Tm_d_absorbance(curve, celsius_min=5, celsius_max=95, whatever=False):
    """
    For a first pass before direct fit. Only fits Tm because fmax and fmin are unknown.
    """
    # == Calculate d_p_unfold ==
    curve_used = curve.query(f'celsius >= {celsius_min} & celsius <= {celsius_max}').sort_values(by='celsius')
    acorr_2diff = get_2nd_diff_autocorrelation(curve_used.absorbance)

    if (not whatever) and (curve_used.absorbance.isnull().values.any() or np.max(np.abs(acorr_2diff)) > 0.005):
        # run away fast
        # if has bad values or jumps (intervention) in the series
        return dict(is_usable=False, Tm=np.nan)
        
    # Upsampled 10x
    x = np.arange(curve_used.celsius.iloc[0], curve_used.celsius.iloc[-1], 0.1)
    signal_used = signal.savgol_filter(curve_used.absorbance, 9, 3, mode='nearest')
    f = interp1d(curve_used.celsius, 
                 signal_used, 
                 kind='cubic')
    # dx/dT = dx/0.1, where dT = 0.1 from upsampling
    d_p_unfold = np.diff(f(x)) * 10
    d_p_unfold = signal.savgol_filter(d_p_unfold, 99, 3)
    
    # == Find a reasonable range to use ==
    is_usable = True
    d_p_sign = np.sign(d_p_unfold)
    # tag BAD if more than a certain part of the curve is decreasing
    if np.count_nonzero(d_p_sign == -1) > 0.2 * len(d_p_unfold):
        is_usable = False
    
    # common problem: decreasing at high temperature
    # solution: overwrite celsius_max when it first starts to decrease monotomically
    # doesn't affect
    win_len = 15 # 1.5 °C
    local_sum = np.convolve(d_p_sign, np.ones(win_len), 'valid')
    # if not found, celsius_max_idx will be 0
    celsius_max_idx = np.argmax(local_sum <= win_len + 1)
    celsius_min_idx = np.argmax(local_sum >= win_len - 1)
    if celsius_max_idx > 0:
        celsius_max = x[celsius_max_idx]
    if celsius_min_idx > 0:
        celsius_min = x[celsius_min_idx]
    
    # == Call peaks ==
    peaks = signal.find_peaks(d_p_unfold)[0]
    try:
        ind_max = np.argmax(d_p_unfold[peaks])
        Tm = x[peaks[ind_max]]
    except:
        Tm = np.nan
    
    if whatever:
        is_usable = True
        
    result_dict = dict(Tm=Tm, is_usable=is_usable, celsius_min=celsius_min, celsius_max=celsius_max)
    return result_dict

def fit_param_d_absorbance(curve, out, celsius_min=5, celsius_max=95, smooth=True, plot_title=''):
    """
    Deprecated. Other parameters than Tm are not very reliable from this method.
    """
    curve_used = curve.query(f'celsius >= {celsius_min} & celsius <= {celsius_max}').sort_values(by='celsius')
    x = np.arange(curve_used.celsius.iloc[0], curve_used.celsius.iloc[-1], 0.1)
    signal_used = signal.savgol_filter(curve_used.absorbance, 9, 3, mode='nearest')
    signal_used = (signal_used - out.params['fmin'] - out.params['slope'] * curve_used.celsius.values) / (out.params['fmax'] - out.params['fmin'])
    f = interp1d(curve_used.celsius, 
                 signal_used, 
                 kind='cubic')
    d_p_unfold = np.diff(f(x)) * 10
    if smooth:
        d_p_unfold = signal.savgol_filter(d_p_unfold, 99, 3)
    peaks = signal.find_peaks(d_p_unfold)[0]
    ind_max = np.argmax(d_p_unfold[peaks])
    peak = peaks[ind_max]
    
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(x[:-1], d_p_unfold, 'k')
    ax.axvline(x=x[peak], ls='--', c='gray')
    ax.set_xlabel('temperature (°C)')
    ax.set_ylabel(r"$p_{unfold}'$")
    ax.set_title(plot_title)
    sns.despine()
    
    result = {}
    result['Tm_diff'] = x[peak]
    result['dH_diff'] = - d_p_unfold[peak] * 4 * 0.0019872 * (result['Tm_diff'] + 273.15)**2
    result['dS_diff'] = result['dH_diff'] / (result['Tm_diff'] + 273.15)
    result['dG_37_diff'] = get_dG(result['dH_diff'], result['Tm_diff'], 37)
    result['rmse_diff'] = rmse(curve_model(curve_used.celsius, result['dH_diff'], result['Tm_diff'], out.params['fmin'], out.params['fmax'], out.params['s1'], out.params['s2']),
                                           curve_used.absorbance.values)
    
    return result


### Master Function ###
def fit_curve(fn, figdir='', verbose=False, debug=False, 
              blank=None, **kwargs):
    def fit():
        curve = read_curve(fn)
        if isinstance(blank, pd.DataFrame):
            curve['absorbance'] -= blank['absorbance']
        elif isinstance(blank, float):
            curve['absorbance'] -= blank
        
        # Shift up so all values are positive
        curve['absorbance'] = curve['absorbance'] - np.min(curve['absorbance'].values) + 0.01
        
        # Save the blanked curve to disk
        blanked_fn = get_blanked_fn(fn)
        curve.to_csv(blanked_fn, index=False, header=False)
            
        curve_name = parse_curve_name(fn)
        if verbose:
            print(curve_name['curve_str'])
        d_absorbance_result_dict = fit_Tm_d_absorbance(curve, **kwargs)
        if not d_absorbance_result_dict['is_usable'] or np.isnan(d_absorbance_result_dict['Tm']):
            # give up fast and run away
            raise Exception("Sorry, curve is too crazy for %s"%fn)
            
        Tm = d_absorbance_result_dict['Tm']
        kwargs['celsius_max'] = d_absorbance_result_dict['celsius_max']
        out = fit_param_direct(curve, Tm=Tm, 
                               plot_title=curve_name['curve_str'], **kwargs)
        save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_direct_fit.png"))
        # result = fit_param_d_absorbance(curve, out, plot_title=curve_name['curve_str'], **kwargs)
        # save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_d_p_unfold.png"))
        result_dict = format_fit_result(out)
        result_dict.update(kwargs)
        result_dict.update(curve_name)
        result_dict['data_file'] = fn
        if verbose:
            print('\tDone!')
        return result_dict
        
    if debug:
        result_dict = fit()
        return result_dict
    else:
        try:
            result_dict = fit()
            return result_dict
        except:
            # print("Trouble with", fn)
            return dict()

def fit_cd_curve(fn, figdir='', verbose=False, debug=False, 
              **kwargs):
    def fit():
        curve = pd.read_table(fn, header=None)
        curve.columns = ['celsius', 'cd', 'v', 'absorbance']
        curve.drop(columns = ['cd', 'v'], inplace=True)
                    
        curve_str = os.path.splitext(os.path.split(fn)[-1])[0]
        curve_name = dict(curve_str=curve_str,
                          curve_name=curve_str.split('-')[1],
                          seqid=curve_str.split('-')[0])
        if verbose:
            print(curve_name['curve_str'])
        d_absorbance_result_dict = fit_Tm_d_absorbance(curve, whatever=True, **kwargs)
        if not d_absorbance_result_dict['is_usable'] or np.isnan(d_absorbance_result_dict['Tm']):
            print(d_absorbance_result_dict)
            # give up fast and run away
            raise Exception("Sorry, curve is too crazy for %s"%fn)
            
        Tm = d_absorbance_result_dict['Tm']
        kwargs['celsius_max'] = d_absorbance_result_dict['celsius_max']
        out = fit_param_direct(curve, Tm=Tm, 
                               plot_title=curve_name, **kwargs)
        save_fig(os.path.join(figdir, f"{curve_name['curve_str']}_direct_fit.png"))
        # result = fit_param_d_absorbance(curve, out, plot_title=curve_name['curve_str'], **kwargs)
        # save_fig(os.path.join(figdir, curve_name['curve_date'], f"{curve_name['curve_num']}_{curve_name['curve_name']}_d_p_unfold.png"))
        result_dict = format_fit_result(out)
        result_dict.update(kwargs)
        result_dict.update(curve_name)
        result_dict['data_file'] = fn
        if verbose:
            print('\tDone!')
        return result_dict
        
    if debug:
        result_dict = fit()
        return result_dict
    else:
        try:
            result_dict = fit()
            return result_dict
        except:
            # print("Trouble with", fn)
            return dict()

def fit_all_cd_curves(datadir):
    datafiles = [x for x in os.listdir(datadir) if x.endswith('.txt')]
    
    result_columns = ['curve_str', 'seqid', 'curve_name', 
                'dH', 'dH_std', 'Tm', 'Tm_std', 
                'fmax', 'fmax_std', 'fmin', 'fmin_std', 
                's1', 's1_std', 's2', 's2_std', 'rmse',
                'celsius_min', 'celsius_max', 'data_file']
    result_df = pd.DataFrame(index=np.arange(len(datafiles)), columns=result_columns)
    
    for i,fn in enumerate(datafiles):
        result_dict = fit_cd_curve(os.path.join(datadir,fn), figdir=os.path.join(datadir, 'fig'), 
                        debug=True, verbose=True)
        result_df.iloc[i, :] = result_dict
        
    return result_df.sort_values(by=['seqid', 'curve_name'])
   
def make_empty_result_df(data_list, sample_sheet: pd.DataFrame, blank: bool=False):
    #----- make the index and column names for result_df -----
    result_index = []
    curve_date, curve_num, curve_name, blank_to_list, data_file = [], [], [], [], []
    
    for fn in data_list:
        curve_dict = parse_curve_name(fn)
        row = query_curve_in_df(sample_sheet, curve_dict)
        if len(row) > 0:
            result_index.append(curve_dict['curve_str'])
            curve_date.append(curve_dict['curve_date'])
            curve_num.append(curve_dict['curve_num'])
            curve_name.append(curve_dict['curve_name'])
            data_file.append(fn)
            if row.Blank.values[0] == 'manual':
                blank_to_list.append(find_blank_reference_curve_str(curve_dict['curve_str'], row.BlankTo.values[0]))
            else:
                blank_to_list.append('no_manual_blank')
        elif len(row) == 0:
            data_list.remove(fn)

    result_columns = ['curve_date', 'curve_num', 'curve_name',
                    'dH', 'dH_std', 'Tm', 'Tm_std', 
                    'fmax', 'fmax_std', 'fmin', 'fmin_std', 
                    's1', 's1_std', 's2', 's2_std', 'rmse',
                    'celsius_min', 'celsius_max', 'data_file']

    result_df = pd.DataFrame(index=result_index, columns=result_columns)
    result_df.curve_date = np.array(curve_date, dtype=str)
    result_df.curve_num = np.array(curve_num, dtype=str)
    result_df.curve_name = curve_name
    result_df.data_file = data_file

    if blank:
        result_df['blank'] = blank_to_list
    
    return result_df
    

### Fit from sample_sheet ###
def fit_all_manual_blank(datadir:str, sample_sheet_file:str, result_file:str='uvmelt.csv',
                         qc_criterion = 'rmse < 0.015 & dH_std < 10 & Tm_std < 5'):
    """
    This function blanks all the curves first before fitting.
    If not manual blank or cannot find blank, use the original curve.
    Reads and caches all blank data at once first.
    """

    #----- Read files -----
    sample_sheet = read_sample_sheet(sample_sheet_file)#.query("Blank == 'manual'")
    data_list = [fn for fn in absolute_file_paths(datadir) if (fn.endswith('.csv') and (not fn.endswith('_blanked.csv')))]
    
    #----- make the index and column names for result_df -----
    result_df = make_empty_result_df(data_list, sample_sheet, blank=True)
    
    #----- read blank curves -----
    all_blanks = np.unique(result_df['blank'])
    blank_dict = dict()
    for blank_str in all_blanks:
        if not blank_str in result_df.index:
            # check the blanks are in the dataset
            print( "blank data %s not in the dataset!" % blank_str )
            blank_dict[blank_str] = 0
        else:
            blank_fn = data_list[result_df.index.to_list().index(blank_str)]
            blank_dict[blank_str] = read_curve(blank_fn)
            # QC blank curvel
            if not qc_blank_curve(blank_dict[blank_str]):
                blank_dict[blank_str] = np.nan # throw affected curves away
            
    #----- fit curves -----
    for fn in tqdm(data_list):
        curve_name = parse_curve_name(fn)
        row = query_curve_in_df(sample_sheet, curve_name)

        if len(row) == 0 or (curve_name['curve_str'] in all_blanks) or (not row['Usable'].values[0]):
            continue
        else:
            try:
                blank_curve = blank_dict[result_df.loc[curve_name['curve_str'], 'blank']]
            except:
                blank_curve = 0
                
            result_dict = fit_curve(fn, figdir=os.path.join(datadir,'fig'), 
                                    blank=blank_curve,
                                    debug=False,
                                    celsius_min=row.at[row.index[0],'celsius_min'],
                                    celsius_max=row.at[row.index[0],'celsius_max'])
            result_df.loc[curve_name['curve_str'], :] = result_dict
            
    result_df.dropna(subset=['dH', 'Tm', 'rmse'], inplace=True)
        
    result_df['pass_qc'] = result_df.eval(qc_criterion)
    
    result_df['SEQID'] = ''
    for col in ['SEQID', 'conc_uM', 'Na_mM', 'celsius_min', 'celsius_max', 'Cuvette']:
        result_df[col] = lookup_sample_df(result_df, sample_sheet, col)
        
    result_df['dG_37'] = get_dG(result_df['dH'], result_df['Tm'], celsius=37)
    result_df['dS'] = result_df['dH'] / (result_df['Tm'] + 273.15)
    result_df.to_csv(result_file)
    
    return result_df
        
    

def fit_all_no_blank(datadir:str, sample_sheet_file:str, result_file:str='uvmelt.csv'):
    #----- Hardcoded QC -----
    qc_criterion = 'rmse < 0.015 & dH_std < 10 & Tm_std < 5'
    
    #----- Read files -----
    sample_sheet = read_sample_sheet(sample_sheet_file).query("Blank != 'manual'")
    data_list = [fn for fn in absolute_file_paths(datadir) if fn.endswith('.csv')]
    
    #----- make the index and column names for result_df -----
    result_df = make_empty_result_df(data_list, sample_sheet)

    #----- fit curves -----
    for fn in tqdm(data_list):
        curve_name = parse_curve_name(fn)
        row = query_curve_in_df(sample_sheet, curve_name)

        if len(row) == 0:
            print(fn)
            continue
        else:
            result_dict = fit_curve(fn, figdir=os.path.join(datadir,'fig'), 
                                    celsius_min=row.at[row.index[0],'celsius_min'],
                                    celsius_max=row.at[row.index[0],'celsius_max'])

            result_df.loc[curve_name['curve_str'], :] = result_dict
            
    result_df.dropna(inplace=True)
        
    result_df['pass_qc'] = result_df.eval(qc_criterion)
    
    for col in ['SEQID', 'conc_uM', 'Na_mM', 'celsius_min', 'celsius_max']:
        result_df[col] = lookup_sample_df(result_df, sample_sheet, col)
        
    result_df['dG_37'] = get_dG(result_df['dH'], result_df['Tm'], celsius=37)
    result_df['dS'] = result_df['dH'] / (result_df['Tm'] + 273.15)
    result_df['isCooling'] = result_df.curve_name.apply(lambda x: 'Cooling' in x)
    result_df.to_csv(result_file)
    
    #----- Plot QC -----
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    sns.scatterplot(data=result_df, 
                    x='rmse', y='Tm_std', color='gray',
                    ax=ax[0])
    sns.scatterplot(data=result_df.query(qc_criterion), 
                    x='rmse', y='Tm_std', color='salmon',
                    ax=ax[0])
    ax[0].axhline(5, linestyle='--', c='gray')
    ax[0].axvline(0.015, linestyle='--', c='gray')
    ax[1].set_ylim([0, 100])

    sns.scatterplot(data=result_df, 
                    x='rmse', y='dH_std', color='gray',
                    ax=ax[1])
    sns.scatterplot(data=result_df.query(qc_criterion), 
                    x='rmse', y='dH_std', color='salmon',
                    ax=ax[1])
    ax[1].set_ylim([-5, 100])
    ax[1].axvline(0.015, linestyle='--', c='gray')
    ax[1].axhline(10, linestyle='--', c='gray')
    sns.despine()

    plt.suptitle('%.2f%% (%d / %d) passed QC' % (100 * result_df.eval(qc_criterion).sum() / len(result_df), result_df.eval(qc_criterion).sum(), len(result_df)))
    save_fig(os.path.join(datadir, 'fig', 'QC.pdf'), fig)
    plt.show()
    
    return result_df
    
def fit_all(datadir:str, sample_sheet_file:str, result_file:str='uvmelt.csv'):
    # noblank_result_df = fit_all_no_blank(datadir, sample_sheet_file, result_file.replace(".csv", "_no_blank.csv"))
    blank_result_df = fit_all_manual_blank(datadir, sample_sheet_file, result_file.replace(".csv", "_manual_blank.csv"))
    return blank_result_df
    
###### Aggregate the results in each sample #####
def agg_fit_result(uvmelt_result_file, agg_result_file, sample_sheet_file,
                   single_curve_qc_criteria=None,
                   Tm_std_thresh=0.5, dH_std_thresh=1.5, clean=True, only_use_cooling=False):
    """
    Aggregates multiple heat-cool cycles for a given cuvette, e.g. sample level
    Not aggregated to sequence level!
    """
    uv = lambda x: processing.get_combined_param_bt(x, result_df.loc[x.index, x.name+'_std'])['p']
    uv_std = lambda x: processing.get_combined_param_bt(x, result_df.loc[x.index, x.name+'_std'])['e']
    
    if single_curve_qc_criteria is None:
        result_df = pd.read_csv(uvmelt_result_file, index_col=0).query('pass_qc')
    else:
        result_df = pd.read_csv(uvmelt_result_file, index_col=0)
        result_df['pass_qc'] = result_df.eval(single_curve_qc_criteria)
        result_df = result_df.query('pass_qc')
        sns.scatterplot(data=result_df.query('dH_std < 1e2 & Tm_std < 50 & dH < 0'), x='dH_std', y='Tm_std', hue='pass_qc')
            
    if only_use_cooling:
        result_df['isCooling'] = result_df.curve_name.apply(lambda x: 'Cooling' in x)
        result_df = result_df.query('isCooling')
    
    try:
        agg_stat = [uv, uv_std, len]
        result_agg_df = result_df.groupby(['SEQID', 'curve_date', 'curve_num']).agg(dict(dH=agg_stat, Tm=agg_stat)).reset_index()
        result_agg_df.columns = [f'{x[0]}_{x[1]}'.strip('_').replace('<lambda_0>', 'uv').replace('<lambda_1>', 'uv_std') for x in result_agg_df.columns]
    except:
        agg_stat = [np.median, np.std, len]
        result_agg_df = result_df.groupby(['SEQID', 'curve_date', 'curve_num']).agg(dict(dH=agg_stat, Tm=agg_stat)).reset_index()
        result_agg_df.columns = [f'{x[0]}_{x[1]}'.strip('_').replace('median', 'uv').replace('std', 'uv_std') for x in result_agg_df.columns]

    result_agg_df = result_agg_df.fillna(0)
    result_agg_df.rename(columns=dict(dH_len='n_curve'), inplace=True)
    result_agg_df.drop(columns='Tm_len', inplace=True)

    result_agg_df['dG_37_uv'] = get_dG(dH=result_agg_df.dH_uv, Tm=result_agg_df.Tm_uv, celsius=37)
    result_agg_df['dG_37_uv_std'] = get_dG_err(result_agg_df.dH_uv, result_agg_df.dH_uv_std, 
                                               result_agg_df.Tm_uv, result_agg_df.Tm_uv_std,
                                               celsius=37)
    result_agg_df['dS_uv'] = get_dS(dH=result_agg_df.dH_uv, Tm=result_agg_df.Tm_uv)
    result_agg_df['dS_uv_std'] = get_dS_err(result_agg_df.dH_uv, result_agg_df.dH_uv_std, 
                                            result_agg_df.Tm_uv, result_agg_df.Tm_uv_std)
    
    #TODO
    result_agg_df['is_hairpin'] = result_agg_df.SEQID.apply(lambda x: False if ('_' in x) or x.startswith('D') else True)
    result_agg_df['SEQID'] = result_agg_df.SEQID.apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    sample_sheet = read_sample_sheet(sample_sheet_file)
    for col in ['Na_mM', 'conc_uM', 'Purification']:
        result_agg_df[col] = lookup_sample_df(result_agg_df, sample_sheet, col)
        
    qc_criterion = 'Tm_uv_std < %f & dH_uv_std < %f' % (Tm_std_thresh, dH_std_thresh)
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    sns.scatterplot(data=result_agg_df,
                    x='dH_uv_std', y='Tm_uv_std', hue='n_curve',
                    palette='plasma', ax=ax[0])
    ax[0].axhline(Tm_std_thresh, linestyle='--', c='gray')
    ax[0].axvline(dH_std_thresh, linestyle='--', c='gray')
    ax[0].set_title('%.2f%% (%d / %d) variants passed QC' % 
                 (100 * result_agg_df.eval(qc_criterion).sum() / len(result_agg_df), result_agg_df.eval(qc_criterion).sum(), len(result_agg_df)))
    sns.scatterplot(data=result_df.query('dH_std < 1e2 & Tm_std <20'), x='dH_std', y='Tm_std', hue='rmse', ax=ax[1])
    ax[1].set_title('single curves used')
    sns.despine()
    save_fig(agg_result_file.replace('.csv', '.pdf'), fig)
    
    if clean:
        result_agg_df = result_agg_df.query(qc_criterion)
    else:
        result_agg_df['pass_qc'] = result_agg_df.eval(qc_criterion)
        
    result_agg_df.to_csv(agg_result_file)
    
    return result_agg_df
        
    
###### Adjust systematic offset ######
def prep_x_y_yerr(df, param, adjusted=True):
    if adjusted:
        x_suffix = '_adjusted'
    else:
        x_suffix = ''
        
    x = df[param+x_suffix].values.reshape(-1,1)
    y = df[param+'_uv'].values.reshape(-1,1)
    yerr = df[param+'_uv_std'].values.reshape(-1,1)
    yerr[yerr==0] = np.median(yerr)
    return add_intercept(x), y, yerr

def fit_param_offset(df, param, fix_slope=False, adjusted=True):
    x, y, yerr = prep_x_y_yerr(df, param, adjusted)
    ols = LinearRegressionSVD(param=param)
    if fix_slope:
        ols.coef_ = np.array([1.0, 0.0])
        ols.fit_intercept_only(x, y)
    else:
        ols.fit(x, y, yerr)
    return ols
    
def correct_param(df, correction_dict, param:str):
    """
    x - your data
    """
    x = add_intercept(df[param].values.reshape(-1,1))
    return x @ correction_dict[param].reshape(2,1)