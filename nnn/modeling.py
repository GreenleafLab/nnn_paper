import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from nnn import util, plotting, motif_fit
from sklearn.metrics import r2_score

matplotlib.rcParams['pdf.fonttype'] = 42
PNAMES = ['dH', 'Tm', 'dG_37']

def make_model_validation_df(val_data_df, **pred_kwargs):
    """
    Args:
        df - dataframe. Uses column names
            `RefSeq` and `TargetStruct`
        **pred_kwargs - passed to `get_model_prediction`
            set `sodium='varied'` to read from df
    Returns:
        val_res_df - dataframe. 
            Only has Index, RefSeq, TargetStruct, actual and predicted values
    """
    pred_df = get_model_prediction(df=val_data_df, **pred_kwargs)

    pred_columns = [c+'_pred' for c in pred_df.columns]
    pred_df.columns = pred_columns
    val_res_df = val_data_df[['RefSeq', 'TargetStruct'] + [x for x in PNAMES if x in val_data_df.columns]]
    val_res_df[pred_columns] = pred_df.values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    return val_res_df
    

def get_model_prediction(df=None, seq_list=None, struct_list=None, sodium=0.088,
                         model:str='nupack', model_param_file:str='./models/dna-nnn.json', 
                         model_kwargs=dict(),
                         append_df_suffix=None):
    """
    Predicts thermodynamic parameters from input sequences and (optionally) structures
    Input is either df or list of strings
    Args:
        df - dataframe. ignores seq_list and struct_list if given. Uses column names
            `RefSeq` and `TargetStruct`
        seq_list, struct_list - List[str]
        sodium - float, in M for constant sodium
            str:'varied' for varied sodium provided in `df`
        model - str, {'nupack', 'linear_regression', 'knn', 'gnn', 'unet'}
            all models other than unet requires structures (unet gives you paring prob mat)
        model_param_file - str, points to the file to use for the model
        model_kwargs - Dict, extra parameters passed to the model
            for nupack:
                if duplex, need 'DNA_conc:float or List[float]' in Molar
                ensemble - bool
        append_df_suffix - If given and there is an input df, append the prediction to the 
            df and return. Otherwise, only return the prediction values.
    Returns:
        If given a dataframe, returns a dataframe, either appended to the input or
        just the prediction. If given lists, returns result_df
    """
    # pnames = ['dH', 'Tm', 'dG_37', 'dS']
    # result_df = pd.DataFrame(columns=pnames)
    
    if isinstance(sodium, str):
        if sodium == 'varied':
            try:
                sodium = df.sodium
            except:
                raise "Must provide df with sodium column if sodium is set to varied!"
        else:
            raise ValueError("Unrecognized value %s for sodium"%sodium)
    
    if model in {'nupack', 'linear_regression', 'knn', 'gnn'}:
        # requires structures
        # == format input ==
        if df is not None:
            seq_list = df.RefSeq
            struct_list = df.TargetStruct
        else:
            assert len(seq_list) == struct_list
            
        # run model prediction
        result_df = globals()[f'run_{model}'](seq_list, struct_list, sodium, model_param_file, model_kwargs)
            
    elif model == 'unet':
        # TODO
        return result_df
    else:
        raise "model has to be one of {'nupack', 'linear_regression', 'knn', 'gnn', 'unet'}"
    
    # == format output ==
    if (df is not None) and (append_df_suffix is not None):
        # TODO append results to df
        result_df.index = df.index.copy()
        df = df.join(result_df, rsuffix=append_df_suffix)
        return df
    else:
        return result_df
        
        
def run_linear_regression(seq_list, struct_list, sodium, model_param_file=None, model_kwargs=None):

    result_df = pd.DataFrame(index=np.arange(len(seq_list)), columns=PNAMES)
    model_kwargs_copy = model_kwargs.copy()    
    '''sodium'''
    if not isinstance(sodium, float):
        varied_sodium = True
    else:
        varied_sodium = False
        
    '''lr_dict'''
    try:
        lr_dict = model_kwargs_copy.pop('lr_dict')
    except:
        raise ValueError("`model_kwargs` dict must have key `lr_dict`!")
        
    try:
        DNA_conc = model_kwargs_copy.pop('DNA_conc')
    except:
        DNA_conc = 1e-6
        Warning('DNA_conc set to default')
        
    for i in range(len(seq_list)):
        seq, struct = seq_list[i], struct_list[i]
        
        if varied_sodium:
            na = sodium[i]
        else:
            na = sodium
            
        # `seq`` could either be str or a list of str (e.g. duplex)
        if not '+' in struct:
            # Hairpins
            row_result_dict = motif_fit.pred_variant(seq, struct, na, 
                                                     lr_dict=lr_dict, feature_list_kwargs_dict=model_kwargs_copy)
        else:
            # > 1 strand, Duplexes
                
                
            if isinstance(DNA_conc, float):
                dna_conc = DNA_conc
            else:
                dna_conc = DNA_conc[i]
                
            feature_list_kwargs = model_kwargs
            row_result_dict = motif_fit.pred_variant(seq, struct, na, DNA_conc=dna_conc, 
                                                 lr_dict=lr_dict, feature_list_kwargs_dict=model_kwargs)
            
        result_df.iloc[i,:] = {k:v for k,v in row_result_dict.items() if k in PNAMES}
    
    return result_df
        
        
def run_nupack(seq_list, struct_list, sodium, model_param_file, model_kwargs):

    result_df = pd.DataFrame(index=np.arange(len(seq_list)), columns=PNAMES)
    
    # sodium
    if not isinstance(sodium, float):
        varied_sodium = True
    else:
        varied_sodium = False
        
    # structure or ensemble energy
    ensemble = False
    if 'ensemble' in model_kwargs:
        if model_kwargs['ensemble']:
            ensemble = True
    
    for i in range(len(seq_list)):
        seq, struct = seq_list[i], struct_list[i]
        
        if varied_sodium:
            na = sodium[i]
        else:
            na = sodium
        
        # `seq`` could either be str or a list of str (e.g. duplex)
        if not '+' in struct:
            # Hairpins
            row_result_dict = util.get_nupack_dH_dS_Tm_dG_37(
                seq, struct, sodium=na, ensemble=ensemble,
                return_dict=True, param_set=model_param_file)
        else:
            # > 1 strand, Duplexes
            
            '''DNA_conc'''
            try:
                DNA_conc=model_kwargs['DNA_conc']
            except:
                raise ValueError("`model_kwargs` dict must have key `DNA_conc`!")
                
            if isinstance(DNA_conc, float):
                dna_conc = DNA_conc
            else:
                dna_conc = DNA_conc[i]
                
            Tm = util.calculate_tm(seq, struct,
                 sodium=1.0, DNA_conc=dna_conc, param_set=model_param_file)
            row_result_dict = dict(Tm=util.get_Na_adjusted_Tm(Tm, None, util.get_GC_content(''.join(seq)), Na=na))
            
        result_df.iloc[i,:] = {k:v for k,v in row_result_dict.items() if k in PNAMES}
    
    return result_df
    
def get_metric_dict(val_result_df, param):
    metric = defaultdict()
    
    val_result_df_no_nan = val_result_df.dropna(subset=[param+'_pred'])
    val_result_df_no_nan = val_result_df_no_nan.loc[np.abs(val_result_df_no_nan[param+'_pred']) < 400]
    y = val_result_df_no_nan[param].values
    y_pred = val_result_df_no_nan[param+'_pred'].values
    bias = np.nanmean(y_pred) - np.nanmean(y)
    y_pred_adj = y_pred - bias
    
    metric['bias'] = bias
    metric['corr'] = util.pearson_r(y, y_pred)
    metric['rmse'] = util.rmse(y, y_pred)
    metric['adjusted_rmse'] = util.rmse(y, y_pred_adj)
    metric['mae'] = util.mae(y, y_pred)
    metric['adjusted_mae'] = util.mae(y, y_pred_adj)
    metric['r2'] = r2_score(y, y_pred_adj)
        
    return metric

    
def plot_validation_result(val_result_df, param, ax=None, 
                           color_by_density=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2,2))
    matplotlib.rc('axes',edgecolor='k', linewidth=.5)    
    # kwargs = dict(show_cbar=False, lim=lim, color_by_density=color_by_density)
    plotting.plot_colored_scatter_comparison(data=val_result_df, x=param, y=param+'_pred', 
                                             ax=ax, show_cbar=False, color_by_density=color_by_density,
                                             **kwargs)
    if param == 'dG_37':
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(colors='k', width=.5)
    ax.set_aspect('equal')
    
    metric = get_metric_dict(val_result_df, param)
    ax.set_title('Corr. = %.2f\nMAE = %.2f' % (metric['corr'], metric['mae']))
    plt.tight_layout()
    
    
def plot_validation_result_all_params(val_result_df, **kwargs):    
    _, ax = plt.subplots(1,3,figsize=(6,2))
    for i,pname in enumerate(PNAMES):
        plot_validation_result(val_result_df, param=pname, ax=ax[i], **kwargs)
        
def plot_metric_bar(metric_dict, metric_name=None, metric_name_list=None, 
                    abs_value=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.25,3.5))
    if metric_name is None and metric_name_list is None:
        "plot all metric"
        tmp = pd.DataFrame(data=metric_dict).reset_index(names='metric').melt(id_vars=['metric'])
        if abs_value:
            tmp['value'] = np.abs(tmp['value'])
        sns.barplot(data=tmp, hue='variable', y='value', x='metric', ax=ax,
                    width=.4, palette='viridis', edgecolor='k', linewidth=.5)
        
    elif metric_name_list is not None:
        tmp = pd.DataFrame(data=metric_dict).loc[metric_name_list,:].reset_index(names='metric').melt(id_vars=['metric'])
        if abs_value:
            tmp['value'] = np.abs(tmp['value'])
        sns.barplot(data=tmp, hue='variable', y='value', x='metric', ax=ax,
                    width=.4, palette='viridis', edgecolor='k', linewidth=.5)
        
    elif metric_name is not None:
        tmp_row = pd.DataFrame(data=metric_dict).loc[metric_name,:]
        ax.bar(x=tmp_row.index.tolist(), height=tmp_row.values,
                width=.3, color=np.array([193,167,47])/256, edgecolor='k', linewidth=.5)
        # ax.set_xlim([-.5,1.5])
        ax.set_title(metric_name)
    util.beutify(ax, y_locator=.2)
    # sns.despine()
    # ax.tick_params(colors='k', width=.5)