import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple
from datetime import datetime
import os
import itertools

from . import fileio, util, plotting

param_name_dict = {'dH':'dH', 'dG':'dG_37'}

""" Helper Functions """
def add_2_dict_val(mydict, value):
    for key in mydict:
        mydict[key] += value
    return mydict
    
def get_dict_mean_val(mydict):
    return np.nanmean(list(mydict.values()))

def center_new_param(old_dict, new_dict, verbose=False):
    """
    Can only center one layer deep. If nested dictionary, need to call
    this function separately for each category you want to center
    Returns:
        new_dict - centered `new_dict`
    """
    try:
        target_mean = np.mean([old_dict[k] for k in new_dict if k in old_dict])
        fitted_mean = get_dict_mean_val(new_dict)
            
        new_dict = add_2_dict_val(new_dict, -1 * fitted_mean)
        new_dict = add_2_dict_val(new_dict, target_mean)

        if verbose:
            print('\tMean value of old_dict is %.3f' % target_mean)
            print('\tMean value of fitted new_dict is %.3f' % fitted_mean)
            print('\tMean value of centered new_dict is %.3f\n' % get_dict_mean_val(new_dict))
    except:
        print('Nothing happened.\n')
    return new_dict
    
""" Formatting Functions """

def update_template_dict(template_dict, coef_dict):
    """ 
    Overwrite tempate_dict with values in coef_dict
    Up to 2 levels deep as in the parameter file
    """
    ignored_keys = ['name', 'type', 'material', 'references', 'time_generated']
    new_dict = template_dict.copy()
    for param in template_dict:
        if not param in ignored_keys:
            if isinstance(template_dict[param], List) and param in coef_dict:
                new_dict[param] = coef_dict[param]
            elif isinstance(template_dict[param], dict):
                for key in template_dict[param]:
                    if key in coef_dict[param] and not (key in ignored_keys):                    
                        if isinstance(coef_dict[param][key], dict):
                            for p,v in coef_dict[param][key].items():
                                new_dict[param][key][p] = v
                        else:
                            new_dict[param][key] = coef_dict[param][key]
                
    return new_dict

        

def coef_df_2_dict(coef_df, template_dict=None,
                   center_new_parameters=False):
    """
    Convert lr.coef_df to a NUPACK style dictionary
    without modifying the contents
    """
    coef_dict = defaultdict(dict)
    for _,row in coef_df.iterrows():
        pclass, pname = row.name.split('#')
        coef_dict[pclass][pname] = row.values[0]
          
    # Convet numerical keys & values to lists
    for key, sub_dict in coef_dict.items():
        if isinstance(sub_dict, dict):
            inds = list(sub_dict.keys())
            if inds[0].isdigit():
                inds = [int(x) for x in inds]
                if template_dict is None:
                    new_values = np.zeros(np.max(inds))
                else:
                    new_values = template_dict[key].copy()
                    
                for ind, value in sub_dict.items():
                    new_values[int(ind)-1] = value
                coef_dict[key] = list(new_values)
    
    # Overwrite tempate_dict
    if not template_dict is None:
        if center_new_parameters:
            print('Centering new parameters...')
            for p_group in coef_dict:
                if p_group in template_dict:
                    print('group', p_group)
                    coef_dict[p_group] = center_new_param(old_dict=template_dict[p_group], 
                                             new_dict=coef_dict[p_group], verbose=True)
            
        new_dict = update_template_dict(template_dict, coef_dict)
    else:
        new_dict = coef_dict
            
    return new_dict
    
    
def coef_dict_2_df(coef_dict):
    """
    NUPACK style dict to lr.coef_df style
    Args:
        coef_dict - param_set_dict['dH'] level
    """
    flat_coef_dict = defaultdict()
    for pclass in coef_dict:
        if not pclass in ['name', 'type', 'material', 'references', 'time_generated']:
            if isinstance(coef_dict[pclass], list):
                for i,value in enumerate(coef_dict[pclass]):
                    coef_name = '%s#%d' % (pclass, i+1)
                    flat_coef_dict[coef_name] = value
            elif isinstance(coef_dict[pclass], float):
                flat_coef_dict[pclass] = coef_dict[pclass]
            else:
                for pname in coef_dict[pclass]:
                    try:
                        coef_name = pclass + '#' + pname
                        flat_coef_dict[coef_name] = coef_dict[pclass][pname]
                    except:
                        print(coef_name)
        else:
            pass
            
    return pd.DataFrame(index=['value'], data=flat_coef_dict).T
    
    
def get_fixed_params(param_set_template_file:str, fixed_pclass:List[str],
                     features_not_fixed:List[str]=None, return_full_coef_df=False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gets the params in `param_set_template_file` that starts with a str in `fixed_pclass`.
    Returns:
        fixed_coef_df - dataframe, contains the values for the fixed parameters
        return_full_coef_df - if True, return all coef. Useful for regularized fitting with prior
    """
    param_set_dict = fileio.read_json(param_set_template_file)
    
    ori_coef_df = pd.concat((coef_dict_2_df(param_set_dict['dH']), coef_dict_2_df(param_set_dict['dG'])), axis=1)
    ori_coef_df.columns = ['dH', 'dG']

    if return_full_coef_df:
        fixed_coef_df = ori_coef_df.copy()
    else:
        fixed_coef_df = ori_coef_df.loc[[x for x in ori_coef_df.index if x.split('#')[0] in fixed_pclass]]
        if features_not_fixed is not None:
            fixed_coef_df.drop(labels=features_not_fixed, inplace=True)
        
    fixed_coef_df.fillna(0, inplace=True)
    fixed_feature_names = fixed_coef_df.index.tolist()
    
    return fixed_coef_df, fixed_feature_names
    
""" Convert parameters between different models """

def get_hairpin_seq_df(lr:util.LinearRegressionSVD, param:str, loop_len:int=3) -> pd.DataFrame:
    """
    Converts `feature_list.get_feature_list()` style hairpin parameters 
    to nupack hairpin_triloop or hairpin_tetraloop style parameters
    """
    loop_mid_param = lr.coef_df.loc[[x for x in lr.coef_df.index if x.endswith('_'+'.'*loop_len)]]
    hairpinmm_param = lr.coef_df.loc[[x for x in lr.coef_df.index if (x.endswith('_(.+.)') and (not x.startswith('x')))]]

    full_loop_list = []
    param_list = []
    for loop_mid in loop_mid_param.index:
        loop_mid_seq = loop_mid.split('_')[0]
        for hairpinmm in hairpinmm_param.index:
            # hairpinmm - 'NN+NN_(.+.)'
            nt1, nt2 = hairpinmm[1], hairpinmm[3]
            hairpinmm_seq = hairpinmm.split('_')[0]
            if nt1 == loop_mid_seq[0] and nt2 == loop_mid_seq[-1]:
                full_triloop = hairpinmm_seq[0] + loop_mid_seq + hairpinmm_seq[-1]
                full_loop_list.append(full_triloop)
                param_list.append(loop_mid_param.loc[loop_mid][0] + hairpinmm_param.loc[hairpinmm][0])

    loop_df = pd.DataFrame(index = full_loop_list, data=param_list, columns=[param])
    return loop_df
    
    
def get_hairpin_mismatch(lr:util.LinearRegressionSVD):
    """
    Formats `feature_list.get_feature_list()` style hairpin_mismatch `NN+NN_(.+.)`
    to NUPACK style
    Args:
        
    Returns:
        Dict, equivalent to `hairpin_dict['dH']['hairpin_mismatch']`
    """
    def convert_mm_name(x):
        seq = x.split('_')[0]
        return seq[-2:] + seq[:2]
        
    hairpinmm_param = lr.coef_df.loc[[x for x in lr.coef_df.index if (x.endswith('_(.+.)') and (not x.startswith('x')))]]
    hairpinmm_param.index = [convert_mm_name(x) for x in hairpinmm_param.index]
    param = hairpinmm_param.keys()[0]
    return hairpinmm_param.to_dict()[param]
    
    
def get_adjusted_triloop_terminal_penalty(hairpin_triloop_dict, terminal_penalty_dict):
    """
    Called by `lr_dict_2_nupack_json()`
    """
    for key,value in hairpin_triloop_dict.items():
        closing_pair = key[0] + key[-1]
        terminal_penalty = terminal_penalty_dict[closing_pair]
        hairpin_triloop_dict[key] = value - terminal_penalty
    
    return hairpin_triloop_dict
    
def get_hpmm_from_hptetra_hptri(p_dict):
    """
    Extracts hairpin mismatch parameters from tetraloop and triloop parameters
    Needed because the hairpin mismatch parameters fitted in the NUPACK-compatible model
    is not the same as that in the NUPACK file: they require an extra loop_mis parameter
    Very confusing.
    Args:
        p_dict - param_set_dict['dH'] or param_set_dict['dG'], with p_dict['hairpin_tetraloop']
            and p_dict['hairpin_triloop']
    Returns:
        hairpin_mismatch - dict, equivalent to p_dict['hairpin_mismatch']
    """
    hptetra = p_dict['hairpin_tetraloop']           
    hptri = p_dict['hairpin_triloop']
    
    
    
def lr_dict_2_nupack_json(lr_dict:util.LinearRegressionSVD, template_file:str, out_file:str, 
                          lr_step:str='full', center_new_parameters=False,
                          adjust_triloop_terminal_penalty:bool=True,
                          extract_hairpin_mismatch:bool=False, comment=''):
    """
    Formats and saves the parameters from the final regression object to 
    NUPACK json file
    Args:
        lr_dict - Dict, keys 'dH' and 'dG', 
            values are instances of the LinearRegressionSVD() class
        lr_step - str, {'hairpin', 'full'}. If hairpin, only update the hairpin seq params.
            hairpin is for two-step fitting (archived)
        center_parameter - bool, if True, center the newly fitted parameters to the template category
        adjust_trilop_terminal_penalty - bool, only used when lr_step = 'hairpin', adjust the 
            terminal penalty off the hairpin_triloop parameters
        extract_hairpin_mismatch - bool, if True, fit hairpin_mismatch parameters from triloop 
            and tetraloop parameters
    """
    param_name_dict = {'dH':'dH', 'dG':'dG_37'}
    
    print('\nTemplate file:', template_file)
    ori_param_set_dict = fileio.read_json(template_file)
    param_set_dict = defaultdict()
    
    if lr_step == 'full':
        # Only centering new parameters here once as the ones in populated loopup tables are already 
        # built on the centered ones
        # but hairpin_triloop and hairpin_tetraloop contains hairpin_loop_mid, which is not in the original ones
        # need to center these two once more after populating the lookup table
        for p in param_name_dict:
            param_set_dict[p] = coef_df_2_dict(lr_dict[p].coef_df, template_dict=ori_param_set_dict[p],
                                               center_new_parameters=center_new_parameters)
            if 'intercept#intercept' in lr_dict[p].coef_df.index:
                param_set_dict[p]['intercept'] = lr_dict[p].coef_df.at['intercept#intercept', lr_dict[p].coef_df.columns[0]]

        param_set_dict = update_template_dict(ori_param_set_dict, param_set_dict)
        
        ### Populate the lookup tables ###
        for p in param_name_dict:
            # the original uncentered parameters
            # `param_set_dict` is centered
            coef_p_dict = coef_df_2_dict(lr_dict[p].coef_df)
        
            # `interior_n1_n2` (mismatches)
            for n1, n2 in [(1,1), (1,2), (2,2)]:
                interior_name = 'interior_%d_%d' % (n1, n2)
                for seq in param_set_dict[p][interior_name]:
                    seq1, seq2 = seq[:n1+2], seq[n1+2:]
                    mm1 = seq2[-2] + seq2[-1] + seq1[0] + seq1[1]
                    mm2 = seq1[-2] + seq1[-1] + seq2[0] + seq2[1]
                    new_value = param_set_dict[p]['interior_size'][n1+n2-1] + \
                        param_set_dict[p]['interior_mismatch'][mm1] + \
                        param_set_dict[p]['interior_mismatch'][mm2]
                    if n1 == 1 and n2 == 1:
                        mm_stacks = seq1[0] + seq1[-1] + seq2[0] + seq2[-1]
                        try:
                            new_value += coef_p_dict['interior_mismatch_stacks'][mm_stacks]
                        except:
                            pass
                    param_set_dict[p][interior_name][seq] = new_value
                    
            # hairpin loops
            loop_size_dict = dict(triloop=3, tetraloop=4)
            for loop_size in loop_size_dict:
                hairpin_name = 'hairpin_' + loop_size
                
                loop_seqs = [''.join(x) + util.rcompliment(x[0]) # iterate all possible loop sequences
                              for x in itertools.product(list('ATCG'), repeat=loop_size_dict[loop_size] + 1)]
                
                # calculate values for each full hairpin loop sequuence
                for seq in loop_seqs:
                    hp_value = calc_hp_value(template_file, param_set_dict, p, coef_p_dict, seq, loop_size)
                    param_set_dict[p][hairpin_name][seq] = hp_value
                    
                # center the calculated hairpin_tetraloop and hairpin_triloop again
                if center_new_param:
                    print(f'\nCentering {p} of {hairpin_name} to {template_file}')
                    ori_param_set_dict = fileio.read_json(template_file)
                    param_set_dict[p][hairpin_name] = center_new_param(
                        old_dict=ori_param_set_dict[p][hairpin_name], 
                        new_dict=param_set_dict[p][hairpin_name],
                        verbose=True)
                    
                    
    elif lr_step == 'hairpin':
        hairpin_dict = {'dH': dict(hairpin_triloop=None, hairpin_tetraloop=None),
                        'dG': dict(hairpin_triloop=None, hairpin_tetraloop=None)}
        
        for p in param_name_dict:
            if extract_hairpin_mismatch:
                # this only happens when individual parameters for each triloop & tetraloop were fitted
                # semi-archived
                mm_dict = get_hairpin_mismatch(lr_dict[p])
                # hairpin_dict[p]['hairpin_mismatch'] = center_new_param(ori_param_set_dict[p]['hairpin_mismatch'], new_dict=mm_dict)

            hairpin_dict[p]['hairpin_tetraloop'] = get_hairpin_seq_df(lr_dict[p], p, loop_len=4).to_dict()[p]
            hairpin_dict[p]['hairpin_triloop'] = get_hairpin_seq_df(lr_dict[p], p, loop_len=3).to_dict()[p]
            
            if adjust_triloop_terminal_penalty:
                hairpin_dict[p]['hairpin_triloop'] = get_adjusted_triloop_terminal_penalty(
                    hairpin_dict[p]['hairpin_triloop'], ori_param_set_dict[p]['terminal_penalty'])
                
            # for hairpin_p in ['hairpin_triloop', 'hairpin_tetraloop']:
            #     hairpin_dict[p][hairpin_p] = center_new_param(ori_param_set_dict[p][hairpin_p], 
            #                                                         hairpin_dict[p][hairpin_p])
                              
        param_set_dict = update_template_dict(ori_param_set_dict, hairpin_dict)
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    param_set_dict['time_generated'] = current_time
    param_set_dict['comment'] = comment
    param_set_dict['name'] = os.path.split(out_file)[1].replace('.json', '')
    
    fileio.write_json(param_set_dict, out_file)

def calc_hp_value(template_file, param_set_dict, p, coef_p_dict, seq, loop_size):
    hp_mm = seq[-2:] + seq[:2]
    loop_mid = seq[2:-2]
    # `hairpin_loop_mid` is an intermediate parameter not in the final file
    try:
        hp_value = coef_p_dict['hairpin_loop_mid'][loop_mid]
    except:
        lr_dict_template = fileio.read_pickle(template_file.replace('.json', '_lr_dict.pkl'))
        hp_value = lr_dict_template[p].coef_df.loc['hairpin_loop_mid#'+loop_mid].values[0]                       
        
    # this happens for both tri and tetra
    try:
        hp_value += coef_p_dict[p]['hairpin_mismatch'][hp_mm] 
    except:
        # use the parameter copied from the template
        hp_value += param_set_dict[p]['hairpin_mismatch'][hp_mm]
        
                    
    # NUPACK adds hairpin_mismatch on top of hairpin_tetraloop but not for triloop
    # also takes out terminal penalty
    if loop_size == 'triloop':
        hp_value -= param_set_dict[p]['terminal_penalty'][seq[-1]+seq[0]]
        
    return hp_value


""" Plotting functions """

def plot_mupack_nupack(data, x_suffix, param, lim, color_by_density=False):
    fig, ax = plt.subplots(1,2,figsize=(4,2))
    kwargs = dict(show_cbar=False, lim=lim, color_by_density=color_by_density)
    plotting.plot_colored_scatter_comparison(data=data, x=param+x_suffix, y=param+'_MUPACK', 
                                             ax=ax[0], **kwargs)
    plotting.plot_colored_scatter_comparison(data=data, x=param+x_suffix, y=param+'_NUPACK_salt_corrected', 
                                             ax=ax[1], **kwargs)
    mae = defaultdict()
    mae['new'] = util.mae(data[param+x_suffix], data[param+'_MUPACK'])
    mae['original'] = util.mae(data[param+x_suffix], data[param+'_NUPACK_salt_corrected'])
    ax[0].set_title('MAE = %.2f' % mae['new'])
    ax[1].set_title('MAE = %.2f' % mae['original'])
    plt.tight_layout()


